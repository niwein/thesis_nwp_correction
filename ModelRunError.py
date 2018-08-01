import json
import os
import pickle as pkl
import sys
from time import time, strftime, gmtime
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

import model.Baseline as Baseline
from utils import ModelUtils
from utils.Logger import Logger
from utils.data import DataUtils, DataLoaders

# main method for training a neural network to predict COSMO-1 error
def runModel(config, data_dictionary, data_statistics, train_test_folds):
    program_start_time = time()

    # assign all program arguments to local variables
    with open(config['model']['path']) as handle:
        ModelDict = json.loads(handle.read())

    # check if station and grid time invariant features should be used and set the list of desired parameters
    if not ('grid_time_invariant' in ModelDict and ModelDict['grid_time_invariant']): config['grid_time_invariant_parameters'] =[]
    if not ('station_time_invariant' in ModelDict and ModelDict['station_time_invariant']): config['station_parameters'] = []

    # update general static model information
    experiment_info = config
    experiment_info['model'] = ModelDict
    experiment_info['code_commit'] = ModelUtils.get_git_revision_short_hash()


    # if needed, load time invariant features
    with open("%s/%s/grid_size_%s/time_invariant_data_per_station.pkl" % (config['input_source'], config['preprocessing'], config['original_grid_size']), "rb") as input_file:
        time_invarian_data = pkl.load(input_file)


    # initialize feature scaling function for each feature
    featureScaleFunctions = DataUtils.getFeatureScaleFunctions(ModelUtils.ParamNormalizationDict, data_statistics)

    # get optimizer config
    optimizer_config = config['optimizer']

    # generate output path for experiment information
    setting_string = '%s_grid_%s_bs_%s_tf_%s_optim_%s_lr_%s_sl_%s' % (
        config['model']['name'], config['grid_size'], config['batch_size'], config['test_fraction'], optimizer_config['algorithm'], optimizer_config['learning_rate'], config['slice_size'])
    output_path = '%s/%s' % (config['experiment_path'], setting_string)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # time for the set up until first run
    experiment_info['set_up_time'] = time() - program_start_time
    print('[Time]: Set-up %s' % strftime("%H:%M:%S", gmtime(experiment_info['set_up_time'])))
    sys.stdout.flush()

    # initialize statistics
    error_statistics = None
    run_times = None
    skip_statistics = None
    if 'per_station_rmse' in config:
        error_per_station_statistics = None

    # keep used learning rates
    experiment_info['scheduled_learning_rates'] = []

    # cross validation
    for run in range(config['runs']):
        # logger  for tensorboardX
        train_logger = Logger(output_path + '/logs/run_%s/train' % run)
        test_logger = Logger(output_path + '/logs/run_%s/test' % run)

        print('[Run %s] Cross-validation test fold %s' % (str(run + 1), str(run + 1)))

        # take the right preprocessed train/test data set for the current run
        train_fold, test_fold = train_test_folds[run]

        # initialize best epoch test error
        best_epoch_test_rmse = float("inf")


        # initialize train and test dataloaders
        trainset = DataLoaders.ErrorPredictionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=train_fold,
            featureScaling=featureScaleFunctions,
            time_invariant_data=time_invarian_data)
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                                 num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        testset = DataLoaders.ErrorPredictionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=test_fold,
            featureScaling=featureScaleFunctions,
            time_invariant_data=time_invarian_data)
        testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        # initialize network, optimizer and loss function
        net = Baseline.model_factory(ModelDict, trainset.n_parameters, trainset.n_grid_time_invariant_parameters,
                                     config['grid_size'], config['prediction_times'])
        # store class name
        experiment_info['model_class'] = net.__class__.__name__

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

        if torch.cuda.is_available():
            net.cuda()

        # load number of train and test samples
        n_train_samples, n_test_samples = len(train_fold), len(test_fold)

        optimizer, scheduler = ModelUtils.initializeOptimizer(optimizer_config, net)
        criterion = nn.MSELoss()

        # keep number of processed smaples over all epochs for tensorboard
        processed_train_samples_global = 0
        processed_test_samples_global = 0

        # start learning
        for epoch in range(config['epochs']):
            epoch_train_time = np.zeros((5,))
            epoch_start_time = time()
            print('Epoch: ' + str(epoch + 1) + '\n------------------------------------------------------------')

            # adapt learning rate and store information in experiment attributes
            if scheduler is not None:
                scheduler.step()
                if run == 0: experiment_info['scheduled_learning_rates'] += scheduler.get_lr()
                print('Using learning rate %s' % str(scheduler.get_lr()))

            # TRAINING
            # initialize variables for epoch statistics
            LABELS, MODELoutputs, COSMOoutputs = None, None, None
            processed_train_samples = 0
            net.train(True)

            train_start_time = time()
            # loop over complete train set
            for i, data in enumerate(trainloader, 0):
                time_start = time()
                try:
                    # get training batch, e.g. label, cosmo-1 output and time inv. features for station
                    DATA = data
                    # DATA has only length 4 if we do not use the station time invariant features
                    if len(DATA) == 4:
                        Blabel, Bip2d, BTimeData, init_station_temp = DATA
                        station_time_inv_input = None
                    elif len(DATA) == 5:
                        Blabel, Bip2d, BTimeData, StationTimeInv, init_station_temp = DATA
                        station_time_inv_input = ModelUtils.getVariable(StationTimeInv).float()
                    else:
                        raise Exception('Unknown data format for training...')
                    input = ModelUtils.getVariable(Bip2d).float()
                    time_data = ModelUtils.getVariable(BTimeData).float()
                    target = ModelUtils.getVariable(Blabel).float()
                    cosmo_out = ModelUtils.getVariable(init_station_temp[2]).float()

                except TypeError:
                    # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                    # collate_fn would return an empty list
                    print('Value error...')
                    continue
                time_after_data_preparation = time()

                processed_train_samples += len(Blabel)

                optimizer.zero_grad()
                out = net(input, time_data, station_time_inv_input)
                time_after_forward_pass = time()
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                time_after_backward_pass = time()

                if LABELS is None:
                    LABELS = init_station_temp[3].data[...,None]
                    MODELoutputs = cosmo_out.data - out.data
                    COSMOoutputs = init_station_temp[2].data
                else:
                    LABELS = np.vstack((LABELS, init_station_temp[3].data[...,None]))
                    MODELoutputs = np.vstack((MODELoutputs, cosmo_out.data - out.data))
                    COSMOoutputs = np.vstack((COSMOoutputs, init_station_temp[2].data))

                time_after_label_stack = time()

                if (i + 1) % 64 == 0:

                    print('Sample: %s \t Loss: %s' % (processed_train_samples, float(np.sqrt(loss.data))))

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        setting_string: np.sqrt(loss.item()),
                    }

                    for tag, value in info.items():
                        train_logger.scalar_summary(tag, value, processed_train_samples_global + processed_train_samples)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        train_logger.histo_summary(tag, ModelUtils.to_np(value), i + 1)
                        train_logger.histo_summary(tag + '/grad', ModelUtils.to_np(value.grad), i + 1)

                    epoch_train_time += np.array((time_start - time_end,
                                                  time_after_data_preparation - time_start,
                                                  time_after_forward_pass - time_after_data_preparation,
                                                  time_after_backward_pass - time_after_forward_pass,
                                                  time_after_label_stack - time_after_backward_pass))

                time_end = time()

            # calculate error statistic of current epoch
            diff_model = MODELoutputs - LABELS
            diff_cosmo = COSMOoutputs - LABELS
            epoch_train_rmse_model = np.apply_along_axis(func1d=ModelUtils.rmse, arr=diff_model, axis=0)
            epoch_train_rmse_cosmo = np.apply_along_axis(func1d=ModelUtils.rmse, arr=diff_cosmo, axis=0)


            # update global processed samples
            processed_train_samples_global += processed_train_samples

            if np.isnan(epoch_train_rmse_model).any():
                print("Learning rate too large resulted in NaN-error while training. Stopped training...")
                return
            # print epoch training times
            print('Timing: Waiting on data=%s, Data Preparation=%s,'
                  'Forward Pass=%s, Backward Pass=%s, Data Stacking=%s' % tuple(list(epoch_train_time / len(epoch_train_time))))

            # RMSE of epoch
            print('Train/test statistic for epoch: %s' % str(epoch + 1))
            print('Train RMSE COSMO: ' , ", ".join(["T=%s: %s" % (idx, epoch_train_rmse_cosmo[idx]) for idx in range(len(epoch_train_rmse_cosmo))]))
            print('Train RMSE Model: ' , ", ".join(["T=%s: %s" % (idx, epoch_train_rmse_model[idx]) for idx in range(len(epoch_train_rmse_model))]))
            sys.stdout.flush()

            train_time = time() - train_start_time

            # TESTING
            test_start_time = time()

            LABELS, MODELoutputs, COSMOoutputs, STATION = None, None, None, None
            processed_test_samples = 0
            net.eval()
            for i, data in enumerate(testloader, 0):
                try:
                    # get training batch, e.g. label, cosmo-1 output and time inv. features for station
                    DATA = data
                    # DATA has only length 4 if we do not use the station time invariant features
                    if len(DATA) == 4:
                        Blabel, Bip2d, BTimeData, init_station_temp = DATA
                        station_time_inv_input = None
                    elif len(DATA) == 5:
                        Blabel, Bip2d, BTimeData, StationTimeInv, init_station_temp = DATA
                        station_time_inv_input = ModelUtils.getVariable(StationTimeInv).float()
                    else:
                        raise Exception('Unknown data format for training...')
                    input = ModelUtils.getVariable(Bip2d).float()
                    time_data = ModelUtils.getVariable(BTimeData).float()
                    target = ModelUtils.getVariable(Blabel).float()
                    cosmo_out = ModelUtils.getVariable(init_station_temp[2]).float()

                except TypeError:
                    # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                    # collate_fn would return an empty list
                    print('Value error...')
                    continue

                processed_test_samples += len(Blabel)

                out = net(input, time_data, station_time_inv_input)
                loss = criterion(out, target)

                if LABELS is None:
                    LABELS = init_station_temp[3].data[...,None]
                    MODELoutputs = cosmo_out.data - out.data
                    COSMOoutputs = init_station_temp[2].data
                    STATION = init_station_temp[1].data
                else:
                    LABELS = np.vstack((LABELS, init_station_temp[3].data[...,None]))
                    MODELoutputs = np.vstack((MODELoutputs, cosmo_out.data - out.data))
                    COSMOoutputs = np.vstack((COSMOoutputs, init_station_temp[2].data))
                    STATION = np.hstack((STATION, init_station_temp[1].data))

                if i % 16:
                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        setting_string: np.sqrt(loss.item()),
                    }

                    for tag, value in info.items():
                        test_logger.scalar_summary(tag, value, processed_test_samples_global + processed_test_samples)

            # calculate error statistic of current epoch
            diff_model = MODELoutputs - LABELS
            diff_cosmo = COSMOoutputs - LABELS

            # rmse
            epoch_test_rmse_model = np.apply_along_axis(func1d=ModelUtils.rmse, arr=diff_model, axis=0)
            epoch_test_rmse_cosmo = np.apply_along_axis(func1d=ModelUtils.rmse, arr=diff_cosmo, axis=0)
            overall_test_rmse_model = ModelUtils.rmse(diff_model)
            overall_test_rmse_cosmo = ModelUtils.rmse(diff_cosmo)

            # mae
            epoch_test_mae_model = np.apply_along_axis(func1d=ModelUtils.mae, arr=diff_model, axis=0)
            epoch_test_mae_cosmo = np.apply_along_axis(func1d=ModelUtils.mae, arr=diff_cosmo, axis=0)
            overall_test_mae_model = ModelUtils.mae(diff_model)
            overall_test_mae_cosmo = ModelUtils.mae(diff_cosmo)

            # calculate per station rmse if desired (especially for K-fold station generalization experiment
            if "per_station_rmse" in config:
                max_station_id = 1435

                squared_errors_per_epoch = np.array((np.square(diff_model), np.square(diff_cosmo))).squeeze()

                # the highest index of data is 1435, thus we expect at least 1435 entries, which we can access by
                # station id
                test_samples_per_station = np.bincount(STATION, minlength=max_station_id+1)
                model_squared_error_per_station = np.bincount(STATION, weights=squared_errors_per_epoch[0], minlength=max_station_id+1)
                cosmo_squared_error_per_station = np.bincount(STATION, weights=squared_errors_per_epoch[1], minlength=max_station_id+1)

                # set division by zero/NaN warning to 'ignore'
                np.seterr(divide='ignore', invalid='ignore')

                # calculate rmse per station
                rmse_per_station = np.vstack((np.sqrt(np.divide(model_squared_error_per_station, test_samples_per_station)),
                                              np.sqrt(np.divide(cosmo_squared_error_per_station, test_samples_per_station)))).T

                # set division by zero/NaN warning to 'warn'
                np.seterr(divide='warn', invalid='warn')






            # update global processed samples
            processed_test_samples_global += processed_test_samples

            # RMSE of epoch
            print('Test RMSE COSMO: ', ", ".join(
                ["T=%s: %s" % (idx, epoch_test_rmse_cosmo[idx]) for idx in range(len(epoch_test_rmse_cosmo))]),
                  " (Overall: %s" % overall_test_rmse_cosmo)
            print('Test RMSE Model: ' , ", ".join(["T=%s: %s" % (idx, epoch_test_rmse_model[idx]) for idx in range(len(epoch_test_rmse_model))]),
                  " (Overall: %s" % overall_test_rmse_model)
            # mae of epoch
            print('Test MAE COSMO: ', ", ".join(
                ["T=%s: %s" % (idx, epoch_test_mae_cosmo[idx]) for idx in range(len(epoch_test_mae_cosmo))]),
                  " (Overall: %s" % overall_test_mae_cosmo)
            print('Test MAE Model: ' , ", ".join(["T=%s: %s" % (idx, epoch_test_mae_model[idx]) for idx in range(len(epoch_test_mae_model))]),
                  " (Overall: %s" % overall_test_mae_model)

            sys.stdout.flush()

            test_time = time() - test_start_time

            # time for epoch
            epoch_time = time() - epoch_start_time

            # update error statistics
            error_statistics = ModelUtils.updateErrorStatistic(error_statistics,
                                                               np.array([epoch_train_rmse_model, epoch_test_rmse_model])[None, None, ...],
                                                               run, epoch, config['prediction_times'])
            # update run times statistic
            run_times = ModelUtils.updateRuntimeStatistic(run_times, np.array([epoch_time, train_time, test_time])[None, None, ...],
                                                          run, epoch)
            # update skip statistic
            skip_statistics = ModelUtils.updateSkipStatistic(skip_statistics,
                                                             np.array([n_train_samples, processed_train_samples,
                                                                       n_test_samples, processed_test_samples])[None, None, ...],
                                                             run, epoch)

            # update per station rmse data array over runs if desired (especially for K-fold station generalization experiment
            if "per_station_rmse" in config:
                error_per_station_statistics = ModelUtils.updatePerStationErrorStatistic(error_per_station_statistics, rmse_per_station, run, epoch, np.arange(max_station_id+1))

            # store model if it was the best yes
            is_best = overall_test_rmse_model <= best_epoch_test_rmse
            best_epoch_test_rmse = min(overall_test_rmse_model, best_epoch_test_rmse)
            ModelUtils.save_checkpoint({
                'epoch': epoch,
                'run': run,
                'arch': net.__class__.__name__,
                'state_dict': net.state_dict(),
                'overall_test_rmse': overall_test_rmse_model,
                'lead_test_rmse' : overall_test_rmse_model,
                'best_epoch_test_rmse': best_epoch_test_rmse,
                'optimizer': optimizer.state_dict(),
            }, is_best, output_path + '/stored_models/run_%s' % run)

            # flush output to see progress
            sys.stdout.flush()

    # update statistics dict
    ModelUtils.get_model_details(experiment_info, net, optimizer, criterion)

    # complete program runtime
    experiment_info['program_runtime'] = time() - program_start_time

    # generate data set of all experiment statistics and additional information
    experiment_statistic = xr.Dataset({
        'error_statistic' : error_statistics,
        'run_time_statistic': run_times,
        'samples_statistic' : skip_statistics}).assign_attrs(experiment_info)

    # dump experiment statistic
    with open(output_path + '/experiment_statistic.pkl', 'wb') as handle:
        pkl.dump(experiment_statistic, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if 'per_station_rmse' in config:
        # dump experiment statistic
        with open(output_path + '/rmse_per_station.pkl', 'wb') as handle:
            pkl.dump(error_per_station_statistics, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # print program execution time
    m, s = divmod(experiment_info['program_runtime'], 60)
    h, m = divmod(m, 60)
    print('Experiment has successfully finished in %dh %02dmin %02ds' % (h, m, s))