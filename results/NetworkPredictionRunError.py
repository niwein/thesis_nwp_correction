import json
import os
import pickle as pkl
import sys
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader

import model.Baseline as Baseline
from utils import ModelUtils
from utils.data import DataUtils, DataLoaders

# this method runs a prediction run for a trained neural network predicting the COSMO-1 error
def runModel(config, data_dictionary, data_statistics, train_test_folds):
    program_start_time = time()

    # assign all program arguments to local variables
    with open(config['model']['path']) as handle:
        ModelDict = json.loads(handle.read())

    # check if station and grid time invariant features should be used and set the list of desired parameters
    if not ('grid_time_invariant' in ModelDict and ModelDict['grid_time_invariant']): config[
        'grid_time_invariant_parameters'] = []
    if not ('station_time_invariant' in ModelDict and ModelDict['station_time_invariant']): config[
        'station_parameters'] = []

    # update general static model information
    experiment_info = config
    experiment_info['model'] = ModelDict
    experiment_info['code_commit'] = ModelUtils.get_git_revision_short_hash()

    # if needed, load time invariant features
    with open("%s/%s/grid_size_%s/time_invariant_data_per_station.pkl" % (
    config['input_source'], config['preprocessing'], config['original_grid_size']), "rb") as input_file:
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
        raise Exception('Node folder of training run has been found for "%s"' % output_path)

    ds = xr.Dataset()

    # cross validation
    for run in range(config['runs']):
        print('[Run %s] Cross-validation test fold %s' % (str(run + 1), str(run + 1)))

        stations = sorted(config['stations'])

        # take the right preprocessed train/test data set for the current run
        train_fold, test_fold = train_test_folds[run]

        # get all inits
        all_inits_set = set(config['inits'])

        # get train and test inits
        train_inits_set = set([t[1] for t in train_fold])
        test_inits_set = set([t[1] for t in test_fold])

        # get all filtered inits
        filtere_inits = set(
            [init for init in all_inits_set if init not in train_inits_set and init not in test_inits_set])

        # make sure, that all sets are distinct
        assert filtere_inits ^ train_inits_set ^ test_inits_set == all_inits_set

        init_type_mapping = {}
        for init in train_inits_set: init_type_mapping[init] = 'train'
        for init in test_inits_set: init_type_mapping[init] = 'test'
        for init in filtere_inits: init_type_mapping[init] = 'filterd'

        all_inits = sorted(list(all_inits_set))
        all_data = [(station, init) for init in all_inits for station in stations]

        n_data_points = len(all_data)

        # keep mappings from init and station to index of result numpy array
        station_index_dict = {}
        for station_idx, station in enumerate(stations): station_index_dict[station] = station_idx
        init_index_dict = {}
        for init_idx, init in enumerate(all_inits): init_index_dict[init] = init_idx

        # initialize train and test dataloaders
        dataset = DataLoaders.ErrorPredictionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=all_data,
            featureScaling=featureScaleFunctions,
            time_invariant_data=time_invarian_data)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        # initialize network, optimizer and loss function
        net = Baseline.model_factory(model_dict=ModelDict, params=dataset.n_parameters, time_invariant_params=dataset.n_grid_time_invariant_parameters,
                                     grid=config['grid_size'], prediction_times=config['prediction_times'])

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

        optimizer = optim.SGD(net.parameters(), lr=optimizer_config['learning_rate'], momentum=optimizer_config['momentum'])

        net, optimizer, *_ = ModelUtils.load_checkpoint(output_path + '/stored_models/run_%s' % run, model=net,
                                                        optimizer=optimizer)


        if torch.cuda.is_available():
            net.cuda()

        # we do not train, but only output the evaluation of the network on train and test data
        net.eval()

        # initialize result array of errors per init and station and initialize it with NaN
        run_error_statistics = np.empty((len(init_index_dict), len(station_index_dict), 5))
        run_error_statistics.fill(np.nan)

        # loop over complete data set
        for i, data in enumerate(dataloader, 0):
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

            except TypeError:
                # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                # collate_fn would return an empty list
                print('Value error...')
                continue

            out = net(input, time_data, station_time_inv_input).squeeze()
            target = target.squeeze()
            diff = (out - target).squeeze()

            for item in range(Blabel.shape[0]):
                init = init_station_temp[0][item]
                station = init_station_temp[1][item].item()
                cosmo_temperature = init_station_temp[2][item].item()
                target_temperature = init_station_temp[3][item].item()
                station_idx = station_index_dict[station]
                init_idx = init_index_dict[init]
                run_error_statistics[init_idx, station_idx, :] = np.array((out[item].item(), cosmo_temperature, target[item].item(), diff[item].item(), target_temperature))

            processed_samples = (i + 1)  * int(config['batch_size'])
            if (i+1) % np.max((1, ((n_data_points // config['batch_size']) // 100))) == 0:
                print("%s samples have been processed. [%2.1f%%]" % (processed_samples, (processed_samples / n_data_points) * 100))
                sys.stdout.flush()


        da = xr.DataArray(run_error_statistics, dims=('init', 'station', 'data'),
                          coords=[all_inits, stations, ['prediction', 'cosmo', 'target', 'difference', 'target_temperature']])
        da = da.sortby(variables='init')
        da.attrs['init_type_mapping'] = sorted(list(init_type_mapping.items()))

        ds['run_%s' % run] = da
        ds.attrs['config'] = config

        print('Error results of run %s have been processed.' % run)
        # flush output to see progress
        sys.stdout.flush()

    if not os.path.exists(output_path):
        raise Exception('Node folder of training run has been found for "%s"' % output_path)

    # dump experiment statistic
    with open(output_path + '/model_run_error.pkl', 'wb') as handle:
        pkl.dump(ds, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # print program execution time
    m, s = divmod(time() - program_start_time, 60)
    h, m = divmod(m, 60)
    print('Experiment has successfully finished in %dh %02dmin %02ds' % (h, m, s))
