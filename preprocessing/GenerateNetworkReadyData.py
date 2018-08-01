import json
import os
import pickle as pkl

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils import ModelUtils
from utils.data import DataUtils, DataLoaders


def CreateData(config, data_dictionary, data_statistics, train_test_folds):

    # assign all program arguments to local variables
    with open(config['model']['path']) as handle:
        ModelDict = json.loads(handle.read())

    # check if station and grid time invariant features should be used and set the list of desired parameters
    if not ('grid_time_invariant' in ModelDict and ModelDict['grid_time_invariant']): config['grid_time_invariant_parameters'] =[]
    if not ('station_time_invariant' in ModelDict and ModelDict['station_time_invariant']): config['station_parameters'] = []

    # if needed, load time invariant features
    with open("%s/%s/grid_size_%s/time_invariant_data_per_station.pkl" % (config['input_source'], config['preprocessing'], config['original_grid_size']), "rb") as input_file:
        time_invarian_data = pkl.load(input_file)

    # initialize feature scaling function for each feature
    featureScaleFunctions = DataUtils.getFeatureScaleFunctions(ModelUtils.ParamNormalizationDict, data_statistics)

    # add revision short hash to the config
    config['code_commit'] = ModelUtils.get_git_revision_short_hash()

    # take the right preprocessed train/test data set for the first run
    train_fold, test_fold = train_test_folds[0]

    # initialize train and test dataloaders
    trainset = DataLoaders.CosmoDataGridData(
        config=config,
        station_data_dict=data_dictionary,
        files=train_fold,
        featureScaling=featureScaleFunctions,
        time_invariant_data=time_invarian_data)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

    testset = DataLoaders.CosmoDataGridData(
        config=config,
        station_data_dict=data_dictionary,
        files=test_fold,
        featureScaling=featureScaleFunctions,
        time_invariant_data=time_invarian_data)
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

    # loop over complete train set
    train_data = None
    train_inits = []
    train_stations = None
    for i, data in enumerate(trainloader, 0):
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

            try:
                batch_data = np.concatenate((input.squeeze(), station_time_inv_input, time_data, target, init_station_temp[2]), axis=1)
            except:
                batch_data = np.concatenate((input.squeeze(), time_data, target, init_station_temp[2]), axis=1)

            train_inits += init_station_temp[0]

            if train_data is None:
                train_data = batch_data
                train_stations = init_station_temp[1]
            else:
                train_data = np.vstack((train_data, batch_data))
                train_stations = np.hstack((train_stations, init_station_temp[1]))

        except TypeError:
            # when the batch size is small, it could happen, that all labels have been corrupted and therefore
            # collate_fn would return an empty list
            print('Value error...')
            continue

    # define column names for data frame
    column_names = ['Pressure', 'Wind U-Comp.', 'Wind V-Comp.', 'Wind VMAX', '2m-Temperature',
                    'Temp. of Dew Point',
                    'Cloud Coverage (High)', 'Cloud Coverage (Medium)', 'Cloud Coverage (Low)',
                    'Tot. Precipitation',
                    'ALB_RAD', 'ASOB', 'ATHB', 'HPBL', '2m-Temperature (Lead=0)']
    column_names += ['Grid Height', 'Grid-Station Height Diff.', 'Fraction of Land', 'Soiltype', 'Latitiude',
                     'Longitued', 'Grid-Station 2d Distance']
    if train_data.shape[1] >= 31:
        column_names += ['Station Height', 'Station Latitude', 'Station Longitude']
    column_names += ['Hour (Cosine)', 'Hour (Sine)', 'Month (Cosine)', 'Month (Sine)', 'Lead-Time']
    column_names += ['Target 2m-Temp.']
    column_names += ['COSMO 2m-Temp.']

    train_keys = pd.DataFrame.from_dict({'Station' : train_stations, 'Init' : train_inits})
    train_data = pd.DataFrame(data=train_data, columns=column_names)
    train_ds = pd.concat([train_keys, train_data], axis=1)

    test_data = None
    test_inits = []
    test_stations = None
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

            try:
                batch_data = np.concatenate((input.squeeze(), station_time_inv_input, time_data, target, init_station_temp[2]), axis=1)
            except:
                batch_data = np.concatenate((input.squeeze(), time_data, target, init_station_temp[2]), axis=1)

            test_inits += init_station_temp[0]

            if test_data is None:
                test_data = batch_data
                test_stations = init_station_temp[1]
            else:
                test_data = np.vstack((test_data, batch_data))
                test_stations = np.hstack((test_stations, init_station_temp[1]))


        except TypeError:
            # when the batch size is small, it could happen, that all labels have been corrupted and therefore
            # collate_fn would return an empty list
            print('Value error...')
            continue

    test_keys = pd.DataFrame.from_dict({'Station' : test_stations, 'Init' : test_inits})
    test_data = pd.DataFrame(data=test_data, columns=column_names)
    test_ds = pd.concat([test_keys, test_data], axis=1)

    network_ready_data_path = config['input_source'] + '/network_ready_data'
    if not os.path.exists(network_ready_data_path):
        os.makedirs(network_ready_data_path)

    network_ready_train_data_path = network_ready_data_path + '/train_data'
    network_ready_test_data_path = network_ready_data_path + '/test_data'

    train_ds.to_pickle(network_ready_train_data_path)
    test_ds.to_pickle(network_ready_test_data_path)

    # shap specific config entries for analysis in jupyter notebook
    config['train_data_path'] = network_ready_data_path + '/train_data'
    config['test_data_path'] = network_ready_data_path + '/test_data'

    # dump config
    with open(network_ready_data_path + '/config.pkl', 'wb') as handle:
        pkl.dump(config, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print('Network ready data analysis successfully executed.')
