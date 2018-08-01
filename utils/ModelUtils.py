import json
import os
import re
import pickle as pkl
import sys

import shutil
from torch.autograd import Variable
import numpy as np
import torch
import xarray as xr
import subprocess

from torch.optim.lr_scheduler import LambdaLR

from utils.data import DataUtils
from utils.external.yellowfin import YFOptimizer

# List of parameters with corresponding normalization (n=normalize, s=standardize)
# the normalization type of each COSMO-1 feature can be determined here
ParamNormalizationDict = {
    'P': 'n',
    'U': 'n',
    'V': 'n',
    'VMAX': 'n',
    'T': 'n',
    'TD': 'n',
    'CLCH': 'n',
    'CLCM': 'n',
    'CLCL': 'n',
    'TOT_PREC': 'n',
    'ALB_RAD': 'n',
    'ASOB': 'n',
    'ATHB': 'n',
    'HPBL': 'n'
}

# get basic information about the model
def get_model_details(dict, model, loss, criterion):
    dict.update({
        'name' : str(model.__class__.__name__),
        'layers' : [str(l) for l in model.children()],
        'criterion' : loss.state_dict(),
        'loss' : str(criterion.__class__.__name__)
    })
    return dict

# transform torch variable into numpy array
def to_np(x):
    return x.data.cpu().numpy()

# create torch variable
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.Variable(x)

# create torch variable from tensor
def getVariable(tensor):
    if torch.cuda.is_available():
        input = Variable(tensor.cuda())
    else:
        input = Variable(tensor)
    return input


def mae(diff):
    """Mean Absolute Error"""
    return np.mean(np.abs(diff))


def mse(diff):
    """Mean Squared Error"""
    return np.mean(np.power(diff, 2))


def rmse(diff):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean(np.power(diff, 2)))

# The following methods are used to update statistics about the model error, the runtime and the number of training
# samples skipped on training a nerual network
def updateErrorStatistic(error_statistics, error ,run, epoch, prediction_times):
    try:
        return error_statistics.combine_first(xr.DataArray(error, dims=('run', 'epoch', 'error', 'prediction_times'),
                                       coords=[[run], [epoch], ['train', 'test'], prediction_times]))
    except AttributeError:
        return xr.DataArray(error, dims=('run', 'epoch', 'error', 'prediction_times'),
                                       coords=[[run], [epoch], ['train', 'test'], prediction_times])

def updatePerStationErrorStatistic(error_statistics, error ,run, epoch, stations):
    try:
        return error_statistics.combine_first(xr.DataArray(error[None, None, ...], dims=('run', 'epoch', 'station', 'error'),
                                       coords=[[run], [epoch], list(stations), ['model', 'cosmo']]))
    except AttributeError:
        return xr.DataArray(error[None, None, ...], dims=('run', 'epoch', 'station', 'error'),
                                       coords=[[run], [epoch], list(stations), ['model', 'cosmo']])

def updateRuntimeStatistic(run_times, times, run, epoch):
    try:
        return run_times.combine_first(xr.DataArray(times, dims=('run', 'epoch', 'time'),
                                       coords=[[run], [epoch], ['epoch', 'train', 'test']]))
    except AttributeError:
        return xr.DataArray(times, dims=('run', 'epoch', 'time'),
                            coords=[[run], [epoch], ['epoch', 'train', 'test']])

def updateSkipStatistic(skip_statistics, statistic, run, epoch, ):
    try:
        return  skip_statistics.combine_first(xr.DataArray(statistic, dims=('run', 'epoch', 'count'),
                                              coords=[[run], [epoch],
                                                      ['n_train', 'n_train_processed', 'n_test', 'n_test_processed']]))
    except AttributeError:
        return xr.DataArray(statistic, dims=('run', 'epoch', 'count'),
                                              coords=[[run], [epoch],
                                                      ['n_train', 'n_train_processed', 'n_test', 'n_test_processed']])

# process all necessary steps tu run a nerual network
def setUpModelRun(options, G):
    # count available CUDA devices
    if torch.cuda.is_available():
        n_cuda_dev = torch.cuda.device_count()
        print('Cuda is available with %s devices.' % n_cuda_dev)

    # prepare model run config
    config = prepareConfig(options, G)

    # set standard optimizer as default, if none is specified
    if 'optimizer' not in config:
        print("No optimizer config found. Using standard sgd optimizer with learning rate 0.001 and momentum 0.9.")
        config['optimizer'] = {
            'algorithm': 'sgd',
            'learning_rate': 0.001,
            'momentum': 0.9
        }

    # load preprocessed time invariant data per stations
    with open("%s/%s/grid_size_%s/time_invariant_data_per_station.pkl" % (config['input_source'], config['preprocessing'], config['original_grid_size']), "rb") as input_file:
        time_invarian_data = pkl.load(input_file)

    # load all station ids
    config['stations'] = time_invarian_data.station.data

    # if preprocessing "station" is used, load complete data into memory accessible by a dictionary. Access keys are the station ids
    data_dictionary = None
    config['inits'] = None
    if config['preprocessing'] == 'station':
        # load data per station into dictionary
        data_dictionary = {}
        for station in config['stations']:
                ds = xr.open_dataset(config['input_source'] + "/station/grid_size_%s/station_%s_data.nc" % (config['original_grid_size'], station))
                data_dictionary[station] = ds.copy(deep=True)
                ds.close()

        # get all init times we have data for
        config['inits'] = ds.coords['init'].data

    # load or generate training and test folds, this requires several parameters, best described in the method itself
    train_test_folds = prepareTrainTestFolds(config)

    data_statistics = DataUtils.getDataStatistics(config=config)

    # the definitions of the grid time invariant parameters and the station parameters are hard-code
    config['grid_time_invariant_parameters'] = ['HH', 'HH_DIFF', 'FR_LAND', 'SOILTYP', 'LAT', 'LON', 'ABS_2D_DIST']
    config['station_parameters'] = ['height', 'lat', 'lon']

    return config, train_test_folds, data_dictionary, data_statistics

# process all necessary stepts tu run the bias-corrected baseline
def setUpBaseline(options):
    # get number of available CUDA devices
    if torch.cuda.is_available():
        n_cuda_dev = torch.cuda.device_count()
        print('Cuda is available with %s devices.' % n_cuda_dev)

    # set up config for model run
    config = prepareConfig(options)

    # load data per station into dictionary
    data_dictionary = {}

    file_paths = [f for f in os.listdir(config['input_source'] + '/baseline') if
                  re.match(r'^station_([0-9]+?)_data.nc$', f)]

    for file_path in file_paths:
            ds = xr.open_dataset(config['input_source'] + '/baseline/%s' % file_path)
            station = ds.station_id
            data_dictionary[station] = ds.copy(deep=True)
            ds.close()

    # get all init times we have data for
    config['inits'] = data_dictionary[list(data_dictionary.keys())[0]].coords['init'].data
    # get all staitons
    config['stations'] = sorted(data_dictionary.keys())
    # get training and test folds based on data splitting algorithm
    train_test_folds = prepareTrainTestFolds(config)

    return config, train_test_folds, data_dictionary

# this method prepares the configuration used in to run the models
def prepareConfig(options, G=None):
    # load experiment configuration
    with open(options.experiment_path + '/experiment_parameters.txt') as handle:
        experiment_parameters = json.loads(handle.read())

    config = experiment_parameters
    config['preprocessing'] = options.preprocessing
    config['input_source'] = options.input_source
    config['experiment_path'] = options.experiment_path
    config['test_filter_window'] = 66
    config['seed'] = 23021
    config['is_normalization'] = True
    config['model'] = {}
    config['original_grid_size'] = G if G is not None else -1
    return config

# this method creates/loads the data splits according to the algorithm provided in the thesis.
# required parameters:
# runs: number of runs train/test folds are generated
# slice_size: number of initialization times in a chunk
# test filter window: window in hours around a test initialization, where training initializations are filtered
# time series length: not used, can be set to zero
# seed: globally set in Runner.py to govern randomization, revealing the same train/test splits, if the method is run repetitively
def prepareTrainTestFolds(config):
    train_test_folds_file_name = '/train_test_folds_r_%s_sl_%s_tfw_%s_tf_%s_series_%s_s_%s.pkl' % (config['runs'],config['slice_size'],
                                                                                        config['test_filter_window'],
                                                                                        config['test_fraction'],
                                                                                        config['time_serie_length'] if 'time_serie_length' in config else 0,
                                                                                        config['seed'])
    # if not already existing, generate filtered data splits for each run
    if not os.path.exists(config['input_source'] + train_test_folds_file_name):
        data_folds = DataUtils.getDataFolds(config=config)
        train_test_folds = DataUtils.getTrainTestFolds(config=config, data_folds=data_folds)
    # else load the existing filtered data split
    else:
        with open(config['input_source'] + train_test_folds_file_name, 'rb') as f:
            train_test_folds = pkl.load(file=f)
            print('Loaded existing train/test folds.')
            sys.stdout.flush()

    return train_test_folds


# get the hash of the currently active git commit
def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])[:-1].decode('ascii')

# this method intializes the pytorch optimizer based on the configured properties, e.g. optimization algorithm,
#  learning rate, weight decay, momentum
def initializeOptimizer(optimizer_config, net):
    # if not explicitly specified, don't use regularization
    if 'weight_decay' not in optimizer_config: optimizer_config['weight_decay'] = 0

    if optimizer_config['algorithm'] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_config['learning_rate'],
                                    momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])
        lr_decay_factor = optimizer_config['lr_decay'] if 'lr_decay' in optimizer_config else 0.8
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay_factor ** epoch)

    elif optimizer_config['algorithm'] == 'yellowFin':
        optimizer = YFOptimizer(net.parameters(), lr=optimizer_config['learning_rate'], weight_decay=optimizer_config['weight_decay'])
        scheduler = None
    elif optimizer_config['algorithm'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_config['learning_rate'], weight_decay=optimizer_config['weight_decay'])
        scheduler = None
    else:
        raise Exception('Optimization algorithm %s not known.' % optimizer_config['algorithm'])

    return optimizer, scheduler



# checkpoint handling form https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
# 24.06.2018
def save_checkpoint(state, is_best, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    filename = destination_path + '/checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, destination_path +'/model_best.pth.tar')

def load_checkpoint(path, model, optimizer):
    file_path = path + '/model_best.pth.tar'
    if os.path.isfile(file_path):
        print("Loading checkpoint from: %s" % file_path)
        checkpoint = torch.load(file_path)
        epoch = checkpoint['epoch'] + 1
        best_epoch_test_rmse = checkpoint['best_epoch_test_rmse']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint with best test rmse %s (epoch %s)" % (best_epoch_test_rmse, checkpoint['epoch']))
    else:
        raise Exception("No checkpoint found at %s" % file_path)
    return model, optimizer, epoch, best_epoch_test_rmse