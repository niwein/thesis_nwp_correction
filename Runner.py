import os
import platform
import random
from time import time

import ModelRun
import ModelRunError
from model import BiasCorrectedBaseline
from preprocessing import GenerateNetworkReadyData
from preprocessing.grid_approach import GenerateFeatureValues,PreprocessingDataDistribution,CreateDataByStation,CreateDataByStationAndInit
from preprocessing.validation import ValidatePreprocessing, ValidateFeatureNormalization
from preprocessing.baseline import CreateBaselineData
from utils import ExperimentUtils, ModelUtils, PlotUtils
from utils.data import DataUtils
from results import NetworkPredictionRun, BiasCorrectedBaselinePredictionRun, NetworkPredictionRunError

# generate input parser and parse arguments
parser = ExperimentUtils.getOptionParser()
options, _ = parser.parse_args()

# check if the script runs locally or on remote, adapt the name of the local computer
isLocal =  1 if platform.uname().node == 'n1no-ThinkPad-T460s' else 0

# hard coded parameters for preprocessing
# include topo features into preprocessed data
withTopo = 1
# parameter list of COSMO-1 parameters used for preprocessing
ListParam = ['P', 'U', 'V', 'VMAX', 'T', 'TD', 'CLCH', 'CLCM', 'CLCL', 'TOT_PREC', 'ALB_RAD', 'ASOB', 'ATHB',
             'HPBL']
# parameter list of additional topop features used for preprocessing
TopoListParam = ['HH', 'HH_DIFF', 'FR_LAND', 'SOILTYP', 'LAT', 'LAT_DIFF', 'RLAT', 'LON', 'LON_DIFF', 'RLON',
                 'ABS_2D_DIST', 'ABS_2D_DIST_RAW']

# constants to be set if the code is either run locally or on remote server
# G: grid size of patch around station. Used for preprocessing to generate desired grid size. Used for model run to load desired data
# DB: first intialization of COSMO-1 data to be used (only for preprocessing)
# DE: last initialization of COSMO-1 data to be used (only for preprocessing)
# T: lead times (used for preprocessing and baseline runs)
# n_parallel: number of processor cores to be used for parallel parts in the code
# observation_path: the path of the station observation containing the target temperature
if isLocal:
    G = 1
    DB = '15093012'
    DE = '17110312'
    T = [0,4,8,12]
    n_parallel = 4
    observation_path = '/home/n1no/Documents/ethz/master_thesis/code/project/data/observations/meteoswiss_t2m_20151001-20180331.nc'
else:
    G = 1
    DB = '15093012'
    DE = '18022821'
    T = list(range(0, 34))
    n_parallel = 16
    observation_path = '/mnt/ds3lab-scratch/ninow/observations/meteoswiss_t2m_20151001-20180331.nc'

# run with: --experiment  "plotTrainTestDistribution"
# plots the training and test data distribution based on the data splitting algorithm
if options.script == 'plotTrainTestDistribution':
    print('Starting to run %s' % options.script)

    # need to specify the folder with the preprocessed data and the desired output path for the plots
    if isLocal:
        source_path = '/home/n1no/Documents/ethz/master_thesis/code/project/data/preprocessed_data'
        output_path = '/home/n1no/Documents/ethz/master_thesis/code/project/preprocessing/experiments/train_test_distribution'
    else:
        source_path = '/mnt/data/ninow/preprocessed_data'
        output_path = '/home/ninow/master_thesis/code/project/preprocessing/experiments/train_test_distribution'
    # generate data splits and plot training/test data disribution
    PreprocessingDataDistribution.plotTrainTestDataDistribution(source_path=source_path,output_path=output_path)

# plot the results of a prediction run over all initializations as an average over all stations and per station seperately
# requires the folder of the model on which the prediction run was made as a run parameter: ---input-source "path_to_model_folder"
elif options.script == 'plotPredictionRun':
    print('Starting to run %s' % options.script)
    PlotUtils.plotPredictionRun(source_path=options.input_source, observation_path=observation_path, n_parallel=n_parallel)

# generates the network training and test results of previously trained network
# requires the folder of the model on which the prediction run was made as a run parameter: ---input-source "path_to_model_folder"
elif options.script == 'plotNetworkTrainingResults':
    print('Starting to run %s' % options.script)
    PlotUtils.plotExperimentErrorEvaluation(options.input_source)

# main preprocessing methods, generating preprocessed data based on the raw COSMO-1 data for neural network training
# ATTENTION: The paths of the data files (i.e. OBS, TOPO, COSMO) net to be specified in the preprocessing classes (CreateDataByStation/CreateDataByStationAndInit)
# run parameter "--preprocessing" can either be "station" or "station_init"
# for "station": a single processed file is generate per station => large memory footprint, since all data in RAM, fast data loading
# for "station_init": a preprocessed file is generated per (station, init) => small memory footprint, slow data loading, because each file is loaded seperately
# for environments with large memory and data on a mounted partition, "station" should be preferred
# if data is on the same machine as the model is run, also "station_init" can be fast
elif options.script == 'createStationGrids':

    print('Starting to run %s' % options.script)

    if options.preprocessing == 'station':
        CreateDataByStation.CreateDataByStation(GridSize=G, DateBegin=DB, DateEnd=DE, PredictionWindow=T,
                                                 ListParam=ListParam, WithTopo=withTopo, TopoListParam=TopoListParam,
                                                 isLocal=isLocal, n_parallel=n_parallel)
    elif options.preprocessing == 'station_init':
        CreateDataByStationAndInit.CreateDataByStationAndInit(GridSize=G, DateBegin=DB, DateEnd=DE, PredictionWindow=T,
                                                 ListParam=ListParam, WithTopo=withTopo, TopoListParam=TopoListParam,
                                                 isLocal=isLocal, n_parallel=n_parallel)
    else:
        raise Exception('Unknown preprocessing: %s' % options.preprocessing)

# main preprocessing methods, generating preprocessed data based on the raw COSMO-1 data for baseline run
# ATTENTION: The paths of the data files (i.e. OBS, TOPO, COSMO) net to be specified in the preprocessing class (CreateBaselineData)
elif options.script == 'createBaselineData':
    print('Starting to run %s' % options.script)

    CreateBaselineData.CreateBaselineData(DateBegin=DB, DateEnd=DE, PredictionWindow=T,
                                            isLocal=isLocal, n_parallel=n_parallel)

# necessary additional preprocessing method, calculating the feature statistic of the preprocessed data for normalization/standardization
# ATTENTION: The paths to the preprocessed data must be specified in "GenerateFeatureValues"
elif options.script == 'generateFeatureValues':
    print('Starting to run %s' % options.script)
    # averageSampleRate determines the percentage of preprocessed data samples to calculate the mean per feature
    averageSampleRate = 0.1
    # calculate and store feature value statistics (min,max,mean,var)
    GenerateFeatureValues.GenerateFeatureValues(G, ListParam, TopoListParam, averageSampleRate, isLocal, n_parallel)

# NOT WORKING
# Was locally used to visually validate the preprocessing
elif options.script == 'validatePreprocessing':
    print('Starting to run %s' % options.script)

    S=[x for x in range(144)]
    ValidatePreprocessing.GetData(S, G, withTopo, DB, DE, T, isLocal)

# main methods to run nerual network model runs
# this requires >=1 model config file in the "models" folder of an experiment and an "experiment_parameters.txt" file
# sample model configs can be found under /results/runs/neural_network
elif options.script == 'runModel':
    # take time of start of the experiment
    experiment_start = time()

    # this method executes all prerequisite steps to run a model, i.e. preparation of run config, generating/loading data splits,
    # loading the data into a dictionary for "station" preprocessed data, loading data statistic of features for normalization
    config, train_test_folds, data_dictionary, data_statistics = ModelUtils.setUpModelRun(options=options, G=G)

    print('Starting to run %s' % options.script)

    # validation method of feature normalization, can be ignored
    if options.model_type == "featureNormalizationValidation":
        ValidateFeatureNormalization.runModel(config=config,
                          data_dictionary=data_dictionary,
                          data_statistics=data_statistics,
                          train_test_folds=train_test_folds)
        print('Finished validation of feature normalization.')
    # all model runs, or experiment with a similar setup
    else:
        # get all paths of model configuration files. This allows to define several models at a time to be run
        models = [f[:-4] for f in os.listdir(config['experiment_path'] + '/models')]
        n_models = len(models)
        print('%s models found to run.' % n_models)
        # run all found models
        for m_idx, m in enumerate(models):
            config['model']['path'] = config['experiment_path'] + '/models/%s.txt' % m
            config['model']['name'] = m
            print('Starting experiment with model %s' % m)

            # prediction run does predict a complete training and test split using a previously trained model
            # neds to be run after "network"
            # only works for label definition of directly predicting 2m-temperature
            if options.model_type == "predictionRun":
                NetworkPredictionRun.runModel(config=config,
                                              data_dictionary=data_dictionary,
                                              data_statistics=data_statistics,
                                              train_test_folds=train_test_folds)

            # main method to train a nerual network on directly predicting the 2m-temperature.
            # Produces tensorflow outputs, training and test plots, stored trained model and an error statistic
            elif options.model_type == "network":
                ModelRun.runModel(config=config,
                                  data_dictionary=data_dictionary,
                                  data_statistics=data_statistics,
                                  train_test_folds=train_test_folds)

            # main method to train a nerual network on predicting the COMSO-1 error.
            # Produces tensorflow outputs, training and test plots, stored trained model and an error statistic
            elif options.model_type == "network_error_prediction":
                ModelRunError.runModel(config=config,
                                  data_dictionary=data_dictionary,
                                  data_statistics=data_statistics,
                                  train_test_folds=train_test_folds)

            # prediction run does predict a complete training and test split using a previously trained model
            # neds to be run after "network_error_prediction"
            # only works for label definition of predicting the COSMO-1 error
            elif options.model_type == "network_error_prediction_run":
                NetworkPredictionRunError.runModel(config=config,
                                  data_dictionary=data_dictionary,
                                  data_statistics=data_statistics,
                                  train_test_folds=train_test_folds)

            # generates preprocessed data for SHAP analysis and for neural embedding scripts
            elif options.model_type == "generateNetworkReadyTestData":
                GenerateNetworkReadyData.CreateData(config=config,
                                                       data_dictionary=data_dictionary,
                                                       data_statistics=data_statistics,
                                                       train_test_folds=train_test_folds)
            else:
                raise Exception('Unknown model type for "runModel": %s' % options.model_type)
            print('Finished experiment with model %s (%2.1f%%)' % (m, ((m_idx+1) / n_models) * 100))

        # after the network training or a prediction run to directly predicting the 2m-temperatures, Useful plots are automatically
        # generated and stored in the experiment folder under "/plots"
        if options.model_type == "predictionRun":
            PlotUtils.plotPredictionRun(config['experiment_path'], observation_path)
        elif options.model_type == "network":
            PlotUtils.plotExperimentErrorEvaluation(config['experiment_path'])

# main method to run baseline experiment. Needs to be run after creating the baseline preprocessed files
# this requires an "experiment_parameters.txt" file, containing parameters for the experiment
# a sample baseline experiments parameter file can be found under /results/runs/2d_baseline
elif options.script == 'runBaseline':
    experiment_start = time()

    # setting up of prerequisites for baseline prediction. Similar as for NN model run, with the difference of loading the
    # baseline specific preprocessed data and not requiring an optimizer to be defined in the config
    config, train_test_folds, data_dictionary = ModelUtils.setUpBaseline(options=options)
    # always uses all lead times defined on the top of the Runner.py class
    config['lead_times'] = T
    print('Starting to run %s' % options.script)
    base_experiment_path = config['experiment_path']

    config['experiment_path'] = base_experiment_path

    # a prediction run uses previously computed biases by BiasCorrectedBaseline and can therefore not be run before
    if options.model_type == 'predictionRun':
        BiasCorrectedBaselinePredictionRun.runBaseline(config=config,
                                          data_dictionary=data_dictionary,
                                          train_test_folds=train_test_folds)
    # calculates all biases on the training data and predicts on the test data
    else:
        BiasCorrectedBaseline.runBaseline(config=config,
                                          data_dictionary=data_dictionary,
                                          train_test_folds=train_test_folds)

    # print program execution time
    m, s = divmod(time() - experiment_start, 60)
    h, m = divmod(m, 60)
    print('Baseline with %s distance experiment has successfully finished in %dh %02dmin %02ds' % (
    config['distance_metric'], h, m, s))

# runs a generalization experiment on stations only used in prediction
# this requires >=1 model config file in the "models" folder of an experiment and an "experiment_parameters.txt" file
# sample model configs and "experiment_parameters.txt" files can be found under /results/runs/spatial_generalization
# IMPORTANT: in the "experiment_parameters.txt" file one has to specify, how many test stations should be use (and therefore
# left out for training) and what station is the first in consecutive order to defined the test stations
elif options.script == 'spatialGeneralizationExperiment':
    experiment_start = time()

    config, train_test_folds, data_dictionary, data_statistics = ModelUtils.setUpModelRun(options=options, G=G)

    # we filter out in config specified test station from train set and filter all train
    # stations from test set for each run
    train_test_folds = DataUtils.filterUnseenTestStations(train_test_folds=train_test_folds, config=config)

    print('Starting to run %s' % options.script)
    print("Test Stations:", config['test_stations'])

    models = [f[:-4] for f in os.listdir(config['experiment_path'] + '/models')]
    n_models = len(models)
    print('%s models found to run.' % n_models)
    for m_idx, m in enumerate(models):
        config['model']['path'] = config['experiment_path'] + '/models/%s.txt' % m
        config['model']['name'] = m

        ModelRun.runModel(config=config,
                          data_dictionary=data_dictionary,
                          data_statistics=data_statistics,
                          train_test_folds=train_test_folds)
        print('Finished experiment with model %s (%2.1f%%)' % (m, ((m_idx+1) / n_models) * 100))