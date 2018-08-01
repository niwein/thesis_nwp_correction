from torch.utils.data import DataLoader
from utils.data import DataUtils, DataLoaders
from utils import PlotUtils
import pickle as pkl

# List of parameters with corresponding normalization (n=normalize, s=standardize)
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

def runModel(config, data_dictionary, data_statistics, train_test_folds):
    # load time invariant data
    source_path = config['input_source']
    experiment_path = config['experiment_path']

    # assign all program arguments to local variables
    config['batch_size'] = 1
    config['runs'] = 3
    config['grid_size'] = 9


    # if needed, load time invariant features
    with open("%s/%s/grid_size_%s/time_invariant_data_per_station.pkl" % (config['input_source'], config['preprocessing'], config['original_grid_size']), "rb") as input_file:
        time_invarian_data = pkl.load(input_file)

    # initialize feature scaling function for each feature
    featureScaleFunctions = DataUtils.getFeatureScaleFunctions(ParamNormalizationDict, data_statistics)

    plot_config = {
        'features' : config['input_parameters'],
        'time_invariant_features' : config['grid_time_invariant_parameters'],
        'station_features' : config['station_parameters']
    }

    # cross validation
    for run in range(config['runs']):
        print('[Run %s] Cross-validation test fold %s' % (str(run + 1), str(run + 1)))

        # take the right preprocessed train/test data set for the current run
        train_fold, test_fold = train_test_folds[run]

        # initialize train and test dataloaders
        trainset = DataLoaders.SinglePredictionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=train_fold,
            featureScaling=featureScaleFunctions,
            time_invariant_data=time_invarian_data)
        trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                                 num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        testset = DataLoaders.SinglePredictionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=test_fold,
            featureScaling=featureScaleFunctions,
            time_invariant_data=time_invarian_data)
        testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        train_features = [[] for _ in trainset.parameters]
        train_time_invariant_grid_features = [[] for _ in trainset.grid_time_invariant_parameters]
        train_station_features = [[] for _ in trainset.station_parameters]
        train_labels = []
        # loop over complete train set
        for i, data in enumerate(trainloader, 0):
            try:
                # get training batch, e.g. label, cosmo-1 output and external features
                Blabel, Bip2d, StationTimeInv = data
            except ValueError:
                # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                # collate_fn would return an empty list
                print('Value error...')
                continue

            train_labels += list(Blabel.numpy().flatten())
            for feature_idx, _ in enumerate(trainset.parameters):
                train_features[feature_idx] += list(Bip2d[:,feature_idx,:,:].numpy().flatten())
            for ti_feature_idx, _ in enumerate(trainset.grid_time_invariant_parameters):
                train_time_invariant_grid_features[ti_feature_idx] += list(Bip2d[:,trainset.n_parameters + ti_feature_idx,:,:].numpy().flatten())
            for station_feature_idx, _ in enumerate(trainset.station_parameters):
                train_station_features[station_feature_idx] += list(StationTimeInv[:,station_feature_idx].numpy().flatten())

        test_features = [[] for _ in testset.parameters]
        test_time_invariant_grid_features = [[] for _ in testset.grid_time_invariant_parameters]
        test_station_features = [[] for _ in testset.station_parameters]
        test_labels = []
        # loop over complete train set
        for i, data in enumerate(testloader, 0):
            try:
                # get training batch, e.g. label, cosmo-1 output and external features
                Blabel, Bip2d, StationTimeInv = data
            except ValueError:
                # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                # collate_fn would return an empty list
                print('Value error...')
                continue

            test_labels += list(Blabel.numpy().flatten())
            for feature_idx, _ in enumerate(testset.parameters):
                test_features[feature_idx] += list(Bip2d[:,feature_idx,:,:].numpy().flatten())
            for ti_feature_idx, _ in enumerate(testset.grid_time_invariant_parameters):
                test_time_invariant_grid_features[ti_feature_idx] += list(Bip2d[:, testset.n_parameters + ti_feature_idx,:,:].numpy().flatten())
            for station_feature_idx, _ in enumerate(testset.station_parameters):
                test_station_features[station_feature_idx] += list(StationTimeInv[:,station_feature_idx].numpy().flatten())

        plot_config['run'] = run
        PlotUtils.plotFeatureDistribution(output_path=experiment_path, config=plot_config,
                                          train_features=train_features, train_time_invariant_grid_features=train_time_invariant_grid_features,
                                          train_station_features=train_station_features, train_labels=train_labels,
                                          test_features=test_features, test_time_invariant_grid_features=test_time_invariant_grid_features,
                                          test_station_features=test_station_features, test_labels=test_labels)