import os
import pickle as pkl
import sys
from time import time

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from utils.data import DataLoaders

# this method runs a prediction run for the bias corrected baselines
def runBaseline(config, data_dictionary, train_test_folds):
    program_start_time = time()

    # assign all program arguments to local variables
    Runs = config['runs']
    SliceLength = config['slice_size']
    TestFraction = config['test_fraction']

    # update general static model information
    experiment_info = config
    experiment_info['experiment'] = 'bias_corrected_baseline'
    experiment_info['train_test_distribution'] = []

    # output path for experiment information
    setting_string = 'bias_corrected_baseline_tf_%s_sl_%s' % (TestFraction, SliceLength)
    output_path = '%s/%s' % (config['experiment_path'], setting_string)

    # time for the set up until first run
    experiment_info['set_up_time'] = time() - program_start_time
    sys.stdout.flush()

    # container for error data of all runs of all bias corrected baseline versions
    error_container = BiasCorrectedErrorContainer()

    # cross validation
    for run in range(Runs):
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

        error_container.resetDataArray(run=run,
                                       station_index_dict=station_index_dict,
                                       init_index_dict=init_index_dict,
                                       all_inits=all_inits,
                                       stations=stations,
                                       init_type_mapping=init_type_mapping)

        # initialize train and test dataloaders
        dataset = DataLoaders.BiasCorrectionCosmoData(
            config=config,
            station_data_dict=data_dictionary,
            files=all_data)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['n_loaders'], collate_fn=DataLoaders.collate_fn)

        # load bias
        print('Load bias data...')
        run_output_path = output_path + '/run_%s' % run
        with open(run_output_path + '/station_bias.pkl', 'rb') as handle:
            station_bias = pkl.load(handle)
        with open(run_output_path + '/station_hour_bias.pkl', 'rb') as handle:
            station_hour_bias = pkl.load(handle)
        with open(run_output_path + '/station_hour_month_bias.pkl', 'rb') as handle:
            station_hour_month_bias = pkl.load(handle)
        print('Loaded bias.')

        sys.stdout.flush()

        # loop over complete data set and sample all data points the same way as with the NN models
        for i, data in enumerate(dataloader, 0):
            try:
                # get training batch, e.g. label, cosmo-1 output and external features
                Label, Prediction, Station, Month, Hour = tuple(map(lambda x: x.numpy().squeeze(), data[:5]))
                Init = data[5]
            except ValueError:
                # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                # collate_fn would return an empty list
                print('Skipped ValueError in test.')
                continue

            for batch_sample in range(Label.shape[0]):
                init = Init[batch_sample]
                station = Station[batch_sample]
                prediction = Prediction[batch_sample]
                hour = Hour[batch_sample]
                month = Month[batch_sample]
                label = Label[batch_sample]

                # calculate the prediction with subtracting the learned mean error of the station over all hours
                try:
                    bias_corrected_output = prediction - station_bias.sel(station=station).data.squeeze()
                except KeyError:
                    # if we have no data for a station and hour, we do not correct the data
                    print('Station bias not corrected, because no bias for station %s available.' % station)
                    bias_corrected_output = prediction

                # calculate the prediction with subtracting the learned mean error of the station at a specific hour
                time_bias_corrected_output = []
                time_month_bias_corrected_output = []

                # this could be rewritten as a loop over all lead times, a.t.m. we only use lead time = 1
                lead_idx = 1
                lead = 1

                lead_hour = hour[lead_idx]
                lead_month = month[lead_idx]

                try:
                    time_bias_corrected_output += [
                        prediction[lead_idx] - station_hour_bias.sel(lead=lead, station=station,
                                                                     hour=lead_hour).data.squeeze()]
                except KeyError:
                    # if we have no data for a station and hour, we do not correct the data
                    print('Station time-bias not corrected, because no bias for station %s and time %s available.'
                          % (station, lead_hour))
                    time_bias_corrected_output += [prediction[lead_idx]]

                # calculate the prediction with subtracting the learned mean b
                # of the station at a specific hour for specific month
                try:
                    time_month_bias_corrected_output += [
                        prediction[lead_idx] - station_hour_month_bias.sel(lead=lead, station=station,
                                                                           hour=lead_hour,
                                                                           month=lead_month).data.squeeze()]
                except KeyError:
                    # if we have no data for a station and hour, we do not correct the data
                    print(
                        'Station time-month-bias not corrected, because no bias for station %s, time %s and month % s available.'
                        % (station, lead_hour, lead_month))
                    time_month_bias_corrected_output += [prediction[lead_idx]]

                CbcOut = bias_corrected_output
                CtbcOut = np.array(time_bias_corrected_output)
                CtmbcOut = np.array(time_month_bias_corrected_output)

                error_container.updateDataArray(station=station, init=init, label=label[lead_idx], prediction=prediction[lead_idx],
                                                CbcOut=CbcOut[lead_idx], CtbcOut=CtbcOut, CtmbcOut=CtmbcOut)

            processed_samples = (i + 1)  * config['batch_size']
            if (i+1) % np.max((1, ((n_data_points // config['batch_size']) // 100))) == 0:
                print("%s samples have been processed. [%2.1f%%]" % (processed_samples, (processed_samples / n_data_points) * 100))
                sys.stdout.flush()

        # completed error data arrays per baseline version are added to corresponding data set with current
        # run number as label
        error_container.updateAllDataSets()

        print('Error results of run %s have been processed.' % run)
        # flush output to see progress
        sys.stdout.flush()

    error_container.storeDataSets(config['experiment_path'])

    # print program execution time
    m, s = divmod(time() - program_start_time, 60)
    h, m = divmod(m, 60)
    print('Experiment has successfully finished in %dh %02dmin %02ds' % (h, m, s))

# this class is a container to capture the errors of the bias corrections during the prediction run
class BiasCorrectedErrorContainer():
    error_statistic = {}
    ds = {}

    def __init__(self):
        # initialize data set for each bias corrected version of baseline
        self.ds['b'] = xr.Dataset()
        self.ds['b'].attrs['model'] = 'Baseline errror'
        self.ds['sbcb'] = xr.Dataset()
        self.ds['sbcb'].attrs['model'] = 'Station bias corrected error'
        self.ds['stbcb'] = xr.Dataset()
        self.ds['stbcb'].attrs['model'] = 'Station-time bias corrected error'
        self.ds['stmbcb'] = xr.Dataset()
        self.ds['stmbcb'].attrs['model'] = 'Station-time-month bias corrected error'

    def resetDataArray(self, run, init_index_dict, station_index_dict, all_inits, stations, init_type_mapping):
        # dictionary to keep track of index in numpy array for given init time and station
        self.run = run
        self.init_index_dict = init_index_dict
        self.station_index_dict = station_index_dict
        self.all_inits = all_inits
        self.stations = stations
        self.init_type_mapping = init_type_mapping

        # initialize result array of errors per init and station and initialize it with NaN
        for key in self.ds.keys():
            self.error_statistic[key] = np.empty((len(self.init_index_dict), len(self.station_index_dict), 3))
            self.error_statistic[key].fill(np.nan)

    def updateDataArray(self, station, init, label, prediction, CbcOut, CtbcOut, CtmbcOut):
        # calculate error statistic of current epoch
        diff = prediction - label

        # calculate station bias corrected error statistic of current epoch
        BCdiff = CbcOut - label

        # calculate time-station bias corrected error statistic of current epoch
        TBCdiff = CtbcOut - label

        # calculate time-station bias corrected error statistic of current epoch
        TMBCdiff = CtmbcOut - label

        station_idx = self.station_index_dict[station]
        init_idx = self.init_index_dict[init]

        self.error_statistic['b'][init_idx, station_idx, :] = np.array(
            (prediction, label, diff))

        self.error_statistic['sbcb'][init_idx, station_idx, :] = np.array(
            (CbcOut, label, BCdiff))

        self.error_statistic['stbcb'][init_idx, station_idx, :] = np.array(
            (CtbcOut, label, TBCdiff))

        self.error_statistic['stmbcb'][init_idx, station_idx, :] = np.array(
            (CtmbcOut, label, TMBCdiff))

    def updateAllDataSets(self):
        for key in self.ds.keys():
            da = xr.DataArray(self.error_statistic[key], dims=('init', 'station', 'data'),
                              coords=[self.all_inits, self.stations, ['prediction', 'target', 'difference']])

            da = da.sortby(variables='init')
            da.attrs['init_type_mapping'] = sorted(list(self.init_type_mapping.items()))

            self.ds[key]['run_%s' % self.run] = da

    def storeDataSets(self, results_path):
        for key in self.ds.keys():
            path = results_path + '/prediction_run/%s' % key

            if not os.path.exists(path):
                os.makedirs(path)

            # dump experiment statistic
            with open(path + '/model_run_error.pkl', 'wb') as handle:
                pkl.dump(self.ds[key], handle, protocol=pkl.HIGHEST_PROTOCOL)
