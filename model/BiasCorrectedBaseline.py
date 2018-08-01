import os
import pickle as pkl
import sys
from multiprocessing import Process
from multiprocessing.connection import Pipe
from time import time, strftime, gmtime

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from utils.data import DataLoaders

# this python script contains the code for running the baseline experiment
def runBaseline(config, data_dictionary, train_test_folds):
    program_start_time = time()

    # assign all program arguments to local variables
    Runs = config['runs']
    LeadTimes = config['lead_times']
    SliceLength = config['slice_size']
    TestFraction = config['test_fraction']
    BatchSize = config['batch_size']
    stations = config['stations']
    n_stations = len(stations)
    n_loaders = config['n_loaders']
    n_workers = config['n_workers']

    # update general static model information
    experiment_info = config
    experiment_info['experiment'] = 'bias_corrected_baseline'
    experiment_info['train_test_distribution'] = []

    # output path for experiment information
    setting_string = 'bias_corrected_baseline_tf_%s_sl_%s' % (TestFraction, SliceLength)
    output_path = '%s/%s' % (config['experiment_path'], setting_string)

    # time for the set up until first run
    experiment_info['set_up_time'] = time() - program_start_time
    print('[Time]: Set-up %s' % strftime("%H:%M:%S", gmtime(experiment_info['set_up_time'])))
    sys.stdout.flush()

    # initialize runtime statistic
    run_times = None

    # cross validation
    for run in range(Runs):
        # time stamp at start of run
        run_start_time = time()

        print('[Run %s] Cross-validation test fold %s' % (str(run + 1), str(run + 1)))

        # take the right preprocessed train/test data set for the current run
        train_fold, test_fold = train_test_folds[run]
        n_train, n_test = len(train_fold), len(test_fold)

        # initialize train and test dataloaders
        trainset = DataLoaders.BiasCorrectionCosmoData(
            config=config,
            files=train_fold,
            station_data_dict=data_dictionary)
        trainloader = DataLoader(trainset, batch_size=BatchSize, shuffle=False,
                                 num_workers=n_loaders, collate_fn=DataLoaders.collate_fn)

        testset = DataLoaders.BiasCorrectionCosmoData(
            config=config,
            files=test_fold,
            station_data_dict=data_dictionary)
        testloader = DataLoader(testset, batch_size=BatchSize, shuffle=False,
                                num_workers=n_loaders, collate_fn=DataLoaders.collate_fn)

        # -----------------------------------#
        # ----------- TRAINING  ----------- #
        # ----------------------------------#

        # initialize variables for epoch statistics
        train_start_time = time()

        run_output_path = output_path + '/run_%s' % run
        # load existing bias data
        if (os.path.exists(run_output_path + '/station_bias.pkl') and
                os.path.exists(run_output_path + '/station_hour_bias.pkl') and
                os.path.exists(run_output_path + '/station_hour_month_bias.pkl')):
            print('Load existing bias data and skip training...')
            with open(run_output_path + '/station_bias.pkl', 'rb') as handle:
                station_bias = pkl.load(handle)
            with open(run_output_path + '/station_hour_bias.pkl', 'rb') as handle:
                station_hour_bias = pkl.load(handle)
            with open(run_output_path + '/station_hour_month_bias.pkl', 'rb') as handle:
                station_hour_month_bias = pkl.load(handle)
            print('Existing bias data loaded.')
        # calculate bias on training data and store it
        else:
            # set up multiprocessing workers per hour in [0, 23]
            pipes = []
            processes = []
            station_mapping = {}
            for p in range(n_workers):
                parent, child = Pipe()
                pipes += [parent]
                process_stations = [stations[s] for s in range(p, n_stations, n_workers)]
                for ps in process_stations:
                    station_mapping[ps] = p
                processes += [TrainWorker(child, process_stations, n_train // n_stations, n_test // n_stations, run)]

            for p in processes:
                p.start()

            # loop over complete train set and sample all data points the same way as with the NN models
            for i, data in enumerate(trainloader, 0):
                try:
                    # get training batch, e.g. label, cosmo-1 output and external features
                    Label, Prediction, Station, Month, Hour = tuple(map(lambda x: x.numpy().squeeze(), data[:5]))
                except ValueError:
                    # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                    # collate_fn would return an empty list
                    print('Skipped ValueError in train.')
                    continue

                for batch_sample in range(Label.shape[0]):
                    station = Station[batch_sample]
                    process_station = station_mapping[station]
                    pipe = pipes[process_station]
                    pipe.send((LeadTimes, Prediction[batch_sample], Label[batch_sample], station,
                               Hour[batch_sample], Month[batch_sample]))

            print("All training samples have been queued. Waiting on results...")
            sys.stdout.flush()

            # send poison pill to worker, to signalize end of training phase
            for pipe in pipes:
                pipe.send(None)

            station_hour_month_bias_data_set = None
            for pipe_idx, pipe in enumerate(pipes):
                while True:
                    result = pipe.recv()

                    # check if it is the last result sent by worker
                    if result is None:
                        print("All results of process %s have been consumed." % pipe_idx)
                        break

                    try:
                        station_hour_month_bias_data_set = station_hour_month_bias_data_set.combine_first(result)
                    except AttributeError:
                        station_hour_month_bias_data_set = result

            for pipe in pipes:
                pipe.close()

            sys.stdout.flush()

            station_labels = station_hour_month_bias_data_set.coords['station'].data
            hour_labels = station_hour_month_bias_data_set.coords['hour'].data
            month_labels = station_hour_month_bias_data_set.coords['month'].data

            # calculate the average error made for a specific station
            station_bias = xr.DataArray(np.nanmean(station_hour_month_bias_data_set.data, axis=(2, 3, 4))[..., None],
                                        dims=('lead', 'station', 'bias'),
                                        coords=(LeadTimes, station_labels, ['bias']))

            # calculate the average error made for specific station at a specific hour
            station_hour_bias = xr.DataArray(np.nanmean(station_hour_month_bias_data_set.data, axis=(3, 4))[..., None],
                                             dims=('lead', 'station', 'hour', 'bias'),
                                             coords=(LeadTimes, station_labels, hour_labels, ['bias']))

            station_hour_month_bias = xr.DataArray(np.nanmean(station_hour_month_bias_data_set.data, axis=(4))[..., None],
                                                   dims=('lead', 'station', 'hour', 'month', 'bias'),
                                                   coords=(LeadTimes, station_labels, hour_labels, month_labels, ['bias']))

            # fill missing values (nan) with 0
            station_hour_month_bias = station_hour_month_bias.fillna(0)
            station_bias = station_bias.fillna(0)
            station_hour_bias = station_hour_bias.fillna(0)

            # store time to process complete train set
            train_time = time() - train_start_time
            experiment_info['run_time_train'] = train_time

            # dump experiment statistic
            print('Store bias data after training...')
            run_output_path = output_path + '/run_%s' % run
            if not os.path.exists(run_output_path):
                os.makedirs(run_output_path)
            with open(run_output_path + '/station_bias.pkl', 'wb') as handle:
                pkl.dump(station_bias, handle, protocol=pkl.HIGHEST_PROTOCOL)
            with open(run_output_path + '/station_hour_bias.pkl', 'wb') as handle:
                pkl.dump(station_hour_bias, handle, protocol=pkl.HIGHEST_PROTOCOL)
            with open(run_output_path + '/station_hour_month_bias.pkl', 'wb') as handle:
                pkl.dump(station_hour_month_bias, handle, protocol=pkl.HIGHEST_PROTOCOL)
            print('Stored bias data after training.')

        # -----------------------------------#
        # -----------  TESTING  ----------- #
        # ----------------------------------#

        test_start_time = time()

        # set up multiprocessing workers per hour in [0, 23]
        pipes = []
        processes = []
        station_mapping = {}
        for p in range(n_workers):
            parent, child = Pipe()
            pipes += [parent]
            process_stations = [stations[s] for s in range(p, n_stations, n_workers)]
            for ps in process_stations:
                station_mapping[ps] = p
            processes += [TestWorker(child, process_stations, n_test // n_stations, n_test // n_stations, run)]

        for p in processes:
            p.start()

        trained_stations = set()

        # loop over complete test set and sample all data points the same way as with the NN models
        for i, data in enumerate(testloader, 0):
            try:
                # get training batch, e.g. label, cosmo-1 output and external features
                Label, Prediction, Station, Month, Hour = tuple(map(lambda x: x.numpy().squeeze(), data[:5]))
            except ValueError:
                # when the batch size is small, it could happen, that all labels have been corrupted and therefore
                # collate_fn would return an empty list
                print('Skipped ValueError in test.')
                continue

            for batch_sample in range(Label.shape[0]):
                station = Station[batch_sample]
                trained_stations.add(station)
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

                for lead_idx, lead in enumerate(LeadTimes):
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

                process_station = station_mapping[station]
                pipe = pipes[process_station]
                pipe.send((LeadTimes, label, prediction, station, hour, month, bias_corrected_output,
                           time_bias_corrected_output, time_month_bias_corrected_output))

        # send poison pill to worker, to signalize end of training phase
        for pipe in pipes:
            pipe.send(None)

        # reset bias to not memorize test data from previous run
        error_statistics = None

        for pipe_idx, pipe in enumerate(pipes):
            while True:
                result = pipe.recv()

                # check if it is the last result sent by worker
                if result is None:
                    print("All results of process %s have been consumed." % pipe_idx)
                    break

                try:
                    error_statistics = error_statistics.combine_first(result)
                except AttributeError:
                    error_statistics = result

        for pipe in pipes:
            pipe.close()

        # flush output to see progress
        sys.stdout.flush()

        # time for test
        test_time = time() - test_start_time
        run_time = time() - run_start_time

        # flush output to see progress
        sys.stdout.flush()

        experiment_info['run_time_test'] = test_time
        experiment_info['run_time'] = run_time

        # generate data set of all experiment statistics and additional information
        experiment_statistic = xr.Dataset({
            'error_statistic': error_statistics,
            'run_time_statistic': run_times,
        })
        experiment_statistic.attrs['experiment_info'] = experiment_info

        with open(run_output_path + '/experiment_statistic.pkl', 'wb') as handle:
            pkl.dump(experiment_statistic, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # print program execution time
    m, s = divmod(time() - program_start_time, 60)
    h, m = divmod(m, 60)
    print('Program has successfully finished in %dh %02dmin %02ds' % (h, m, s))

# this class defines an interface for the training and test workers
class Worker(Process):
    def __init__(self, pipe, stations, n_files_train, n_files_test, run):
        Process.__init__(self)
        self.pipe = pipe
        self.stations = stations
        self.current_run = run
        self.n_files_train = len(stations) * n_files_train
        self.n_files_test = len(stations) * n_files_test
        self.ds = None

    def outputProgress(self, n_processed, n_processed_last, last_output_time):
        print("[Process %s] Processed %s (%2.1f%%) with %2.1f files per second." % (
            self.name, n_processed, (n_processed / self.n_files) * 100,
            (n_processed - n_processed_last) / (time() - last_output_time) / len(self.stations)))
        sys.stdout.flush()

    def setOutputInterval(self, n_files):
        self.n_files = n_files
        self.output_intervall = max(n_files // 3, 1)

# a training worker calculates the training bias on the training data split. The training data is split without overlapping
# to learn the baseline biases in parallel in order to increase the performance. The results of each worker are finally
# aggregated by the "runBaseline" method, initiating the parallel processing, to generate the complete final result.
class TrainWorker(Worker):
    def run(self):
        process_name = self.name
        n_processed = 0
        n_processed_last = 0
        self.setOutputInterval(self.n_files_train)
        last_output_time = time()

        # iterate over all data samples in the split and notify the parent process when all the work is done.
        while True:
            job_data = self.pipe.recv()

            if job_data is None:
                # all data samples in the split are processed write out the results and notify the parent process,
                # that the results can be aggregated
                for station in self.stations:
                    self.pipe.send(self.ds.where(self.ds.station == station, drop=True))
                    print("[Process %s] Result for station %s has been sent to pipe." % (process_name, station))
                self.pipe.send(None)
                print('[Process %s]: Exiting after processing %s stations.' % (process_name, len(self.stations)))
                return 1

            LeadTimes, Prediction, Label, Station, Hour, Month = job_data

            if Station not in self.stations: raise AssertionError(
                'Process for stations %s has received data of station %s!' % (str(self.stations), Station))

            station_hour_month_bias_data_set_part = None
            error = Prediction - Label

            for lead_idx, error_value in enumerate(error):
                lead = int(LeadTimes[lead_idx])
                station = int(Station)
                hour = int(Hour[lead_idx])
                month = int(Month[lead_idx])

                try:
                    error_number = \
                        np.count_nonzero(
                            ~np.isnan(self.ds.sel(lead=lead, station=station, hour=hour, month=month).data)) + 1
                    pass
                except (AttributeError, KeyError):
                    error_number = 0

                # generate data set with average error made for a given station, hour and month
                try:
                    station_hour_month_bias_data_set_part = station_hour_month_bias_data_set_part.combine_first(
                        xr.DataArray(error_value.reshape(1, 1, 1, 1, 1),
                                     dims=('lead', 'station', 'hour', 'month', 'error'),
                                     coords=[[lead], [station], [hour], [month], [error_number]]))
                except AttributeError:
                    station_hour_month_bias_data_set_part = xr.DataArray(error_value.reshape(1, 1, 1, 1, 1),
                                                                         dims=(
                                                                             'lead', 'station', 'hour', 'month',
                                                                             'error'),
                                                                         coords=[[lead], [station], [hour], [month],
                                                                                 [error_number]])

            try:
                self.ds = self.ds.combine_first(station_hour_month_bias_data_set_part)
            except AttributeError:
                self.ds = station_hour_month_bias_data_set_part

            n_processed += 1

            if n_processed % self.output_intervall == 0:
                self.outputProgress(n_processed, n_processed_last, last_output_time)
                n_processed_last = n_processed
                last_output_time = time()

# a test worker calculates the test error on a test data split using the calculated biases. The test data is split
# without overlapping to predict the test errors in parallel in order to increase the performance. The results of each
# worker are finally aggregated by the "runBaseline" method, initiating the parallel processing,
# to generate the complete final error result.
class TestWorker(Worker):
    def run(self):
        process_name = self.name
        n_processed = 0
        n_processed_last = 0
        self.setOutputInterval(self.n_files_test)
        last_output_time = time()

        # iterate over all data samples in the split and notify the parent process when all the work is done.
        while True:
            job_data = self.pipe.recv()

            # start_time = time()
            if job_data is None:
                # all data samples in the split are processed write out the results and notify the parent process,
                # that the results can be aggregated
                for station in self.stations:
                    self.pipe.send(self.ds.where(self.ds.station == station, drop=True))
                    print("[Process %s] Result for station %s has been sent to pipe." % (process_name, station))
                self.pipe.send(None)
                print('[Process %s]: Exiting after processing %s stations.' % (process_name, len(self.stations)))
                return 1

            LeadTimes, Label, Prediction, Station, Hour, Month, bias_corrected_output, time_bias_corrected_output, time_month_bias_corrected_output = job_data

            if Station not in self.stations: raise AssertionError(
                'Process for stations %s has received data of station %s!' % (str(self.stations), Station))

            Clabel = Label
            Cout = Prediction
            CbcOut = bias_corrected_output
            CtbcOut = np.array(time_bias_corrected_output)
            CtmbcOut = np.array(time_month_bias_corrected_output)

            # calculate error statistic of current epoch
            diff =  Cout - Clabel

            # calculate station bias corrected error statistic of current epoch
            BCdiff =  CbcOut - Clabel

            # calculate time-station bias corrected error statistic of current epoch
            TBCdiff =  CtbcOut - Clabel

            # calculate time-station bias corrected error statistic of current epoch
            TMBCdiff =  CtmbcOut - Clabel

            error_statistics = None

            for lead_idx, error_value in enumerate(np.vstack((diff, BCdiff, TBCdiff, TMBCdiff)).T):
                lead = int(LeadTimes[lead_idx])
                station = int(Station)
                hour = int(Hour[lead_idx])
                month = int(Month[lead_idx])

                try:
                    error_number = \
                        np.count_nonzero(
                            ~np.isnan(self.ds.sel(lead=lead, station=station, hour=hour, month=month,
                                                  error_type='error').data)) + 1
                    pass
                except (AttributeError, KeyError):
                    error_number = 0

                # generate data set with average error made for a given station, hour and month
                try:
                    error_statistics = error_statistics.combine_first(
                        xr.DataArray(error_value.reshape((1, 1, 1, 1, 1, -1, 1)),
                                     dims=('run', 'lead', 'station', 'hour', 'month',
                                           'error_type', 'error'),
                                     coords=[[self.current_run], [lead], [station], [hour], [month],
                                             ['error', 'sbc_error', 'tsbc_error',
                                              'mtsbc_error'],
                                             [error_number]]))
                except AttributeError:
                    error_statistics = xr.DataArray(error_value.reshape((1, 1, 1, 1, 1, -1, 1)),
                                                    dims=('run', 'lead', 'station', 'hour', 'month',
                                                          'error_type', 'error'),
                                                    coords=[[self.current_run], [lead], [station], [hour], [month],
                                                            ['error', 'sbc_error', 'tsbc_error',
                                                             'mtsbc_error'],
                                                            [error_number]])

            try:
                self.ds = self.ds.combine_first(error_statistics)
            except AttributeError:
                self.ds = error_statistics

            n_processed += 1

            if n_processed % self.output_intervall == 0:
                self.outputProgress(n_processed, n_processed_last, last_output_time)
                n_processed_last = n_processed
                last_output_time = time()
