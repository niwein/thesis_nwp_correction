#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The code was influenced by preprocessing code used on a project of the "Data Science Lab" class in autumn 2018 (Louis de Gaste & Felix Schaumann)

import numpy as np
import os
import xarray as xr
import pickle as pkl
from time import time
from datetime import timedelta
import re
from multiprocessing import Pool

#######################################################################################################################
#
#           Currently only supports preprocessed data from the "per-station" preprocessing, 31.7.2018
#
#######################################################################################################################

# this method calculates and stored max/min/mean/var per feature. Max/min is calculated over all data, however for the
# mean/var only a randomized subsample, determined by sampleRate, is used. Additionally, it calculates the error of
# of each init time + leads for each station. The Label and Prediction it uses comes directly form the preprocessed data
# where the label is the time-corresponding measurement from the station and the prediction is the equivalent
# temperature forecast of the closest (2-d) grid point.
def GenerateFeatureValues(GridSize, FeatureList, TopoFeatureList, sampleRate, isLocal, n_parallel):
    time_begin = time()

    # GridSize: (int) -> size of in preprocessing generated sub-grid around each station
    # FeatureList: list(string) -> all the features we have prerprocessed from the COSMO-1 outputs
    # TopoFeatureList: list(string) -> all the time invariant features we generated
    # averageSampleRate: (float) -> percentage of points we sample per sub-grid
    # isLocal: (bool) -> if the script is run on local machine or in cloud for setting the right paths, etc.
    # n_parallel: (int) -> number of processes to run in parallel

    # output path is of form: .../data/{username}/preprocessed_data/grid_size_{GridSize]
    if isLocal:
        PREPROCESSED_DATA = '/home/n1no/Documents/ethz/master_thesis/code/project/data/preprocessed_data'
        DESTINATION = PREPROCESSED_DATA + '/station/grid_size_' + str(GridSize)  # were to put the generated data
    else:
        PREPROCESSED_DATA = '/mnt/ds3lab-scratch/ninow/preprocessed_data'
        DESTINATION = PREPROCESSED_DATA + '/station/grid_size_' + str(GridSize)  # were to put the generated data

    # file pattern for all preprocessed files
    files = [DESTINATION + '/' + f for f in os.listdir(DESTINATION) if re.match(r'^station_([0-9]+?)_data.nc$', f)]

    print('Found %s files.' % len(files))

    # split the folders into K approx. equal station splits
    n_files = len(files)
    indices = np.linspace(0, n_files, n_parallel+1).astype(int)
    file_splits = [files[indices[i]:indices[i+1]] for i in range(n_parallel)]

    # calculate min/max and select samples on data split in parallel
    with Pool(processes=n_parallel) as pool:
        process_results = []

        # start a new process with the work function for each data split
        for idx_split, split in enumerate(file_splits):
            print('Process %s with range [%s, %s] queued.' % (idx_split, split[0], split[-1]))
            process_results.append(pool.apply_async(processDataSplit, (idx_split, FeatureList, split, sampleRate)))


        # aggregate results from all processes
        for ps_idx, ps_result in enumerate(process_results):

            result = ps_result.get()

            # calculate and update overall min / max
            try:
                overall_min = np.minimum(overall_min, result['min'])
                overall_max = np.maximum(overall_max, result['max'])
                all_sample_elements = np.concatenate([all_sample_elements,
                                                      np.array(result['sampled_elements'])], axis=1)
                df = df.combine_first(result['station_error'])
            # catch exception if variables are not yet defined
            except UnboundLocalError:
                overall_min = result['min']
                overall_max = result['max']
                all_sample_elements = np.array(result['sampled_elements'])
                df = result['station_error']

            print('[Process %s] Results combined...' % ps_idx)

        # store station error data
        with open(PREPROCESSED_DATA + '/baseline_station_error.pkl', 'wb') as handle:
            pkl.dump(df, handle, protocol=pkl.HIGHEST_PROTOCOL)
        df.close()

        # calculate mean and variance for each feature over all sampled data points (only lead time = 0)
        all_sample_elements = np.array(all_sample_elements)
        overall_mean = all_sample_elements.mean(axis=1)
        overall_var = all_sample_elements.var(axis=1)

        # generate xarray data array
        feature_summary = xr.DataArray(np.column_stack((overall_min, overall_max, overall_mean, overall_var)),
                                       dims=('feature', 'characteristic'),
                                       coords=[FeatureList, ['min', 'max', 'mean', 'var']])

        # store feature summary data
        with open(PREPROCESSED_DATA + '/feature_summary_grid_%s.pkl' % GridSize, 'wb') as handle:
            pkl.dump(feature_summary, handle, protocol=pkl.HIGHEST_PROTOCOL)
        feature_summary.close()

        time_end = time()
        print('Preprocessing sucessfully finished in %s.' % str(timedelta(seconds=(time_end - time_begin))))


def processDataSplit(process_id, FeatureList, station_files, sampleRate):
    print('[Process %s] Started...' % process_id)

    n_files = len(station_files)
    sampled_elements = [[] for i in range(len(FeatureList))]
    df = None

    for file_idx, file in enumerate(station_files):  # loop over all  outputs of COSMO-1, e.g. for 3h interval every day
            # ds = pkl.load(input_file)
            ds = xr.open_dataset(file)
            temp_forecast_data = ds.temp_forecast.data - 273.15
            temp_station_data = ds.temp_station.data

            # calculate error for each lead time
            forecast_difference = np.subtract(temp_forecast_data, temp_station_data)[None,...]
            del temp_forecast_data
            del temp_station_data

            cosmo_da = ds.cosmo_data.data
            if df is None:
                min_value = cosmo_da.min(axis=(0,1,2,3))
                max_value = cosmo_da.max(axis=(0,1,2,3))

                df = xr.DataArray(forecast_difference,
                                  dims=('station', 'init_datetime', 'lead'),
                                  coords=[[ds.attrs['station_id']], list(ds.init.data), ds.indexes['lead']])
            else:
                min_value = np.minimum(min_value, cosmo_da.min(axis=(0,1,2,3)))
                max_value = np.maximum(max_value, cosmo_da.max(axis=(0,1,2,3)))

                df = df.combine_first(xr.DataArray(forecast_difference,
                                  dims=('station', 'init_datetime', 'lead'),
                                  coords=[[ds.attrs['station_id']], list(ds.init.data), ds.indexes['lead']]))

            # we only calculate mean and var for lead time = 0
            n_inits, _, n_grid, _, _ = cosmo_da.shape
            n_sample_points = int(n_inits*n_grid**2*sampleRate)
            for i in range(len(FeatureList)):
                sampled_elements[i].extend(np.random.choice(cosmo_da[:,0,:,:,i].flatten(), size=n_sample_points, replace=False))
            del cosmo_da
            print("[Process %s] Processed \"%s\" (%2.1f%%)" % (process_id, file, (file_idx/n_files)*100))

    return_dict = {}
    return_dict['min'] = min_value
    return_dict['max'] = max_value
    return_dict['sampled_elements'] = sampled_elements
    return_dict['station_error'] = df
    return return_dict