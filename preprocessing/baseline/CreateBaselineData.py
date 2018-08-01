#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The code was influenced by preprocessing code used on a project of the "Data Science Lab" class in autumn 2018 (Louis de Gaste & Felix Schaumann)
import json
import os
import pickle as pkl
import re
import sys
import traceback
from datetime import timedelta
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from utils.data import DataUtils


# this exception is used to break innter loop and continue outer loop with next iteration
class SkipException(Exception):
    pass


# this method generates the grids around each stations, that are later used to train a machine learning appraoch
# additionally it generates time invariant data for each station and the corresponding neares grid points
# the grids around each station are aggregated over all initialization times and stored separately for each station
def CreateBaselineData(DateBegin, DateEnd, PredictionWindow, isLocal, n_parallel):
    time_begin = time()

    if DateEnd < DateBegin:
        raise Exception('DateEnd is smaller than DateBegin.')

    # different paths, whether we run the script locally or on a cluster node
    if isLocal:
        ADDRESSdata = '/home/n1no/Documents/ethz/master_thesis/code/project/data/cosmo-1/data_subset'  # COSMO-1 outputs
        ADDRESStopo = '/home/n1no/Documents/ethz/master_thesis/code/project/data'  # base address of topo files
        ADDRESSobst = '/home/n1no/Documents/ethz/master_thesis/code/project/data/observations/'  # base adress of obs files
        DESTINATION = '/home/n1no/Documents/ethz/master_thesis/code/project/data/preprocessed_data/baseline'  # target directory for all generated files
    else:
        ADDRESSdata = '/mnt/data/bhendj/full/cosmo-1'  # COSMO-1 outputs
        ADDRESStopo = '/mnt/ds3lab-scratch/ninow/topo'  # base address of topo files
        ADDRESSobst = '/mnt/ds3lab-scratch/ninow/observations'  # base adress of obs files
        DESTINATION = '/mnt/ds3lab-scratch/ninow/preprocessed_data/baseline'  # target directory for all generated files

    if not os.path.exists(DESTINATION):
        os.makedirs(DESTINATION)

    # create an output folder for each station, based on the station ids
    OBS = xr.open_dataset(ADDRESSobst + '/meteoswiss_t2m_20151001-20180331.nc')
    TOPO = xr.open_dataset(ADDRESStopo + '/topodata.nc')
    station_ids = OBS['station_id'].data

    # extract time invariant features for each station and the corresponding sub-grid
    if not os.path.exists(DESTINATION + '/station_neighbors.pkl'):

        station_neighbors = {}

        # calculate for each station the neighbors on the grid in parallel
        with Pool(processes=n_parallel) as pool:
            process_results = []

            gridHeightData = TOPO.HH.data
            gridLatData = TOPO.lat.data
            gridLonData = TOPO.lon.data

            # start a new process with the work function for each data split
            for idx_S, S in enumerate(station_ids):
                # calculate height difference between grid points and station
                station_height = OBS['height'].sel(station_id=S).data
                station_lat = OBS['lat'].sel(station_id=S).data
                station_lon = OBS['lon'].sel(station_id=S).data

                print('Neighborhood calculation for staiton %s queued.' % S)
                process_results.append(pool.apply_async(getStationNeighbors, (S, gridHeightData, gridLatData,
                                                                              gridLonData, station_height,
                                                                              station_lat, station_lon)))

            # aggregate results from all processes
            for ps_idx, ps_result in enumerate(process_results):
                # sync processes
                S, neighbor_data = ps_result.get()
                station_neighbors[S] = neighbor_data
                print('[Process %s] Synchronized after data creation.' % ps_idx)

        with open(DESTINATION + '/station_neighbors.pkl', 'wb') as handle:
            pkl.dump(station_neighbors, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print('Station time invariant features have been calculated and stored.')
    else:
        with open(DESTINATION + '/station_neighbors.pkl', 'rb') as handle:
            station_neighbors = pkl.load(handle)
        print('Station time invariant features have been found on disk and were therefore not created again.')

    OBS.close()
    TOPO.close()

    for S in station_ids:
        temp_output_path = DESTINATION + '/temp/station_%s' % S
        if not os.path.exists(temp_output_path):
            os.makedirs(temp_output_path)

    # get all COSMO-1 files that are in the given time interval and have not yet been processed and thus do not
    # already exists in the output folder
    folders = DataUtils.getFilesToProcess(ADDRESSdata, DESTINATION, 'Station', DateBegin, DateEnd)
    folders.sort()

    # calculate begin and end index of array to exclude files, that are not in the specified time interval
    begin, end = -1, -1
    for idx, folder in enumerate(folders):
        if folder[:-4] >= DateBegin:
            begin = idx
            break

    for idx, folder in enumerate(folders):
        if folder[:-4] <= DateEnd:
            end = idx
        else:
            break

    if begin == -1 or end == -1:
        raise Exception('Could not find start or end in array.')

    folders = folders[begin:end + 1]
    print('%s files are left to be preprocessed.' % len(folders))

    # split the folders into K approx. equal splits
    if n_parallel <= 1:
        folder_splits = [folders]
    else:
        n_folders = len(folders)
        indices = np.linspace(0, n_folders, n_parallel + 1).astype(int)
        folder_splits = [folders[indices[i]:indices[i + 1]] for i in range(n_parallel)]

    folder_splits = [l for l in folder_splits if len(l) > 0]
    # take timestamp after set-up
    time_setup = time()

    # run preprocessing in parallel for all splits and keep the processes in a list to sync them later
    # calculate min/max and select samples on data split in parallel
    with Pool(processes=n_parallel) as pool:
        process_results = []

        # start a new process with the work function for each data split
        for idx_split, split in enumerate(folder_splits):
            print('Process %s with range [%s, %s] queued.' % (idx_split, split[0], split[-1]))
            process_results.append(pool.apply_async(GetDataWrapper, (idx_split, ADDRESSdata, ADDRESStopo, ADDRESSobst,
                                                                     DESTINATION, split, station_neighbors,
                                                                     PredictionWindow, isLocal)))

        # aggregate results from all processes
        for ps_idx, ps_result in enumerate(process_results):
            # sync processes
            result = ps_result.get()
            print('[Process %s] Synchronized after data creation.' % ps_idx)

        station_folders_paths = [f for f in os.listdir(DESTINATION + '/temp') if re.match(r'^station_([0-9]+?)$', f)]

        process_results = []
        for ps_idx, station_folder in enumerate(station_folders_paths):
            print('Process %s with station folder %s queued.' % (ps_idx, station_folder))
            process_results.append(pool.apply_async(aggregateProcessFiles, (ps_idx, DESTINATION, station_folder)))

        # aggregate results from all processes
        for ps_idx, ps_result in enumerate(process_results):
            # sync processes
            result = ps_result.get()
            print('[Process %s] Synchronized after aggregation.' % ps_idx)

    # take timestamp after completing all processes
    time_end = time()

    # dump preprocessing information in a descriptive JSON file
    preprocessing_information = {
        'data_begin': DateBegin,
        'data_end': DateEnd,
        'future_hours': PredictionWindow,
        'n_processes': n_parallel,
        'time_setup': str(timedelta(seconds=(time_setup - time_begin))),
        'time_preprocessing': str(timedelta(seconds=(time_end - time_setup)))
    }

    preprocessing_information_json = json.dumps(preprocessing_information)
    f = open(DESTINATION + '/setup.json', 'w')
    f.write(preprocessing_information_json)
    f.close()

    print('Station baseline reprocessing sucessfully finished in %s.' % str(timedelta(seconds=(time_end - time_begin))))


def GetDataWrapper(processId, ADDRESSdata, ADDRESStopo, ADDRESSobst, DESTINATION, Files, station_neighbors,
                   PredictionWindow, isLocal):
    try:
        return GetData(processId, ADDRESSdata, ADDRESStopo, ADDRESSobst, DESTINATION, Files, station_neighbors,
                       PredictionWindow, isLocal)
    except Exception:
        sys.stderr('[Process %s]: %s' % (processId, traceback.format_exc()))


def GetData(processId, ADDRESSdata, ADDRESStopo, ADDRESSobst, DESTINATION, Files, station_neighbors, PredictionWindow,
            isLocal):
    # processId: (int) -> the id of the process running this method
    # ADDRESSdata: (string) -> base path to COSMO-1 data
    # ADDRESStopo: (string) -> base path to all topology files
    # ADDRESSobs: (string) -> base path to all observation files
    # DESTINATION: (string) -> base path to target output folder
    # Files: (list(string)) -> list of all files ('yymmddHH') to be processed, e.g. ['15031203', '15031206', ...]
    # station_neighbors (dict(station_id -> neighbor data): contains data about the neighbor grid point for each station
    # PredictionWindow: (list of int) -> all future hours t's [t,t+1,t+2,...] being processed, e.g. y_t, y_t+1,...
    # isLocal: (bool) -> for setting the right paths if the script is running on a local machine or in cloud, etc.

    # path to observation and topological data
    if isLocal:
        OBS = xr.open_dataset(ADDRESSobst + '/meteoswiss_t2m_20151001-20180331.nc')
    else:
        # to fix parallelization errors, each process gets its own set of TOPO and OBS files
        OBS = xr.open_dataset(ADDRESSobst + '/process_%s/meteoswiss_t2m_20151001-20180331.nc' % processId)

    # load all station ids
    station_ids = list(station_neighbors.keys())

    # generate a view on temperature observation at each station
    TempObs = OBS['t2m'].sel(station_id=station_ids)

    # initialize data variables
    DATA = None
    FileLabels = []
    skipped_files = 0

    data_labels = ['target', 'pred_2d', 'pred_3d', 'month', 'hour']

    # we now start the iteration through: Each folder, each file, each parameter, each station
    for file_idx, file in enumerate(Files):  # loop over all  outputs of COSMO-1, e.g. for 3h interval every day
        try:
            # mark start of preprocessing of n-th file
            print('[Process %s] Start processing %s' % (processId, file))

            # adapt file index to skipped files
            file_idx = file_idx - skipped_files

            for idx_T, T in enumerate(PredictionWindow):  # loop over all future predictions, e.g. current hour + T

                if T < 10:
                    NAME = ADDRESSdata + '/' + file + '/c1ffsurf00' + str(T) + '.nc'
                else:
                    NAME = ADDRESSdata + '/' + file + '/c1ffsurf0' + str(T) + '.nc'

                # load netCRF4 dataset
                dataset = xr.open_dataset(NAME)

                # get initialization time of COSMO-1 data point
                t = dataset['time'].data

                # check that we do not process a data point before the first observation
                if t < OBS['time'].data[0]:
                    print('[Process %s] Skipped %s' % (processId, file))
                    raise SkipException()

                # initialize result arrays when data is not outside observation data window
                if DATA is None:
                    skipped_files = file_idx
                    files_in_range = len(Files) - file_idx
                    file_idx = file_idx - skipped_files
                    DATA = np.zeros(
                        (len(station_ids), files_in_range, len(PredictionWindow), len(data_labels)))
                    TimeStamp = np.zeros((files_in_range, len(PredictionWindow)))

                # calculate the month and hour for the prediction for month/hour bias correction
                month = ((DataUtils.passed_days_per_month_dict[int(file[2:4])] + int(file[4:6]) + (
                int(file[6:8]) + T) // 24) // 30.42) % 12
                hour = (int(file[6:8]) + T) % 24

                try:
                    target = TempObs.sel(time=t)
                except RuntimeError:
                    print('Error with time=%s.' % t)
                    raise

                MAP = dataset['T'].data.squeeze()
                for idx_S, S in enumerate(station_ids):
                    closest_2d, closest_3d = station_neighbors[S]
                    pred_2d = MAP[closest_2d[0]] - 273.15
                    pred_3d = MAP[closest_3d[0]] - 273.15

                    DATA[idx_S, file_idx, idx_T, :] = np.array(
                        (target.sel(station_id=S).data.squeeze(), pred_2d, pred_3d, month, hour))

                TimeStamp[file_idx, idx_T] = t[0]

                # this dataset is not used anymore and can be closed
                dataset.close()

            FileLabels += [file[:8]]

            # print that processing of data point has been completed
            print('[Process %s] Finished %s [%2.1f%%]' % (processId, file, (file_idx + 1) / files_in_range * 100))
            if file_idx % 10 == 0:
                sys.stdout.flush()
        except SkipException:
            continue

    OBS.close()

    if DATA is None:
        print(
            '[Process %s] No data was preprocessed. Possibly all files to preprocess were skipped, because their date is before'
            'the first observation.' % processId)
        return None

    # we write the data in a binary file
    for idx_S, S in enumerate(station_ids):
        leads = PredictionWindow
        dims = ('init', 'lead', 'data')
        station_data = xr.DataArray(DATA[idx_S],
                                    dims=dims,
                                    coords=[FileLabels, leads, data_labels])
        ds = xr.Dataset({'station_data': station_data})
        ds.attrs['time_stamp'] = TimeStamp
        ds.attrs['station_id'] = S

        ds.to_netcdf(DESTINATION + '/temp' + '/station_%s/process_%s.nc' % (S, processId))

    print('[Process %s] Data split successfully preprocessed!' % processId)

    return 0


def aggregateProcessFiles(process_idx, DESTINATION, station_folder):
    file_paths = [f for f in os.listdir(DESTINATION + '/temp/%s' % station_folder) if
                  re.match(r'^process_([0-9]+?).nc$', f)]

    n_files = len(file_paths)

    station_result = None
    for file_idx, file in enumerate(file_paths):
        full_file_path = DESTINATION + '/temp/%s/%s' % (station_folder, file)
        ds = xr.open_dataset(full_file_path)
        ds.load()
        S = ds.station_id

        if station_result is None:
            station_result = ds
        else:
            station_result = station_result.combine_first(ds)
        ds.close()

        print('[Process %s] File %s aggregated. [%2.1f%%]' % (process_idx, file, (file_idx + 1) / n_files * 100))

    station_result.to_netcdf(DESTINATION + '/station_%s_data.nc' % S)

    return 0


# construct data set with time invariant features per station (-grid)
def getStationNeighbors(stationId, gridHeightData, gridLatData, gridLonData, station_height, station_lat, station_lon):
    # calculate height difference between grid heights and station heights
    gridHeightDifference = gridHeightData.squeeze() - station_height

    # calculate horizontal distance in meters
    grid_lat_lon_zip = np.array(list(zip(
        gridLatData.ravel(), gridLonData.ravel())), dtype=('float32,float32')) \
        .reshape(gridLatData.shape)
    gridHorizontalDistance = np.vectorize(
        lambda lat_lon_zip: DataUtils.haversine(lat_lon_zip[0], lat_lon_zip[1], station_lat, station_lon))(
        grid_lat_lon_zip)

    closest2dId = gridHorizontalDistance.argmin()
    closest2dId = np.unravel_index(closest2dId, (674, 1058))

    closest3dId = (gridHorizontalDistance + 500 * np.abs(gridHeightDifference)).argmin()
    closest3dId = np.unravel_index(closest3dId, (674, 1058))

    return (stationId, ((closest2dId, gridHorizontalDistance[closest2dId], gridHeightDifference[closest2dId]),
                        (closest3dId, gridHorizontalDistance[closest3dId], gridHeightDifference[closest3dId])))
