#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The code was influenced by preprocessing code used on a project of the "Data Science Lab" class in autumn 2018 (Louis de Gaste & Felix Schaumann)
import json
import os
import pickle as pkl
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
# the grids around the stations are stored in a separate file for each station and initialization
def CreateDataByStationAndInit(GridSize, DateBegin, DateEnd, PredictionWindow, ListParam, WithTopo, TopoListParam, isLocal, n_parallel):
    time_begin = time()

    if DateEnd < DateBegin:
        raise Exception('DateEnd is smaller than DateBegin.')

    assert GridSize % 2 == 1, 'Grid size must be an odd number.'

    # different paths, whether we run the script locally or on a cluster node
    if isLocal:
        ADDRESSdata = '/home/n1no/Documents/ethz/master_thesis/code/project/data/cosmo-1/data_subset' # COSMO-1 outputs
        ADDRESStopo = '/home/n1no/Documents/ethz/master_thesis/code/project/data' # base address of topo files
        ADDRESSobst = '/home/n1no/Documents/ethz/master_thesis/code/project/data/observations/' # base adress of obs files
        DESTINATION = '/home/n1no/Documents/ethz/master_thesis/code/project/data/preprocessed_data/station_init/grid_size_' + str(
            GridSize)  # target directory for all generated files
    else:
        ADDRESSdata = '/mnt/data/bhendj/full/cosmo-1' # COSMO-1 outputs
        ADDRESStopo = '/mnt/ds3lab-scratch/ninow/topo' # base address of topo files
        ADDRESSobst = '/mnt/ds3lab-scratch/ninow/observations' # base adress of obs files
        DESTINATION = '/mnt/ds3lab-scratch/ninow/preprocessed_data/station_init/grid_size_' + str(GridSize) # target directory for all generated files

    # create an output folder for each station, based on the station ids
    OBS = xr.open_dataset(ADDRESSobst + '/meteoswiss_t2m_20151001-20180331.nc')
    station_ids = OBS['station_id'].data
    OBS.close()

    station_paths = []
    for S in station_ids:
        # prepare output folders for each station
        station_paths += [DESTINATION + '/Station_' + str(S)]
        if not os.path.exists(station_paths[-1]):
            os.makedirs(station_paths[-1])

    # get all COSMO-1 files that are in the given time interval and have not yet been processed and thus do not
    # already exists in the output folder
    folders = DataUtils.getFilesToProcess(ADDRESSdata, DESTINATION, 'StationAndInit', DateBegin, DateEnd)
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
        indices = np.linspace(0, n_folders, n_parallel+1).astype(int)
        folder_splits = [folders[indices[i]:indices[i + 1]] for i in range(n_parallel)]

    folder_splits = [l for l in folder_splits if len(l) > 0]

    # take timestamp after set-up
    time_setup = time()

    with Pool(processes=n_parallel) as pool:
        # run preprocessing in parallel for all splits and keep the processes in a list to sync them later
        process_results = []
        for idx_split, split in enumerate(folder_splits):
            print('Process %s with range [%s, %s] queued.' % (idx_split, split[0], split[-1]))
            # only calculate topo data by the first process, since it is invariant
            if idx_split == 0:
                isTopo = WithTopo
            else:
                isTopo = 0

            process_results.append(pool.apply_async(GetData, (idx_split, ADDRESSdata, ADDRESStopo, ADDRESSobst,
                                                              DESTINATION, ListParam, TopoListParam, GridSize,
                                                              isTopo, split, PredictionWindow, isLocal)))

        # forces the parent process to wait on all forked children processes
        for ps_idx, ps_result in enumerate(process_results):
            # sync processes
            _ = ps_result.get()
            print('[Process %s] Synchronized after data creation.' % ps_idx)

    # take timestamp after completing all processes
    time_end = time()

    # dump preprocessing information in a descriptive JSON file
    preprocessing_information = {
        'grid_size': GridSize,
        'data_begin': DateBegin,
        'data_end': DateEnd,
        'parameters': ListParam,
        'future_hours': PredictionWindow,
        'n_processes': n_parallel,
        'time_setup': str(timedelta(seconds=(time_setup - time_begin))),
        'time_preprocessing': str(timedelta(seconds=(time_end - time_setup)))
    }

    preprocessing_information_json = json.dumps(preprocessing_information)
    f = open(DESTINATION + '/setup.json', 'w')
    f.write(preprocessing_information_json)
    f.close()

    print('Preprocessing sucessfully finished in %s.' % str(timedelta(seconds=(time_end - time_begin))))

def GetData(processId, ADDRESSdata, ADDRESStopo, ADDRESSobst, DESTINATION, ListParam, TopoListParam,
            GridSize, WithTopo, Files, PredictionWindow, isLocal):
    # processId: (int) -> the id of the process running this method
    # ADDRESSdata: (string) -> base path to COSMO-1 data
    # ADDRESStopo: (string) -> base path to all topology files
    # ADDRESSobs: (string) -> base path to all observation files
    # DESTINATION: (string) -> base path to target output folder
    # GridSize: (int)-> side length of square around each station
    # WithTopo: (bool)-> whether we want to generate preprocessed time invariant features for each station
    # Files: (list(string)) -> list of all files ('yymmddHH') to be processed, e.g. ['15031203', '15031206', ...]
    # PredictionWindow: (list of int) -> all future hours t's [t,t+1,t+2,...] being processed, e.g. y_t, y_t+1,...
    # isLocal: (bool) -> for setting the right paths if the script is running on a local machine or in cloud, etc.

    # path to observation and topological data
    if isLocal:
        OBS = xr.open_dataset(ADDRESSobst + '/meteoswiss_t2m_20151001-20180331.nc')
        TOPO = xr.open_dataset(ADDRESStopo + '/topodata.nc')
    else:
        # to fix parallelization errors, each process gets its own set of TOPO and OBS files
        OBS = xr.open_dataset(ADDRESSobst + '/process_%s/meteoswiss_t2m_20151001-20180331.nc' % processId)
        TOPO = xr.open_dataset(ADDRESStopo + '/process_%s/topodata.nc' % processId)

    # load all station ids
    stationIds = OBS['station_id'].data

    # generate a view on temperature observation at each station
    TempObs = OBS['t2m'].sel(station_id = stationIds)

    # we need to localize the stations on the 1058*674 grid
    GPSgrid = np.dstack((TOPO['lat'][:, :], TOPO['lon'][:, :]))  # 1058*674*2 grid of lon lat values of each square

    # a list with the (lat,lon)-id of the nearest grid point for each station
    closestGridPointPerStation = []
    # a dictionary with the sub-grid around each station
    stationSquaresDict = {}
    # generate sub-grids for each station of the closest GridSize**2 grid points
    for S in stationIds:

        # we compute each grid square's distance with the station
        # and we take the one with the smalles distance to be our reference
        dist = GPSgrid - np.array([[OBS['lat'].sel(station_id = S), OBS['lon'].sel(station_id = S)]])
        dist *= dist
        Id = (dist.sum(axis=2)).argmin()
        Id = np.unravel_index(Id, (674, 1058))

        closestGridPointPerStation += [Id]  # Id=(x,y) coordinates of the station (approx.. to the closest point) on the 1058*674 grid

        SQUARE = {}
        # the variable stationSquaresData contains, for each station, the coord of the squares which form an N*N grid around the station
        SQUARE['lat_idx'] = [x + Id[0] - int(GridSize / 2) for x in range(GridSize)]
        SQUARE['lon_idx'] = [x + Id[1] - int(GridSize / 2) for x in range(GridSize)]

        stationSquaresDict[S] = SQUARE

    # pandas data frame with dictionary of the sub-grid for each station
    stationSquares = pd.DataFrame(data=stationSquaresDict)

    # extract time invariant features for each station and the corresponding sub-grid
    if WithTopo:
        if not os.path.exists(DESTINATION + '/time_invariant_data_per_station.pkl'):
            ds = DataUtils.getTimeInvariantStationFeatures(TOPO=TOPO,
                                                           OBS=OBS,
                                                           stationSquares=stationSquares,
                                                           stationIds=stationIds,
                                                           closestGridPointPerStation=closestGridPointPerStation,
                                                           GridSize=GridSize,
                                                           Features=TopoListParam)
            with open(DESTINATION + '/time_invariant_data_per_station.pkl', 'wb') as handle:
                pkl.dump(ds, handle, protocol=pkl.HIGHEST_PROTOCOL)
            ds.close()
            print('[Process %s] Time invariant features have been processed and stored.' % processId)
        else:
            print('Time invariant features have been found on disk and were therefore not created again.')

    # we now start the iteration through: Each folder, each file, each parameter, each station
    for file in Files:  # loop over all  outputs of COSMO-1, e.g. for 3h interval every day
        try:
            # mark start of preprocessing of n-th file
            print('[Process %s] Start processing %s' % (processId, file))

            # initialize data variables
            DATA = np.zeros((len(stationIds), len(PredictionWindow), GridSize, GridSize, len(ListParam)))
            TempForecast = np.zeros((len(stationIds), len(PredictionWindow)))
            Target = np.zeros((len(stationIds), len(PredictionWindow)))
            TimeStamp = np.zeros((len(PredictionWindow)))
            TimeData = np.zeros((len(PredictionWindow), 5))

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


                # Transform day and hour into a cyclic datetime feature
                days_rad = (DataUtils.passed_days_per_month_dict[int(file[2:4])] + int(file[4:6])) / 365 * (2 * np.pi)
                hours = (int(file[6:8]) + T) % 24
                hour_rad = hours / 24 * (2 * np.pi)

                TimeData[idx_T] = [np.cos(hour_rad), np.sin(hour_rad),
                                             np.cos(days_rad), np.sin(days_rad), T / 33]
                # ______________________________________________________________________

                for P in range(len(ListParam)):
                    MAP = dataset[ListParam[P]].data.squeeze()
                    for idx_S, S in enumerate(stationIds):
                        stationSquare = stationSquares[S]
                        DATA[idx_S, idx_T, :, :, P] = MAP[stationSquare.lat_idx][:, stationSquare.lon_idx]


                # We compare the forecasted temperature with the actual observation
                MAP = np.squeeze(dataset['T']).data
                TempForecast[:, idx_T] = np.array([MAP[x] for x in closestGridPointPerStation])

                TimeStamp[idx_T] = t[0]

                # this dataset is not used anymore and can be closed
                dataset.close()

                try:
                    Target[:, idx_T] = TempObs.sel(time = t).data
                except RuntimeError:
                    print('Error with time=%s.' % t)
                    raise

            # we write the data in a binary file
            for idx_S, S in enumerate(stationIds):
                leads = PredictionWindow
                time_features = ['cos_hour', 'sin_hour', 'cos_day', 'sin_day', 'lead']
                lats = stationSquares[S].lat_idx
                lons = stationSquares[S].lon_idx
                dims = ('lead', 'lat', 'lon', 'feature')
                cosmo_data = xr.DataArray(DATA[idx_S],
                                          dims=dims,
                                          coords=[leads, lats, lons, ListParam])
                temp_forecast = xr.DataArray(TempForecast[idx_S],
                                             dims=('lead'),
                                             coords=[leads])
                temp_station = xr.DataArray(Target[idx_S],
                                            dims=('lead'),
                                            coords=[leads])
                time_data = xr.DataArray(TimeData,
                                         dims=('lead', 'time_feature'),
                                         coords=[leads, time_features])
                time_data.attrs['time_stamp'] = TimeStamp
                ds = xr.Dataset({'cosmo_data': cosmo_data,
                                 'temp_forecast': temp_forecast,
                                 'temp_station': temp_station,
                                 'time_data': time_data})
                ds.attrs['station_id'] = S

                try:
                    # ds.to_netcdf(DESTINATION + '/Station_%s/%s.nc' % (S, file[:8]))
                    with open(DESTINATION + '/Station_%s/%s.pkl' % (S, file[:8]), 'wb') as handle:
                        pkl.dump(ds, handle, protocol=pkl.HIGHEST_PROTOCOL)
                except FileNotFoundError:
                    fileExists = os.path.exists(DESTINATION + '/Station_%s' % S)
                    print('Error that file does not exist, check says: %s' % str(fileExists))
                    raise
                ds.close()

            # print that processing of data point has been completed
            print('[Process %s] Finished %s' % (processId, file))
        except SkipException:
            continue

    OBS.close()
    TOPO.close()

    print('[Process %s] Data split successfully preprocessed!' % processId)

    return 1
