#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from utils import PlotUtils

# thid method plots lat & lon of the gps grid along the x axis for a fixed y-axis, and vise-versa
# to validate the grid representing the real geographic structure
def plot_GPSGrid(grid):
    dim_x, dim_y, _ = grid.shape

    for i in range(0, dim_x):
        for j in range(0, dim_y):
            continue

    for i in range(0, dim_x, 50):
        x = np.arange(dim_x)
        fig = plt.figure()
        ax = plt.subplot(111)
        line_up, = ax.plot(x, grid[x, i, 0], label='Lat')
        line_down, = ax.plot(x, grid[x, i, 1], label='Lon')
        ax.legend(handles=[line_up, line_down])
        plt.title('Plot x-dim 0 to 674')

        fig.savefig('gps_grid_plots/plot_x_dim_' + str(i) + '.png')

        x = np.arange(dim_y)
        fig = plt.figure()
        ax = plt.subplot(111)
        line_up, = ax.plot(x, grid[i, x, 0], label='Lat')
        line_down, = ax.plot(x, grid[i, x, 1], label='Lon')
        ax.legend(handles=[line_up, line_down])
        plt.title('Plot y-dim 0 to 1058')

        fig.savefig('gps_grid_plots/plot_y_dim_' + str(i) + '.png')

# thid method plots lat & lon of the gps grid along the x axis for a fixed y-axis, and vise-versa
# to validate the grid representing the real geographic structure
def scatter_station_grid(grid):
    dim_x, dim_y, _ = grid.shape

    for i in range(0, dim_x):
        for j in range(0, dim_y):
            continue

    for i in range(0, dim_x, 50):
        x = np.arange(dim_x)
        fig = plt.figure()
        ax = plt.subplot(111)
        line_up, = ax.plot(x, grid[x, i, 0], label='Lat')
        line_down, = ax.plot(x, grid[x, i, 1], label='Lon')
        ax.legend(handles=[line_up, line_down])
        plt.title('Plot x-dim 0 to 674')

        fig.savefig('gps_grid_plots/plot_x_dim_' + str(i) + '.png')
# plot station grids on map to validate selection of these grids
def plot_station_grids(station_grids):
    fig = plt.figure(figsize=(30,20))
    ax = plt.subplot(111)

    lon_points = [grid_point for station_grid in station_grids for grid_point in station_grid[1]]
    lat_points = [grid_point for station_grid in station_grids for grid_point in station_grid[0]]

    rlon, rlat = PlotUtils.get_rlon_rlat(lon_points, lat_points)

    # needed to truncate the background map
    # rlat_min, rlat_max, rlon_min, rlon_max = -1.3, 1, -3, 0.5
    rlat_min, rlat_max, rlon_min, rlon_max = np.min(rlat), np.max(rlat), np.min(rlon), np.max(rlon)

    PlotUtils.plot_map_rlon_rlat(ax=ax, rlat_min=rlat_min, rlat_max=rlat_max, rlon_min=rlon_min, rlon_max=rlon_max, alpha_background=0.5)

    ax.scatter(rlon, rlat, s=1, color='red')
    plt.title('Grid Points around Stations')
    plt.axis('scaled')
    fig.savefig(
        '/home/n1no/Documents/ethz/master_thesis/code/project/preprocessing/station_grid_plots/station_grid_plot.png')
    plt.close()

# plot sequence of points in list on map
def plot_station_grid_sequences(station_grids):
    for S, station_grid in enumerate(station_grids):
        lon_points = [grid_point for grid_point in station_grid[1]]
        lat_points = [grid_point for grid_point in station_grid[0]]

        rlon, rlat = PlotUtils.get_rlon_rlat(lon_points, lat_points)

        # needed to truncate the background map
        # rlat_min, rlat_max, rlon_min, rlon_max = -1.3, 1, -3, 0.5
        rlat_min, rlat_max, rlon_min, rlon_max = np.min(rlat), np.max(rlat), np.min(rlon), np.max(rlon)

        for i in range(len(lon_points)):
            fig = plt.figure(figsize=(16, 12))
            ax = plt.subplot(111)

            PlotUtils.plot_map_rlon_rlat(ax=ax, rlat_min=rlat_min-0.2, rlat_max=rlat_max+0.2, rlon_min=rlon_min-0.4, rlon_max=rlon_max+0.4, alpha_background=0.5)
            ax.scatter(rlon[:i+1], rlat[:i+1], s=20, color='red')
            plt.title('Station %s - Grid Point %s' % (S, i))
            plt.axis('scaled')
            fig.savefig(
                '/home/n1no/Documents/ethz/master_thesis/code/project/preprocessing/station_grid_plots/station_%s_grid_point_%s.png' % (S, i))
            plt.close()

        if S >= 3:
            break

def GetData(Stations, GridSize, WithTopo, DateBegin, DateEnd, PredictionWindow, isLocal):
    # Station: (list of int), which contain the stations indexes
    # Grid size: (int)-> nb of square of the length of the grid around the station
    # With topo: (bool)-> if we also want topological parameters(land fraction, height and soil type)
    # Datebegin: (str)->  first date we're interested in, in format (yymmddhh)
    # Dateend: (str)->  last date we're interested in, in format (yymmddhh)
    # PredictionWindow: (list of int) -> all future hours t's [t,t+1,t+2,...] being processed, e.g. y_t, y_t+1,...
    # isLocal: (bool) -> if the script is run on local machine or in cloud for setting the right paths, etc.

    # List of parameters for which data is produced
    ListParam = ['P', 'U', 'V', 'VMAX', 'T', 'TD', 'CLCH', 'CLCM', 'CLCL', 'TOT_PREC', 'ALB_RAD', 'ASOB', 'ATHB',
                 'HPBL']

    # output path is of form: .../data/{username}/preprocessed_data/grid_size_{GridSize]
    ADDRESSdata = '/home/n1no/Documents/ethz/master_thesis/code/project/data/cosmo-1/data_subset'  # folder adress of folders file
    ADDRESStopo = '/home/n1no/Documents/ethz/master_thesis/code/project/data/topodata.nc'  # full adress of topo file
    ADDRESSobst = '/home/n1no/Documents/ethz/master_thesis/code/project/data/observations/meteoswiss_t2m_20151001-20171114.nc'  # full adress of obs file
    DESTINATION = '/home/n1no/Documents/ethz/master_thesis/code/project/data/preprocessed_data/grid_size_' + str(
        GridSize)  # were to put the generated data

    paths = []

    # we load the data we will need frequently
    OBS = nc.Dataset(ADDRESSobst)
    TOPO = nc.Dataset(ADDRESStopo)
    TempObs = np.array(OBS['t2m'][:, Stations]).T

    # we need to localize the stations on the 1058*674 grid
    GPSgrid = np.dstack((TOPO['lat'][:, :], TOPO['lon'][:, :]))  # 1058*674*2 grid of lon lat values of each square

    STATIONid = []
    STATIONsquares = []

    for S in Stations:
        # we compute each grid square's distance with the station
        # and we take the one with the smalles distance to be our reference
        dist = GPSgrid - np.array([[OBS['lat'][S], OBS['lon'][S]]])
        dist *= dist
        Id = (dist.sum(axis=2)).argmin()
        Id = np.unravel_index(Id, (674, 1058))

        STATIONid += [Id]  # Id=(x,y) coordinates of the station on the 1058*674 grid

        # the variable STATIONsquares contains, for each station, the coord of the squares which form an N*N grid around the station
        BUFF = []
        for x in range(GridSize): BUFF += [x + Id[0] - int(GridSize / 2)] * GridSize
        stationSQUARE = [[tuple(BUFF), tuple([x + Id[1] - int(GridSize / 2) for x in range(GridSize)] * GridSize)]]
        STATIONsquares += stationSQUARE

    plot_station_grids(STATIONsquares)

    plot_station_grid_sequences(STATIONsquares)
