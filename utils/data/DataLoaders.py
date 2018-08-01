import numpy as np
import pickle as pkl
import warnings
from multiprocessing import Manager

from utils.data import DataUtils
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

# adapted collate_fn method, to remove samples with corrupted target
def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return batch
    try:
        return default_collate(batch)
    except RuntimeError:
        pass

# DataSet implementation for COSMO-1 data which allows on-the-fly generation of smaller grid sizes than the master
# grid size used in the preprocessing, where the data as a triple of labels, time dependent features and
# time invariant features. The time dependent feature data comes in the form
# (lead, features, grid_vertical, grid_horizontal). If no station_data_dict is provided, the data set will use the per
# station and init time file preprocessed data.
class CosmoDataGridData(Dataset):
    # grid size and number of features used in preprocessing to generate grids
    manager = Manager()
    station_positions = manager.dict()
    station_time_invariant_grid_data = manager.dict()

    def __init__(self, config, files, featureScaling, station_data_dict=None, time_invariant_data=None):
        self.source_path = config['input_source']
        self.files = files
        self.lead_time = config['lead_times']
        # lead time 0 as training input is not supported
        if self.lead_time == 0:
            warnings.warn('Lead time 0 as input is not supported!')
        self.prediction_times = self.lead_time if not 'prediction_times' in config else config['prediction_times']
        self.n_grid_size = config['grid_size']
        self.parameters =config['input_parameters']
        self.n_parameters = len(self.parameters)
        self.featureScaling = featureScaling
        self.time_invariant_data = time_invariant_data
        self.grid_time_invariant_parameters = config['grid_time_invariant_parameters']
        self.n_grid_time_invariant_parameters = len(self.grid_time_invariant_parameters)
        self.station_parameters = config['station_parameters']
        self.n_station_parameters = len(self.station_parameters)
        self.da = None

        # set correct method to load the data depending on the type of preprocessing
        if station_data_dict is None:
            self.loadData = self.loadDataPerStationAndInit
        else:
            self.station_data_dict = station_data_dict
            self.loadData = self.loadDataPerStation

    def calculateLowerAndUpperGridBound(self):
        # lines to crop from each size to get desired "n_grid_size"
        self.master_grid_size = self.da.lat.size
        self.lower_grid_bound = (self.master_grid_size - self.n_grid_size) // 2
        self.upper_grid_bound = self.master_grid_size - (self.master_grid_size - self.n_grid_size + 1) // 2
        self.closest_point_index = self.master_grid_size // 2

    def __getitem__(self, item):
        if item > self.__len__():
            raise Exception('Tried to get data point out of range.')

        stationId, init = self.files[item]
        stationId = int(stationId)
        self.loadData((stationId, init))

        Label = self.da.temp_station.data
        # if target is corrupted return None. This is later filtered out by the custom "collate_fn()" method
        if min(Label) < -1e10 or np.isnan(Label).any():
            return None

        # if this line throws an exception, we are in the first execution of "__getitem__" and thus, we first have to
        # define the indices of the desired leads, parameters and grid bounds for a fast access of the data directly by
        # indices in the following calls.
        try:
            IP2d = self.da.cosmo_data.data[self.lead_idx, self.lower_grid_bound:self.upper_grid_bound,
                   self.lower_grid_bound:self.upper_grid_bound][:, :, self.parameter_idx]
        except AttributeError:
            self.calculateLowerAndUpperGridBound()
            all_leads = list(self.da.coords['lead'].data)
            all_parameters = list(self.da.coords['feature'].data)
            self.lead_idx = all_leads.index(self.lead_time)
            self.prediction_idx = [all_leads.index(pt) for pt in self.prediction_times]
            self.prediction_idx.sort()
            self.parameter_idx = [all_parameters.index(p) for p in self.parameters]
            self.parameter_idx.sort()
            IP2d = self.da.cosmo_data.data[self.lead_idx, self.lower_grid_bound:self.upper_grid_bound, self.lower_grid_bound:self.upper_grid_bound][:,:,self.parameter_idx]

        # keep un-normalized temperature input from COSMO-1
        TEMP_RAW = np.copy(self.da.cosmo_data.data[self.prediction_idx, self.closest_point_index, self.closest_point_index,4 ] - 273.15)

        for p_idx in range(self.n_parameters): IP2d[:,:,p_idx] = self.featureScaling[p_idx](IP2d[:,:,p_idx])

        # add temperature of lead time 0 (initial time of the model run) regardless to the lead time of the prediciton
        TEMP_T0 = self.featureScaling[4](self.da.cosmo_data.data[0, self.lower_grid_bound:self.upper_grid_bound, self.lower_grid_bound:self.upper_grid_bound][:,:,[4]])
        IP2d = np.concatenate((IP2d, TEMP_T0), 2)

        # for CNN appraoch we need a structure like (batch_item, features, lat, lon)
        IP2d = np.rollaxis(IP2d, 2, 0)

        TimeFeatures = self.da.time_data.data[self.lead_idx]
        # TODO at the moment preprocessed data with grid size 3 has normalization of time features and grid size 1 has not
        if self.master_grid_size == 3:
            TimeFeatures[:-1] = DataUtils.normalizeTimeFeatures(TimeFeatures[:-1])

        # get time invariant data for station if already calculated once
        try:
            (TimeInvGrid, TimeInvStation) = self.station_time_invariant_grid_data[stationId]
        except:
            # calculate time invariant data for station for the first time it is used for this station
            station_data = self.time_invariant_data.sel(station=stationId)
            TimeInvStation = station_data.station_position.sel(positinal_attribute=['height', 'lat', 'lon']).data
            TimeInvGrid = np.rollaxis(station_data.grid_data.sel(feature=self.grid_time_invariant_parameters).data[
                                      self.lower_grid_bound:self.upper_grid_bound,
                                      self.lower_grid_bound:self.upper_grid_bound][...,], 2, 0)
            self.station_time_invariant_grid_data[stationId] = (TimeInvGrid, TimeInvStation)

        return Label[self.prediction_idx], np.concatenate((IP2d, TimeInvGrid), 0), TimeFeatures, TimeInvStation, (init, stationId, TEMP_RAW)

    def loadDataPerStationAndInit(self, file):
        # load data set
        with open(self.source_path + self.getFilePath(file), 'rb') as f:
            self.da = pkl.load(f)
        f.close()

    def loadDataPerStation(self, file):
        station, init = file
        self.da = self.station_data_dict[int(station)].sel(init=init)

    def __len__(self):
        return len(self.files)


# DataSet implementation for COSMO-1 data which allows on-the-fly generation input fo 3NN from 3x3-Grid
# grid size used in the preprocessing, where the data as a triple of labels, time dependent features and
# time invariant features. The time dependent feature data comes in the form
# (lead, features, grid_vertical, grid_horizontal). If no station_data_dict is provided, the data set will use the per
# station and init time file preprocessed data.
class CosmoData3NNData(Dataset):
    # grid size and number of features used in preprocessing to generate grids
    manager = Manager()
    station_positions = manager.dict()
    station_time_invariant_grid_data = manager.dict()

    def __init__(self, config, files, featureScaling, station_data_dict=None, time_invariant_data=None):
        self.source_path = config['input_source']
        self.files = files
        self.lead_time = config['lead_times']
        # lead time 0 as training input is not supported
        if self.lead_time == 0:
            warnings.warn('Lead time 0 as input is not supported!')
        self.prediction_times = self.lead_time if not 'prediction_times' in config else config['prediction_times']
        self.parameters = config['input_parameters']
        self.n_parameters = len(self.parameters)
        self.featureScaling = featureScaling
        self.time_invariant_data = time_invariant_data
        self.grid_time_invariant_parameters = config['grid_time_invariant_parameters']
        self.n_grid_time_invariant_parameters = len(self.grid_time_invariant_parameters)
        self.station_parameters = config['station_parameters']
        self.n_station_parameters = len(self.station_parameters)
        self.da = None

        self.station_data_dict = station_data_dict
        self.loadData = self.loadDataPerStation

        # only works on grid size 3
        self.closest_point_index=1

        self.station_closest_points=pkl.load(open(self.source_path +'/station/transformed_closest_points_G_3.pkl','rb'))


    def __getitem__(self, item):
        if item > self.__len__():
            raise Exception('Tried to get data point out of range.')

        stationId, init = self.files[item]
        stationId = int(stationId)
        self.loadData((stationId, init))

        # get position of 3-NN points on 3x3 grid
        station_3nn_lat, station_3nn_lon = self.station_closest_points[stationId]

        Label = self.da.temp_station.data
        # if target is corrupted return None. This is later filtered out by the custom "collate_fn()" method
        if min(Label) < -1e10 or np.isnan(Label).any():
            return None

        # if this line throws an exception, we are in the first execution of "__getitem__" and thus, we first have to
        # define the indices of the desired leads, parameters and grid bounds for a fast access of the data directly by
        # indices in the following calls.
        try:
            IP2d = self.da.cosmo_data.data[self.lead_idx, station_3nn_lat, station_3nn_lon][:, self.parameter_idx]
        except AttributeError:
            all_leads = list(self.da.coords['lead'].data)
            all_parameters = list(self.da.coords['feature'].data)
            self.lead_idx = all_leads.index(self.lead_time)
            self.prediction_idx = [all_leads.index(pt) for pt in self.prediction_times]
            self.prediction_idx.sort()
            self.parameter_idx = [all_parameters.index(p) for p in self.parameters]
            self.parameter_idx.sort()
            IP2d = self.da.cosmo_data.data[self.lead_idx, station_3nn_lat, station_3nn_lon][:, self.parameter_idx]

        # keep un-normalized temperature input from COSMO-1
        TEMP_RAW = np.copy(self.da.cosmo_data.data[
                               self.prediction_idx, self.closest_point_index, self.closest_point_index, 4] - 273.15)

        for p_idx in range(self.n_parameters): IP2d[:, p_idx] = self.featureScaling[p_idx](IP2d[:, p_idx])

        # add temperature of lead time 0 (initial time of the model run) regardless to the lead time of the prediciton
        TEMP_T0 = self.featureScaling[4](self.da.cosmo_data.data[0, station_3nn_lat, station_3nn_lon][:, [4]])
        IP2d = np.concatenate((IP2d, TEMP_T0), 1)

        # for CNN appraoch we need a structure like (batch_item, features, lat, lon)
        IP2d = np.rollaxis(IP2d, 1, 0)

        # load time data and normalize trigonometric representations of hour and month
        TimeFeatures = self.da.time_data.data[self.lead_idx]
        TimeFeatures[:-1] = DataUtils.normalizeTimeFeatures(TimeFeatures[:-1])

        ##TimeFeatures[:-1] = DataUtils.normalizeTimeFeatures(TimeFeatures[:-1])

        # get time invariant data for station if already calculated once
        try:
            (TimeInvGrid, TimeInvStation) = self.station_time_invariant_grid_data[stationId]
        except:
            # calculate time invariant data for station for the first time it is used for this station
            station_data = self.time_invariant_data.sel(station=stationId)
            TimeInvStation = station_data.station_position.sel(positinal_attribute=['height', 'lat', 'lon']).data
            TimeInvGrid = np.rollaxis(station_data.grid_data.sel(feature=self.grid_time_invariant_parameters).data[
                                          station_3nn_lat, station_3nn_lon][...,], 1, 0)
            self.station_time_invariant_grid_data[stationId] = (TimeInvGrid, TimeInvStation)

        return Label[self.prediction_idx], np.concatenate((IP2d, TimeInvGrid), 0).T.flatten(), TimeFeatures, TimeInvStation, (
        init, stationId, TEMP_RAW)

    def loadDataPerStation(self, file):
        station, init = file
        self.da = self.station_data_dict[int(station)].sel(init=init)

    def __len__(self):
        return len(self.files)

# excluding station,label and raw cosmo 2m-temperature precition
class SinglePredictionCosmoData(CosmoDataGridData):

    def __getitem__(self, item):
        try:
            # Only return label for desired lead time. For other results, remove one dimension. init and station
            # are excluded
            return super(SinglePredictionCosmoData, self).__getitem__(item)[:-1]
        # if super method had a "nan" in Label, it returns None
        except TypeError:
            return None

# label is given as error between cosmo output and label
class ErrorPredictionCosmoData(CosmoDataGridData):

    def __getitem__(self, item):
        try:
            # Change the label to be the difference of prediction, not the temperature
            label, cosmo_grid, time_features, time_invariant_station, additional_data = super(ErrorPredictionCosmoData, self).__getitem__(item)

            cosmo_prediction = additional_data[2]
            error_label = cosmo_prediction-label
            additional_data_with_label = additional_data + tuple(label)
            return error_label, cosmo_grid, time_features, time_invariant_station, additional_data_with_label
        # if super method had a "nan" in Label, it returns None
        except TypeError:
            return None

# Extended dataset implementation to load the data for the bias corrected baseline runs.
# Only the LABEL, PREDICTION, STATION, MONTH, HOUR, INIT are returned
class BiasCorrectionCosmoData(CosmoDataGridData):

    distance_metric_mapping = {
        '2d' : 1,
        '3d' : 2
    }

    def __init__(self, config, files, station_data_dict=None):
        self.source_path = config['input_source']
        self.files = files
        self.lead_times = config['lead_times']
        self.station_data_dict = station_data_dict
        self.predictionIdx = self.distance_metric_mapping[config['distance_metric']]

    def __getitem__(self, item):
        if item > self.__len__():
            raise Exception('Tried to get data point out of range.')

        STATION, INIT = self.files[item]
        STATION = int(STATION)

        ds = self.station_data_dict[STATION]

        try:
            # np.array((Label, Prediction_2d, Prediction_3d, Month, Hour))
            DATA = ds.sel(init=INIT).station_data.data[self.lead_idx]
        except AttributeError:
            all_leads = list(ds.coords['lead'].data)
            self.lead_idx = [all_leads.index(l) for l in self.lead_times]
            self.lead_idx.sort()
            DATA = ds.sel(init=INIT).station_data.data[self.lead_idx]

        LABEL = DATA[:,0]
        PREDICTION = DATA[:,self.predictionIdx]
        MONTH = DATA[:,3]
        HOUR = DATA[:,4]


        # if target is corrupted return None. This is later filtered out by the custom "collate_fn()" method
        if min(LABEL) < -1e10 or np.isnan(LABEL).any():
            return None

        return LABEL, PREDICTION, STATION, MONTH, HOUR, INIT
