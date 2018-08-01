import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# This class contains the definition of the pytorch neural networks. The neural networks are built dynamically based
# on the model configuration file by the "model_factory" method below.

class AbstractNetwork(nn.Module):
    n_station_features = 3
    n_time_features = 5

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def getNumberOfAdditionalFeatures(self, useSTI, useTimeData):
        return (self.n_station_features if useSTI else 0) + (self.n_time_features if useTimeData else 0)

    def getAdditionaFeatureHandling(self, useSTI, useTimeData):
        if useSTI and useTimeData:
            return lambda x, time_features, station_data: torch.cat((x, time_features, station_data), 1)
        elif useSTI:
            return lambda x, _, station_data: torch.cat((x, station_data), 1)
        elif useTimeData:
            return lambda x, time_features, _: torch.cat((x, time_features), 1)
        else:
            return lambda x, *_: x


class CNN(AbstractNetwork):
    def conv_out_size(self, grid_in, grid_conv, stride):
        return int((grid_in - grid_conv) / stride + 1)


class CNN0L(CNN):
    def __init__(self, filter_conv1, grid_conv1, stride, n_parameters, n_time_invariant_parameters, n_grid,
                 useSTI, useTimeData, prediction_times):
        super(CNN0L, self).__init__()
        self.prediciton_times = prediction_times
        self.n_predictions = len(prediction_times)
        self.conv1 = nn.Conv2d(n_parameters + 1 + n_time_invariant_parameters, filter_conv1, grid_conv1)
        self.addNonGridFeatures = self.getAdditionaFeatureHandling(useSTI, useTimeData)
        self.fc1 = nn.Linear(
            filter_conv1 * (self.conv_out_size(n_grid, grid_conv1, stride) ** 2) + self.getNumberOfAdditionalFeatures(
                useSTI, useTimeData), self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.addNonGridFeatures(x, time_features, station_features)
        x = self.fc1(x)
        return x


class CNN1L(CNN0L):
    def __init__(self, filter_conv1, grid_conv1, stride, n_fc1, n_parameters, n_time_invariant_parameters, n_grid,
                 useSTI, useTimeData, prediction_times):
        super(CNN1L, self).__init__(filter_conv1, grid_conv1, stride, n_parameters, n_time_invariant_parameters,
                                    n_grid, useSTI, useTimeData, prediction_times)
        self.fc1 = nn.Linear(
            filter_conv1 * (self.conv_out_size(n_grid, grid_conv1, stride) ** 2) + self.getNumberOfAdditionalFeatures(
                useSTI, useTimeData), n_fc1)
        self.fc2 = nn.Linear(n_fc1, self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = F.relu(super(CNN1L, self).forward(x, time_features, station_features))
        x = self.fc2(x)
        return x


class CNN2L(CNN1L):
    def __init__(self, filter_conv1, grid_conv1, stride, n_fc1, n_fc2, n_parameters, n_time_invariant_parameters,
                 n_grid,
                 useSTI, useTimeData, prediction_times):
        super(CNN2L, self).__init__(filter_conv1, grid_conv1, stride, n_fc1, n_parameters,
                                    n_time_invariant_parameters, n_grid,
                                    useSTI, useTimeData, prediction_times)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = F.relu(super(CNN2L, self).forward(x, time_features, station_features))
        x = self.fc3(x)
        return x


class FullyConnected1L(AbstractNetwork):
    def __init__(self, n_fc1, n_parameters, n_time_invariant_parameters, n_grid, useSTI, useTimeData, prediction_times):
        super(FullyConnected1L, self).__init__()
        self.prediciton_times = prediction_times
        self.n_predictions = len(prediction_times)
        # n_parameters: features of cosmo grid depending on lead time, 1: temperature of lead time = 0,
        # n_time_invariant_parameters: time invariant grid features such as soil type, fraction of land, height diff.
        # to station, etc.
        self.fc1 = nn.Linear((n_parameters + 1 + n_time_invariant_parameters) * (n_grid ** 2) + self.getNumberOfAdditionalFeatures(useSTI, useTimeData), n_fc1)
        self.addNonGridFeatures = self.getAdditionaFeatureHandling(useSTI, useTimeData)
        self.fc2 = nn.Linear(n_fc1, self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = x.view(-1, self.num_flat_features(x))
        x = self.addNonGridFeatures(x, time_features, station_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FullyConnected2L(FullyConnected1L):
    def __init__(self, n_fc1, n_fc2, n_parameters, n_time_invariant_parameters, n_grid, useSTI,
                 useTimeData, prediction_times):
        super(FullyConnected2L, self).__init__(n_fc1, n_parameters, n_time_invariant_parameters, n_grid,
                                               useSTI, useTimeData, prediction_times)
        self.fc2 = nn.Linear(n_fc1 + self.getNumberOfAdditionalFeatures(useSTI, useTimeData), n_fc2)
        self.fc3 = nn.Linear(n_fc2, self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = F.relu(super(FullyConnected2L, self).forward(x, time_features, station_features))
        x = self.fc3(x)
        return x


class FullyConnected3L(FullyConnected2L):
    def __init__(self, n_fc1, n_fc2, n_fc3, n_parameters, n_time_invariant_parameters, n_grid, useSTI,
                 useTimeData, prediction_times):
        super(FullyConnected3L, self).__init__(n_fc1, n_fc2, n_parameters, n_time_invariant_parameters, n_grid,
                                               useSTI, useTimeData, prediction_times)
        self.fc3 = nn.Linear(n_fc2, n_fc3)
        self.fc4 = nn.Linear(n_fc3, self.n_predictions)

    def forward(self, x, time_features, station_features):
        x = F.relu(super(FullyConnected3L, self).forward(x, time_features, station_features))
        x = self.fc4(x)
        return x

class FullyConnectedSingleGrid1L(AbstractNetwork):
    def __init__(self, n_fc1, n_parameters, n_time_invariant_parameters, useSTI, useTimeData, prediction_times, n_points=1, droupout_prob=0):
        super(FullyConnectedSingleGrid1L, self).__init__()
        self.prediciton_times = prediction_times
        self.n_predictions = len(prediction_times)
        # n_parameters: features of cosmo grid depending on lead time, 1: temperature of lead time = 0,
        # n_time_invariant_parameters: time invariant grid features such as soil type, fraction of land, height diff.
        # to station, etc.
        self.addNonGridFeatures = self.getAdditionaFeatureHandling(useSTI, useTimeData)
        self.fc1 = nn.Linear(((n_parameters + 1 + n_time_invariant_parameters) * n_points + self.getNumberOfAdditionalFeatures(useSTI, useTimeData)), n_fc1)
        self.fc2 = nn.Linear(n_fc1, self.n_predictions)
        self.dropout1 = nn.Dropout(p=droupout_prob)
        print('Model uses dropout probability: %s' % droupout_prob)

    def forward(self, x, time_features, station_features):
        try:
            # this awkward squeezing is necessary because we got problems, when the batch is randomly 1 (due to not
            # dividable training or test set size by the batch size)
            x = self.addNonGridFeatures(x.squeeze(dim=-1).squeeze(dim=-1), time_features, station_features)
        except RuntimeError:
            sys.stderr.write("Shapes X, time_features, station_features: " + str(x.shape) + str(time_features.shape) + str(station_features.shape))
            sys.stderr.write("X: " + str(x))
            sys.stderr.write("time_features: ", time_features)
            sys.stderr.write("station_features: ", station_features)
            raise

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class FullyConnectedSingleGrid2L(FullyConnectedSingleGrid1L):
    def __init__(self, n_fc1, n_fc2, n_parameters, n_time_invariant_parameters, useSTI, useTimeData, prediction_times,n_points=1, droupout_prob=0):
        super(FullyConnectedSingleGrid2L, self).__init__(n_fc1, n_parameters, n_time_invariant_parameters,
                                               useSTI, useTimeData, prediction_times,n_points,droupout_prob=droupout_prob)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, self.n_predictions)
        self.dropout2 = nn.Dropout(p=droupout_prob)
        self.i = 0

    def forward(self, x, time_features, station_features):
        x = F.relu(super(FullyConnectedSingleGrid2L, self).forward(x, time_features, station_features))
        x = self.dropout2(x)
        x = self.fc3(x)
        self.i = self.i + 1
        return x


class FullyConnectedSingleGrid3L(FullyConnectedSingleGrid2L):
    def __init__(self, n_fc1, n_fc2, n_fc3, n_parameters, n_time_invariant_parameters, useSTI, useTimeData, prediction_times,n_points=1,droupout_prob=0):
        super(FullyConnectedSingleGrid3L, self).__init__(n_fc1, n_fc2, n_parameters, n_time_invariant_parameters,
                                               useSTI, useTimeData, prediction_times,n_points, droupout_prob=droupout_prob)
        self.fc3 = nn.Linear(n_fc2, n_fc3)
        self.fc4 = nn.Linear(n_fc3, self.n_predictions)
        self.dropout3 = nn.Dropout(p=droupout_prob)

    def forward(self, x, time_features, station_features):
        x = F.relu(super(FullyConnectedSingleGrid3L, self).forward(x, time_features, station_features))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# this method instantiates pytorch neural network objects based on the model configuration file,
# the experiment_parameter.txt file and additional configurations in the Runner.py class.
def model_factory(model_dict, params, time_invariant_params, grid, prediction_times):
    print(model_dict)
    # CNN
    if model_dict['type'] == 'CNN':
        if not 'fc_layers' in model_dict:
            return CNN0L(filter_conv1=model_dict['n_conv1'],
                         grid_conv1=model_dict['grid_conv1'],
                         stride=model_dict['stride'],
                         n_parameters=params,
                         n_time_invariant_parameters=time_invariant_params,
                         n_grid=grid,
                         useSTI=model_dict[
                             'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                         useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                         prediction_times=prediction_times)

        elif model_dict['fc_layers'] == 1:
            return CNN1L(filter_conv1=model_dict['n_conv1'],
                         grid_conv1=model_dict['grid_conv1'],
                         stride=model_dict['stride'],
                         n_fc1=model_dict['n_fc1'],
                         n_parameters=params,
                         n_time_invariant_parameters=time_invariant_params,
                         n_grid=grid,
                         useSTI=model_dict[
                             'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                         useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                         prediction_times=prediction_times)
        elif model_dict['fc_layers'] == 2:
            return CNN2L(filter_conv1=model_dict['n_conv1'],
                         grid_conv1=model_dict['grid_conv1'],
                         stride=model_dict['stride'],
                         n_fc1=model_dict['n_fc1'],
                         n_fc2=model_dict['n_fc2'],
                         n_parameters=params,
                         n_time_invariant_parameters=time_invariant_params,
                         n_grid=grid,
                         useSTI=model_dict[
                             'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                         useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                         prediction_times=prediction_times)

    # Fully connected network
    elif model_dict['type'] == 'FC':
        if grid == 1:
            if 'knn' in model_dict:
                n_points = 3
            else:
                n_points = 1
            if model_dict['fc_layers'] == 1:
                return FullyConnectedSingleGrid1L(n_fc1=model_dict['n_fc1'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times,
                                        n_points=n_points,
                                        droupout_prob=model_dict['dropout_prop'] if 'dropout_prop' in model_dict else 0)
            elif model_dict['fc_layers'] == 2:
                return FullyConnectedSingleGrid2L(n_fc1=model_dict['n_fc1'],
                                        n_fc2=model_dict['n_fc2'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times,
                                        n_points=n_points,
                                        droupout_prob=model_dict['dropout_prop'] if 'dropout_prop' in model_dict else 0)
            elif model_dict['fc_layers'] == 3:
                return FullyConnectedSingleGrid3L(n_fc1=model_dict['n_fc1'],
                                        n_fc2=model_dict['n_fc2'],
                                        n_fc3=model_dict['n_fc3'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times,
                                        n_points=n_points,
                                        droupout_prob=model_dict['dropout_prop'] if 'dropout_prop' in model_dict else 0)

        else:
            if model_dict['fc_layers'] == 1:
                return FullyConnected1L(n_fc1=model_dict['n_fc1'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        n_grid=grid,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times)
            elif model_dict['fc_layers'] == 2:
                return FullyConnected2L(n_fc1=model_dict['n_fc1'],
                                        n_fc2=model_dict['n_fc2'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        n_grid=grid,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times)
            elif model_dict['fc_layers'] == 3:
                return FullyConnected3L(n_fc1=model_dict['n_fc1'],
                                        n_fc2=model_dict['n_fc2'],
                                        n_fc3=model_dict['n_fc3'],
                                        n_parameters=params,
                                        n_time_invariant_parameters=time_invariant_params,
                                        n_grid=grid,
                                        useSTI=model_dict[
                                            'station_time_invariant'] if 'station_time_invariant' in model_dict else True,
                                        useTimeData=model_dict['useTimeData'] if 'useTimeData' in model_dict else True,
                                        prediction_times=prediction_times)

    raise Exception('No matching model found for name \"%s\"' % model_dict['name'])
