from copy import deepcopy
import json
import os

# ATTENTION!!!
# This clacc was used to generate model configurations in an early stage of the work. To be correct with the current
# version of the code, adaptions have to be done.

name_dictionary = {
    "FC" : "FC",
    "layers" : "L",
    "grid_time_invariant" : 'GTI',
    "station_time_invariant" : "STI",
    "time_data" : "TD"
}

def createSingleLayerCNN(max_grid, max_filters):
    configs = []
    grid = 1
    while grid <= max_grid:
        filters = 8
        while filters <= max_filters:
            configs += [{
                'type' : 'CNN',
                'conv_layers' : 1,
                'layers' : 1,
                'grid_conv1' : grid,
                'n_conv1' : filters,
                'stride' : 1
            }]
            filters = 2 * filters
        grid += 1
    return configs


path = "/home/n1no/Documents/ethz/master_thesis/code/project/model/descriptions"
def createFullyConnectedConfigs(max_layers, max_filter, previous_configs=None):
    if max_layers == 0: raise Exception('Layers must not be 0.')
    elif max_layers == 1:
        configs = []
        filters = max_filter
        while filters >= 32:
            if previous_configs is None:
                config = {
                    "type" : "FC",
                    "fc_layers" : 1,
                    "n_fc1" : filters
                }
                configs += [deepcopy(config)]
            else:
                for config in previous_configs:
                    config['fc_layers'] = 1
                    config['n_fc1'] = filters
                    configs += [deepcopy(config)]
            filters = filters // 2
    else:
        previous_configs = createFullyConnectedConfigs(max_layers - 1, max_filter, previous_configs)
        configs = []
        for idx, config in enumerate(previous_configs):
            filters = 32
            while filters < config['n_fc%s' % (max_layers-1)]:
                config['fc_layers'] = max_layers
                config['n_fc%s' % max_layers] =  filters
                configs += [deepcopy(config)]
                filters = 2*filters
    return configs

def setNumOfLayers(previous_configs):
    for config in previous_configs:
        config['layers'] = (config['conv_layers'] if 'conv_layers' in config else 0) + (config['fc_layers'] if 'fc_layers' in config else 0)
    return previous_configs

def createGridTimeInvariantConfigs(previous_configs):
    configs = []
    for config in previous_configs:
        config['grid_time_invariant'] = True
        configs += [deepcopy(config)]
        config['grid_time_invariant'] = False
        configs += [deepcopy(config)]
    return configs

def createStationTimeInvariantConfigs(previous_configs):
    configs = []

    for config in previous_configs:
        config['station_time_invariant'] = False
        configs += [deepcopy(config)]
        config['station_time_invariant'] = True
        configs += [deepcopy(config)]
    return configs

def createTimeDataConfigs(previous_configs):
    configs = []

    for config in previous_configs:
        config['time_data'] = False
        configs += [deepcopy(config)]
        config['time_data'] = True
        configs += [deepcopy(config)]
    return configs

def addNames(previous_configs):
    configs = []
    for config in previous_configs:
        name = config['type']
        if 'conv_layers' in config:
            name += '_CONV'
            for l in range(config['conv_layers']):
                name += "_L%s" % config['conv_layers']
                name += "_N" + str(config['n_conv%s' % str(l + 1)])
                name += "_G" + str(config['grid_conv%s' % str(l + 1)])
            if 'fc_layers' in config:
                name += "_FC"
        if 'fc_layers' in config:
            name += "_L%s" % config['fc_layers']
            for l in range(config['fc_layers']):
                name += "_" + str(config['n_fc%s' % str(l+1)])
        name += "_" + name_dictionary['time_data'] if config['time_data'] else ""
        name += "_" + name_dictionary['grid_time_invariant'] if config['grid_time_invariant'] else ""
        name += "_" + name_dictionary['station_time_invariant'] if config['station_time_invariant'] else ""
        config['name'] = name
        configs += [deepcopy(config)]
    return configs

def dumpConfigs(configs, folder):
    folder_path = path + "/" + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for config in configs:
        with open(folder_path + '/%s.txt' % config['name'], 'w') as file:
            file.write(json.dumps(config))

def generateAndStoreFCConfigs(max_layers, max_filters):
    type = 'FC'
    for max_layer in range(max_layers):
        configs = createFullyConnectedConfigs(max_layers=max_layer + 1, max_filter=max_filters)
        configs = createGridTimeInvariantConfigs(configs)
        configs = createStationTimeInvariantConfigs(configs)
        configs = createTimeDataConfigs(configs)
        configs = setNumOfLayers(configs)
        configs = addNames(configs)

        dumpConfigs(configs, "_".join([type, str(max_layer), str(max_filters)]))

def generateAndStoreCNNConfigs(max_grid, max_filters):
    configs = createSingleLayerCNN(max_grid, max_filters)
    configs = createGridTimeInvariantConfigs(configs)
    configs = createStationTimeInvariantConfigs(configs)
    configs = createTimeDataConfigs(configs)
    configs = setNumOfLayers(configs)
    configs = addNames(configs)

    dumpConfigs(configs, "_".join(["CNN_L1_L0", str(max_grid), str(max_filters)]))

    configs = createSingleLayerCNN(max_grid, max_filters)
    configs = createFullyConnectedConfigs(max_layers=1, max_filter=max_filters, previous_configs=configs)
    configs = createGridTimeInvariantConfigs(configs)
    configs = createStationTimeInvariantConfigs(configs)
    configs = createTimeDataConfigs(configs)
    configs = setNumOfLayers(configs)
    configs = addNames(configs)

    dumpConfigs(configs, "_".join(["CNN_L1_L1", str(max_grid), str(max_filters)]))

    configs = createSingleLayerCNN(max_grid, max_filters)
    configs = createFullyConnectedConfigs(max_layers=2, max_filter=max_filters, previous_configs=configs)
    configs = createGridTimeInvariantConfigs(configs)
    configs = createStationTimeInvariantConfigs(configs)
    configs = createTimeDataConfigs(configs)
    configs = setNumOfLayers(configs)
    configs = addNames(configs)

    dumpConfigs(configs, "_".join(["CNN_L1_L2", str(max_grid), str(max_filters)]))


def main():
    generateAndStoreFCConfigs(max_layers=3, max_filters=1024)
    generateAndStoreCNNConfigs(max_grid=3, max_filters=1024)

    print('Finished...')

if __name__ == '__main__':
    main()