import glob
import json
import operator
import os
import sys
import pickle as pkl
from collections import defaultdict

import utils.PlotUtils as PlotUtils
import utils.data.DataUtils as DataUtils

# this method plots the data distibution of training and test samples over all initializations. An example can be found
# in the appendix of the thesis in the "preprocessing" section
def plotTrainTestDataDistribution(source_path, output_path):
    for train_test_folds_file in glob.glob(source_path + '/train_test_folds_*.pkl', recursive=True):

        with open(train_test_folds_file, 'rb') as f:
            train_test_folds = pkl.load(file=f)
            print('Loaded train/test folds.')
            sys.stdout.flush()
        
        train_date_times = []
        test_date_times = []

        seed = train_test_folds_file.split('_')[-1][:-4]
        file_name = train_test_folds_file.split('/')[-1][:-4]
        
        # cross validation
        for idx, train_test_fold in enumerate(train_test_folds):
            print('Cross-validation test fold %s' % str(idx+1))
            train_fold, test_fold = train_test_fold

            # keep test and train datetimes to calculate distributions
            train_date_times += [list(map(operator.itemgetter(1),train_fold))]
            test_date_times += [list(map(operator.itemgetter(1),test_fold))]
        
        # create folder if necessary
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # # create and dump descriptive json file
        # experiment_info_json = json.dumps(config)
        # f = open(output_path + '/experiment_info.json','w')
        # f.write(experiment_info_json)
        # f.close()
        
        # generate datetime plots
        PlotUtils.plot_datetime_distribution(train_date_times, test_date_times, output_path + '/' + file_name, seed)