import os
import glob
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
import matplotlib
import re
import sys
import pickle as pkl

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime
from cycler import cycler
import xarray as xr
from utils.data import DataUtils



type_color_mapping = {
    'train': 'g',
    'test': 'c',
    'filterd': 'y'
}

# generate a dictionary {station_id -> formatted_staion_name}
def get_station_dict(observation_data, station_ids):
    station_name_dict = {}
    for station_id in station_ids:
        station_name_dict[station_id] = str(observation_data['name'].sel(station_id=station_id).data, encoding='utf-8')
    return station_name_dict


def plot_datetime_distribution(train_data, test_data, output_path, seed=None):
    # prepare output folder
    if not output_path.endswith('/'):
        output_path += '/'
    output_path += 'datetime_distribution_plots'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # loop over folds
    for fold in range(len(train_data)):
        # sort train and test data
        train = np.squeeze(np.array(train_data[fold]))
        train.sort()
        test = np.squeeze(np.array(test_data[fold]))
        test.sort()

        # generate data splits by year
        ranges, split_train, split_test = split_year_intervals(train, test)

        # generate histograms
        n_subplots = len(split_train)
        fig, axes = plt.subplots(n_subplots, figsize=(30, 15))
        for year in range(n_subplots):
            axes[year].hist(split_train[year], 365, color='g', range=(ranges[year], ranges[year + 1]),
                            label='Train samples')
            axes[year].set_xlim([ranges[year], ranges[year + 1]])
            ax2 = axes[year].twinx()
            ax2.hist(split_test[year], 365, color='r', range=(ranges[year], ranges[year + 1]), label='Test samples')
            axes[year].set_ylabel('Occurences')

        axes[n_subplots - 1].set_xlabel('Days')
        handles_train, labels_train = axes[n_subplots - 1].get_legend_handles_labels()
        handles_test, labels_test = ax2.get_legend_handles_labels()
        axes[n_subplots - 1].legend([handles_train[0], handles_test[0]], [labels_train[0], labels_test[0]])

        plt.suptitle('Distribution of train and test samples', fontsize=28)
        if seed == None:
            fig.savefig(output_path + '/fold_' + str(fold) + '.png')
        else:
            fig.savefig(output_path + '/fold_' + str(fold) + '_%s.png' % str(seed))
        plt.close()


def split_year_intervals(train, test):
    # calculate first and last data sample
    max_date = max(datetime.strptime(train[train.shape[0] - 1], '%y%m%d%H'),
                   datetime.strptime(test[test.shape[0] - 1], '%y%m%d%H'))
    min_date = min(datetime.strptime(train[0], '%y%m%d%H'), datetime.strptime(test[0], '%y%m%d%H'))
    diff_days = (max_date - min_date).days

    # number of years we have data
    n_years = int(diff_days / 365) + 1

    # transform datetimes into days since the first observation
    train_days = []
    for d in train:
        train_days += [(datetime.strptime(d, '%y%m%d%H') - min_date).days]

    test_days = []
    for d in test:
        test_days += [(datetime.strptime(d, '%y%m%d%H') - min_date).days]

    # calculate ranges for yearly plots
    ranges = [x * 365 for x in range(n_years + 1)]
    split_train = [[] for y in range(n_years)]
    split_test = [[] for y in range(n_years)]

    for x in train_days:
        split = int(x / 365)
        split_train[split] += [x]

    for x in test_days:
        split = int(x / 365)
        split_test[split] += [x]

    return ranges, split_train, split_test


# plot the map of switzerland as background image
def plot_map_rlon_rlat(ax, rlat_min, rlat_max, rlon_min, rlon_max, alpha_background):
    dd = xr.open_dataset(
        "/home/n1no/Documents/ethz/master_thesis/code/project/data/c1ffsurf000_timeinvariant_lonlat.nc")
    dd.set_coords(["rlon", "rlat"], inplace=True)
    swiss_data = dd.sel(rlon=slice(rlon_min, rlon_max), rlat=slice(rlat_min, rlat_max))
    swiss_data.FR_LAND.isel(time=0).plot.pcolormesh("rlon", "rlat", ax=ax, alpha=alpha_background,
                                                    cmap=plt.cm.get_cmap('GnBu_r'), add_colorbar=False)
    swiss_data.HH.isel(time=0).plot.pcolormesh("rlon", "rlat", ax=ax, alpha=0.8 * alpha_background,
                                               cmap=plt.cm.get_cmap('YlGn'), add_colorbar=False)


def get_rlon_rlat(grid_points_lon, grid_points_lat):
    dd = xr.open_dataset(
        "/home/n1no/Documents/ethz/master_thesis/code/project/data/c1ffsurf000_timeinvariant_lonlat.nc")
    rlon = dd['rlon'][grid_points_lon]
    rlat = dd['rlat'][grid_points_lat]
    return rlon, rlat


def plot_hour_feature_verification(output_path, hour_feature_dict):
    x, y, color_values = [], [], []

    for key, value in hour_feature_dict.items():
        x.append(key[0])
        y.append(key[1])
        color_values.append(value[0])

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(np.asarray(x), np.asarray(y), c=color_values, s=500, cmap='magma')
    fig.savefig(output_path + '/hour_feature_verification.png')
    plt.close()


def plot_month_feature_verification(output_path, month_feature_dict):
    x, y, color_values = [], [], []

    for key, value in month_feature_dict.items():
        x.append(key[0])
        y.append(key[1])
        color_values.append(value[0])

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(np.asarray(x), np.asarray(y), c=color_values, s=300, cmap='magma')
    fig.savefig(output_path + '/month_feature_verification.png')
    plt.close()


def plotFeatureDistribution(output_path, config,
                            train_features, train_time_invariant_grid_features, train_station_features, train_labels,
                            test_features, test_time_invariant_grid_features, test_station_features, test_labels):
    for idx, train_input in enumerate(train_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(train_input, 100)
        plt.suptitle("Train Run %s, Feature %s" % (config['run'], config['features'][idx]))
        fig.savefig(output_path + '/train_run_%s_%s.png' % (config['run'], config['features'][idx]))
        plt.close()

    for idx, train_grid_feature in enumerate(train_time_invariant_grid_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(train_grid_feature, 100)
        plt.suptitle(
            "Train Run %s, Time Invariant Grid Feature %s" % (config['run'], config['time_invariant_features'][idx]))
        fig.savefig(output_path + '/train_run_%s_ti_grid_feature_%s.png' % (
            config['run'], config['time_invariant_features'][idx]))
        plt.close()

    for idx, train_station_feature in enumerate(train_station_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(train_station_feature, 100)
        plt.suptitle("Train Run %s, StationFeature %s" % (config['run'], config['station_parameters'][idx]))
        fig.savefig(
            output_path + '/train_run_%s_station_feature_%s.png' % (config['run'], config['station_parameters'][idx]))
        plt.close()

    for idx, test_input in enumerate(test_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(test_input, 100)
        plt.suptitle("Test Run %s, Feature %s" % (config['run'], config['features']))
        fig.savefig(output_path + '/test_run_%s_%s.png' % (config['run'], config['features'][idx]))
        plt.close()

    for idx, test_grid_feature in enumerate(test_time_invariant_grid_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(test_grid_feature, 100)
        plt.suptitle(
            "Test Run %s, Time Invariant Grid Feature %s" % (config['run'], config['time_invariant_features'][idx]))
        fig.savefig(output_path + '/test_run_%s_ti_grid_feature_%s.png' % (
            config['run'], config['time_invariant_features'][idx]))
        plt.close()

    for idx, test_station_feature in enumerate(test_station_features):
        fig = plt.figure(figsize=(15, 15))
        plt.hist(test_station_feature, 100)
        plt.suptitle("Test Run %s, StationFeature %s" % (config['run'], config['station_parameters'][idx]))
        fig.savefig(
            output_path + '/test_run_%s_station_feature_%s.png' % (config['run'], config['station_parameters'][idx]))
        plt.close()

    fig = plt.figure(figsize=(15, 15))
    plt.hist(train_labels, 100)
    fig.savefig(output_path + '/train_labels_run_%s.png' % (config['run']))
    plt.close()

    fig = plt.figure(figsize=(15, 15))
    plt.hist(test_labels, 100)
    fig.savefig(output_path + '/test_labels_run_%s.png' % (config['run']))
    plt.close()


def plotExperimentErrorEvaluation(source_path):
    output_path = source_path + '/plots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    error_statistics, epoch_results, best_results, run_times, samples_statistic = [], [], [], [], []
    runs = 0
    for path in glob.glob(source_path + '/**/experiment_statistic.pkl', recursive=True):
        model_name = re.search(r'/([^/]+)/experiment_statistic.pkl', path).group(1)
        with open(path, 'rb') as file:
            ds = pkl.load(file)

        if runs == 0:
            runs = ds.runs

        # get times for which prediction were made
        prediction_times = ds.error_statistic.prediction_times.data

        # get the mean of train and test error over all runs, additionally calculate the std. dev. to the mean
        # run_error_mean_std = tuple(mean, std), mean = ndarray(epoch, error), error = train or test error value,
        # lead = prediction for T = lead
        epoch_error_mean_std = (np.mean(ds.error_statistic.data, 0), np.std(ds.error_statistic.data, 0))
        # look up in train/test mean error per epoch, for which epoch the train resp. test error is the best and
        # return the index
        best_train_epoch, best_test_epoch = np.argmin(epoch_error_mean_std[0], 0)
        # advanced indexing: first select mean or variance. then define fore each prediction lead time the best epoch,
        # the error (train = 0 or test = 1) and the prediction lead time itself.
        # This reduces the [n_epoch, 2, n_future_predictions] -> [n__future_predictions
        best_error_mean_std_epoch = (epoch_error_mean_std[0][best_train_epoch, 0, range(len(prediction_times))],
                                     epoch_error_mean_std[1][best_train_epoch, 0, range(len(prediction_times))],
                                     best_train_epoch,
                                     epoch_error_mean_std[0][best_test_epoch, 1, range(len(prediction_times))],
                                     epoch_error_mean_std[1][best_test_epoch, 1, range(len(prediction_times))],
                                     best_test_epoch)

        # get min, max, mean, std time over all runs and epochs
        time_statistic = ds.run_time_statistic.data
        # first tuple element: name, second tuple element: data [axis 0: [min, max, mean, std], axis 1: epoch, train, test]
        run_times += [(model_name, np.array([np.min(time_statistic, axis=(0, 1)),
                                             np.max(time_statistic, axis=(0, 1)),
                                             np.mean(time_statistic, axis=(0, 1)),
                                             np.std(time_statistic, axis=(0, 1))]))]

        error_statistics += [(model_name, ds.error_statistic.data)]
        epoch_results += [(model_name, epoch_error_mean_std)]
        best_results += [(model_name, best_error_mean_std_epoch)]

        samples_statistic += [(model_name, np.mean(ds.samples_statistic.data, axis=(0, 1)))]

    plotErrorOverEpochsForSingleExperiment(output_path=output_path, error_statistics=error_statistics,
                                           prediction_times=prediction_times)
    plotBestResult(output_path=output_path, best_results=best_results, prediction_times=prediction_times)
    plotCombinedErrorOverEpochs(output_path=output_path, results=epoch_results, prediction_times=prediction_times)
    plotRunTimes(output_path=output_path, timings=run_times)
    plotSampleStatistic(output_path=output_path, samples_statistic=samples_statistic)
    generateTrainingResultTable(output_path=output_path, run_times=run_times, epoch_results=epoch_results, best_results=best_results,
                  prediction_times=prediction_times)


def plotCombinedErrorOverEpochs(output_path, results, prediction_times):
    # plot train errors
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
    styles = ['-', '--', ':', '-.']

    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=(cycler('color', colors * len(styles)) +
                               cycler('linestyle',
                                      [style for same_style in [[s] * len(colors) for s in styles] for style in
                                       same_style])))

    for prediction_time_idx, prediction_time in enumerate(prediction_times):
        current_output_path = output_path + '/prediction_T%s' % prediction_time
        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)

        fig, ax = plt.subplots(figsize=(20, 10))

        for name, error in results:
            x = list(range(error[0][:, 0, prediction_time_idx].size))
            ax.plot(x, error[0][:, 0, prediction_time_idx], label=name)

        plt.grid()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('Train Error', fontsize=32)
        plt.xticks(x, ['Epoch %s' % i for i in x])
        ax.legend(loc='best', fancybox=True, framealpha=0.4)

        fig.savefig(current_output_path + '/train_error.png')
        plt.close()

        # plot test errors
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', colors * len(styles)) +
                                   cycler('linestyle',
                                          [style for same_style in [[s] * len(colors) for s in styles] for style in
                                           same_style])))

        fig, ax = plt.subplots(figsize=(20, 10))

        for name, error in results:
            ax.plot(x, error[0][:, 1, prediction_time_idx], label=name)

        plt.grid()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('Test Error', fontsize=32)
        plt.xticks(x, ['Epoch %s' % i for i in x])
        ax.legend(loc='best', fancybox=True, framealpha=0.4)

        fig.savefig(current_output_path + '/test_error.png')
        plt.close()


def plotErrorOverEpochsForSingleExperiment(output_path, error_statistics, prediction_times):
    for idx, (name, error_statistic) in enumerate(error_statistics):
        # shape of error statistics: dim_0=Runs, dim_1=Epochs, dim_2=[train_err, test_err]
        runs, epochs, *_ = error_statistic.shape

        for prediction_time_idx, prediction_time in enumerate(prediction_times):

            current_output_path = output_path + '/prediction_T%s' % prediction_time
            if not os.path.exists(current_output_path):
                os.makedirs(current_output_path)

            x = [i + 1 for i in range(epochs)]

            fig, axes = plt.subplots(2, sharey=True, figsize=(20, 20))
            for run in range(runs):
                axes[0].plot(x, error_statistic[run, :, 0, prediction_time_idx], alpha=0.5, label='Run %s' % str(run))
                axes[1].plot(x, error_statistic[run, :, 1, prediction_time_idx], alpha=0.5, label='Run %s' % str(run))
            axes[0].plot(x, np.mean(error_statistic[:, :, 0, prediction_time_idx], axis=0), 'r--', label='Mean')
            axes[1].plot(x, np.mean(error_statistic[:, :, 1, prediction_time_idx], axis=0), 'r--', label='Mean')

            axes[1].set_xlabel('Epoch', fontsize=16)

            axes[0].set_title('Train Error', fontsize=16)
            axes[1].set_title('Test Error', fontsize=16)
            for i in range(2):
                axes[i].set_xticks(x)
                axes[i].set_ylabel('RMSE [°C]', fontsize=16)
                axes[i].grid(True)
                axes[i].legend()

            axes[0].text(0.05, 0.1,
                         'Min. Mean Error: %s' % "{:2.3f}".format(
                             np.min(np.mean(error_statistic[:, :, 0, prediction_time_idx], axis=0))),
                         fontsize=18, fontweight='bold', transform=axes[0].transAxes)
            axes[1].text(0.05, 0.1,
                         'Min. Mean Error: %s' % "{:2.3f}".format(
                             np.min(np.mean(error_statistic[:, :, 1, prediction_time_idx], axis=0))),
                         fontsize=18, fontweight='bold', transform=axes[1].transAxes)
            axes[0].text(0.05, 0.05,
                         'Min. Error: %s' % "{:2.3f}".format(np.min(error_statistic[:, :, 0, prediction_time_idx])),
                         fontsize=18,
                         transform=axes[0].transAxes)
            axes[1].text(0.05, 0.05,
                         'Min. Error: %s' % "{:2.3f}".format(np.min(error_statistic[:, :, 1, prediction_time_idx])),
                         fontsize=18,
                         transform=axes[1].transAxes)

            plt.ylim(0, 1.2 * np.max(error_statistic[:, :, 0, prediction_time_idx]))
            plt.suptitle(str(" ".join(name.split("_")) + " (Lead Prediction = %s)" % prediction_time), fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=4)
            fig.savefig(current_output_path + '/%s_error_plot.png' % name)
            plt.close()


# best results is a list of the best results for each experiment. The best result is a tuple with train in first
# and test in second position. For each of train and test error, the entry is a triple of
# the form (mean error, std. dev., epoch) representing the best result
def plotBestResult(output_path, best_results, prediction_times):
    # create path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # plot train error
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=(cycler('color', colors)))

    for prediction_time_idx, prediction_time in enumerate(prediction_times):

        current_output_path = output_path + '/prediction_T%s' % prediction_time
        if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)

        best_train_results = sorted(best_results, key=lambda x: x[1][0][prediction_time_idx])
        fig, ax = plt.subplots(figsize=(20, 10))

        for idx, result in enumerate(best_train_results):
            mean_train = result[1][0]
            std_train = result[1][1]
            mean_test = result[1][3]
            std_test = result[1][4]
            ax.bar(x=idx, height=mean_train, width=0.15, yerr=std_train, label=result[0],
                   color=colors[idx % len(colors)], hatch='x')
            ax.bar(x=idx + 0.2, height=mean_test, width=0.15, yerr=std_test,
                   color=colors[idx % len(colors)], hatch='o')

        ax.set_xticks([])
        plt.grid()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('Best Result of all Epochs for each Model averaged over all Runs (sorted by training)', fontsize=32)
        ax.legend(loc='best', fancybox=True, framealpha=0.4)

        fig.savefig(current_output_path + '/best_result_by_train_error.png')
        plt.close()

        # plot test error
        best_test_results = sorted(best_results, key=lambda x: x[1][3][prediction_time_idx])
        fig, ax = plt.subplots(figsize=(20, 10))

        for idx, result in enumerate(best_test_results):
            mean_train = result[1][0]
            std_train = result[1][1]
            mean_test = result[1][3]
            std_test = result[1][4]
            ax.bar(x=idx, height=mean_train, width=0.15, yerr=std_train, label=result[0],
                   color=colors[idx % len(colors)], hatch='x')
            ax.bar(x=idx + 0.2, height=mean_test, width=0.15, yerr=std_test,
                   color=colors[idx % len(colors)], hatch='o')

        ax.set_xticks([])
        plt.grid()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('Best Result of all Epochs for each Model averaged over all Runs (sorted by test)', fontsize=32)
        ax.legend(loc='best', fancybox=True, framealpha=0.4)

        fig.savefig(current_output_path + '/best_result_by_test_error.png')
        plt.close()


def plotRunTimes(output_path, timings):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    n_experiments = len(timings)
    x_base = list(range(0, 3))

    fig, ax = plt.subplots(figsize=(20, 10))
    for idx, (model_name, timing) in enumerate(timings):
        shift = (idx / n_experiments / 2)
        x = [pos + shift for pos in x_base]
        ax.bar(x=x, height=timing[0:3, 0], width=0.05, label=model_name, color=colors[idx % len(colors)])

    plt.xticks([pos + ((n_experiments - 1) / 2) / n_experiments / 2 for pos in x_base],
               ('Fastest Time', 'Slowest Time', 'Mean Time'))
    plt.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title('Min/Max/Mean Time over all Epochs and Runs', fontsize=32)
    ax.legend(loc='best', fancybox=True, framealpha=0.4)

    fig.savefig(output_path + '/epoch_run_times.png')
    plt.close()


def plotSampleStatistic(output_path, samples_statistic):
    # ------------------------------------------------------------------------------------------------------------------------
    # skipped samples statistics due to corrupted measurement stations
    # ------------------------------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(20, 10))
    N = len(samples_statistic)
    ind = np.arange(N)  # the x locations for the groups
    experiment_title = [splitModelNameVertical(name) for name, _ in samples_statistic]
    data = np.zeros((4, N))
    for idx, (_, samples) in enumerate(samples_statistic):
        data[:, idx] = np.array([samples[1], samples[0] - samples[1], samples[3], samples[2] - samples[3]])

    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, data[0, :], width)
    p2 = plt.bar(ind, data[1, :], width, bottom=data[0, :])
    p3 = plt.bar(ind, data[2, :], width, bottom=np.sum(data[0:2, :], axis=0))
    p4 = plt.bar(ind, data[3, :], width, bottom=np.sum(data[0:3, :], axis=0))

    plt.ylabel('Samples')
    plt.title('Processed / Skipped Samples', fontsize=22)
    plt.xticks(ind, experiment_title)
    plt.xlim((-0.5, np.max(ind) + 1))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Processed Train', 'Skipped Train', 'Processed Test', 'Skipped Test'),
               loc='upper right')
    fig.savefig(output_path + '/skip_sample_statistic.png')
    plt.close()


def splitModelNameVertical(name):
    n = 5
    split = name.split('_')
    lines = []
    for i in range(0, len(split), n):
        lines += [" ".join(split[i:i + n])]
    return "\n".join(lines)


def generateTrainingResultTable(output_path, run_times, epoch_results, best_results, prediction_times):
    with open(output_path + '/table_data.csv', "w") as csv_file:
        title_per_epoch = ["Train Mean, Train Std., Test Mean, Test Std."]
        csv_file.write(",".join(["Name, Lead Prediction"] + title_per_epoch * epoch_results[0][1][0].shape[0] + [
            "Best Train Mean, Beat Train Std., Best Test Mean, Best Test Std.\n"]))

        for prediction_time_idx, prediction_time in enumerate(prediction_times):
            for idx, (name, epoch_result) in enumerate(epoch_results):
                # first stack train mean error and std to one tuple and test mean error and std to other tuple, resulting
                # in array([[mean_train_ep_1, std_train_ep_1], ..., [mean_train_ep_n, mean_train_ep_n]]) and
                # array([[mean_test_ep_1, std_test_ep_1], ..., [mean_test_ep_n, mean_test_ep_n]]). Stack those two
                # horizontally to get array([[mean_train_ep_1, std_train_ep_1, mean_test_ep_1, std_test_ep_1], ...,
                # [mean_train_ep_n, mean_train_ep_n, mean_test_ep_n, mean_test_ep_n]]). Finally, flatten to list.
                results = [str(elem) for elem in
                           np.hstack((np.stack(epoch_result, axis=3)[:, 0, prediction_time_idx], np.stack(epoch_result, axis=3)[:, 1, prediction_time_idx])).flatten().tolist()]
                best_train_mean, best_train_std, _, best_test_mean, best_test_std, _ = tuple(map(lambda arr: arr[prediction_time_idx], best_results[idx][1]))

                csv_file.write(",".join([name, str(prediction_time)] + results + [str(elem) for elem in
                                                                                  [best_train_mean, best_train_std,
                                                                                   best_test_mean,
                                                                                   best_test_std]] + ["\n"]))


# plot the error from the bias corrected baseline run. The error statistic has the following dimensions (runs, errors),
# where errors = [raw_error, station_bias_corrected, station_hour_bias_corrected, station_hour_month_bias_corrected']
def plotBiasCorrectedBaselineResults(source_path):
    output_path = source_path + '/plots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    error_data = {
        '2d': None,
        '3d': None
    }
    for path in glob.glob(source_path + '/**/experiment_statistic.pkl', recursive=True):
        with open(path, 'rb') as file:
            ds = pkl.load(file)
        distance_metric = ds.attrs['experiment_info']['distance_metric']
        error_statistic = ds.error_statistic

        try:
            error_data[distance_metric] = xr.concat([error_data[distance_metric], error_statistic], dim='run')
        except TypeError:
            error_data[distance_metric] = error_statistic

    lead_error = {}
    for metric_idx, distance_metric_key in enumerate(error_data.keys()):
        error_by_metric = error_data[distance_metric_key]
        if error_by_metric is None: continue
        mean_squared_error_data = np.sqrt(np.nanmean(np.square(error_by_metric.data), axis=(2, 3, 4, 6)))
        lead_error[distance_metric_key] = mean_squared_error_data

        # TODO nwi plot stuff...


def plotPredictionRun(source_path, observation_path, n_parallel=1):
    plotAveragedPredictionRun(source_path)
    plotPerStationPredictionRun(source_path, observation_path, n_parallel)


def plotAveragedPredictionRun(source_path):
    # gather all models in source folder
    error_data_per_run_dict = {}
    for path in glob.glob(source_path + '/**/model_run_error.pkl', recursive=True):
        model_name = path.split('/')[-2]
        with open(path, 'rb') as file:
            ds = pkl.load(file)
        for data_var in ds.data_vars:
            inits = ds[data_var].init.data
            sample_type_mapping = [mapping[1] for mapping in ds[data_var].init_type_mapping]
            prediction_data = ds[data_var].data
            try:
                error_data_per_run_dict[data_var] += [(model_name, inits, prediction_data, sample_type_mapping)]
            except:
                error_data_per_run_dict[data_var] = [(model_name, inits, prediction_data, sample_type_mapping)]


    # get the prediction lead time to adjust time labels
    prediciton_lead_time = ds.attrs['config']['prediction_times'][0] if 'config' in ds.attrs else 1

    times = DataUtils.getTimeFromFileName(inits, prediciton_lead_time)
    time_labels = [str(t)[:-13] for t in times]

    for run_error_data in error_data_per_run_dict.items():
        run = run_error_data[0]
        model_mean_errors = {}
        n_subplots = 10
        fig, axes = plt.subplots(n_subplots, figsize=(60, 20), sharey=True)
        for model_idx, model_error_data in enumerate(run_error_data[1]):

            N = len(model_error_data[1])
            split_length = N // n_subplots
            ind = np.arange(N)  # the x locations for the groups
            experiment_title = model_error_data[0]
            prediction_data = model_error_data[2]
            init_type_mapping = model_error_data[3]
            train_indices = [idx for idx, item in enumerate(init_type_mapping) if item == 'train']
            test_indices = [idx for idx, item in enumerate(init_type_mapping) if item == 'test']
            filtered_indices = [idx for idx, item in enumerate(init_type_mapping) if item == 'filterd']


            for i in range(n_subplots):
                # split indexes into slices for each subplot
                index_split = ind[i * split_length:(i + 1) * split_length]

                if model_idx == 0:
                    sampleTypeBackgroundColoring(axes[i], index_split,
                                                 init_type_mapping[i * split_length:(i + 1) * split_length])
                    axes[i].set_xlim([np.min(index_split), np.max(index_split)])

                axes[i].plot(index_split,
                             np.nanmean(prediction_data[i * split_length:(i + 1) * split_length,:, 0],axis=1),
                             label=experiment_title, linewidth=0.15, alpha=0.8)

            train_model_bias = np.nanmean(prediction_data[train_indices][:,:,3])
            train_model_rmse = np.sqrt(np.nanmean(np.square(prediction_data[train_indices][:,:,3])))
            train_model_mae = np.nanmean(np.absolute(prediction_data[train_indices][:,:,3]))

            test_model_bias = np.nanmean(prediction_data[test_indices][:,:,3])
            test_model_rmse = np.sqrt(np.nanmean(np.square(prediction_data[test_indices][:,:,3])))
            test_model_mae = np.nanmean(np.absolute(prediction_data[test_indices][:,:,3]))
            
            filtered_model_bias = np.nanmean(prediction_data[filtered_indices][:,:,3])
            filtered_model_rmse = np.sqrt(np.nanmean(np.square(prediction_data[filtered_indices][:,:,3])))
            filtered_model_mae = np.nanmean(np.absolute(prediction_data[filtered_indices][:,:,3]))


            model_mean_errors[experiment_title] = (train_model_bias, train_model_rmse, train_model_mae,
                                                   test_model_bias, test_model_rmse, test_model_mae,
                                                   filtered_model_bias, filtered_model_rmse, filtered_model_mae)


        # add mean errors of cosmo output predictions
        train_diff_cosmo = prediction_data[train_indices][:,:,1] - prediction_data[train_indices][:,:,2]
        train_cosmo_bias = np.nanmean(train_diff_cosmo)
        train_cosmo_rmse = np.sqrt(np.nanmean(np.square(train_diff_cosmo)))
        train_cosmo_mae = np.nanmean(np.absolute(train_diff_cosmo))

        test_diff_cosmo = prediction_data[test_indices][:,:,1] - prediction_data[test_indices][:,:,2]
        test_cosmo_bias = np.nanmean(test_diff_cosmo)
        test_cosmo_rmse = np.sqrt(np.nanmean(np.square(test_diff_cosmo)))
        test_cosmo_mae = np.nanmean(np.absolute(test_diff_cosmo))
        
        filtered_diff_cosmo = prediction_data[filtered_indices][:,:,1] - prediction_data[filtered_indices][:,:,2]
        filtered_cosmo_bias = np.nanmean(filtered_diff_cosmo)
        filtered_cosmo_rmse = np.sqrt(np.nanmean(np.square(filtered_diff_cosmo)))
        filtered_cosmo_mae = np.nanmean(np.absolute(filtered_diff_cosmo))
        
        # add COSMO-1 output prediction error
        model_mean_errors['COSMO-1'] = (train_cosmo_bias, train_cosmo_rmse, train_cosmo_mae,
                                        test_cosmo_bias, test_cosmo_rmse, test_cosmo_mae,
                                        filtered_cosmo_bias, filtered_cosmo_rmse, filtered_cosmo_mae)

        for i in range(n_subplots):
            axes[i].plot(ind[i * split_length:(i + 1) * split_length],
                         np.nanmean(prediction_data[i * split_length:(i + 1) * split_length,:, 1], axis=1), label='COSMO-1',
                         linewidth=0.15, alpha=0.8, color='b', linestyle='-.')
            axes[i].plot(ind[i * split_length:(i + 1) * split_length],
                         np.nanmean(prediction_data[i * split_length:(i + 1) * split_length, 2], axis=1), label='Prediction',
                         linewidth=0.15, alpha=0.8, color='m', linestyle='--')

            tick_step_size = np.maximum(split_length // 30, 1)
            axes[i].set_xticks(ind[i * split_length:(i + 1) * split_length][::tick_step_size])
            axes[i].set_xticklabels(time_labels[i * split_length:(i + 1) * split_length][::tick_step_size])
            axes[i].set_xticks(ind[i * split_length:(i + 1) * split_length], minor=True)
            # And a corresponding grid
            axes[i].grid(which='both')

            # Or if you want different settings for the grids:
            axes[i].grid(which='minor', alpha=0.2)
            axes[i].grid(which='major', alpha=0.5)

            handles, labels = axes[0].get_legend_handles_labels()

        axes[n_subplots - 1].set_xlabel('Time')
        axes[0].legend(handles, labels)
        plt.tight_layout()

        run_path = source_path + '/plots/prediction_runs/%s' % run
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        fig.savefig(run_path + '/averaged_prediction.png', dpi=300)

        generatePredictionResultTable(output_path=run_path, results=model_mean_errors)


def plotPerStationPredictionRun(source_path, observation_path, n_parallel):
    # gather all models in source folder
    error_data_per_run_dict = defaultdict()
    for path in glob.glob(source_path + '/**/model_run_error.pkl', recursive=True):
        model_name = path.split('/')[-2]
        with open(path, 'rb') as file:
            ds = pkl.load(file)
        for data_var in ds.data_vars:
            da = ds[data_var]
            try:
                error_data_per_run_dict[data_var] += [(model_name, da)]
            except:
                error_data_per_run_dict[data_var] = [(model_name, da)]

    # load observations
    OBS = xr.open_dataset(observation_path)
    # get the prediction lead time to adjust time labels
    prediciton_lead_time = ds.attrs['config']['prediction_times'][0] if 'config' in ds.attrs else 1

    for run_error_data in error_data_per_run_dict.items():
        run = run_error_data[0]
        models = run_error_data[1]
        stations = run_error_data[1][0][1].station.data
        inits = run_error_data[1][0][1].init.data
        init_type_mapping = np.array(run_error_data[1][0][1].init_type_mapping)
        train_indices = [idx for idx, item in enumerate(init_type_mapping) if item[1] == 'train']
        test_indices = [idx for idx, item in enumerate(init_type_mapping) if item[1] == 'test']
        sample_type_color_mapping = [mapping[1] for mapping in init_type_mapping]
        times = DataUtils.getTimeFromFileName(inits, prediciton_lead_time)
        time_labels = [str(t)[:-13] for t in times]

        station_name_dict = get_station_dict(OBS, stations)

        model_station_mean_errors = {}
        # plot for each station the prediction run results in parallel
        with Pool(processes=n_parallel) as pool:
            process_results = []

            for station_idx, station in enumerate(stations):
                print('Plotting of prediction run for station %s queued.' % station)
                process_results.append(pool.apply_async(plotPerStationPredictionRunWorker,
                                                        (models, station, train_indices, test_indices,
                                                         station_name_dict,sample_type_color_mapping,
                                                         time_labels, source_path, run)))

            # aggregate results from all processes
            for ps_idx, ps_result in enumerate(process_results):
                # sync processes
                model_station_mean_error = ps_result.get()

                for experiment_title, station_data_list in model_station_mean_error.items():
                    try:
                        model_station_mean_errors[experiment_title] += station_data_list
                    except KeyError:
                        model_station_mean_errors[experiment_title] = station_data_list

                print('[Process %s] Synchronized after plotting station.' % ps_idx)

        run_path = source_path + '/plots/prediction_runs/%s' % run
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        generateStationPredictionResultTable(output_path=run_path, results=model_station_mean_errors)

def plotPerStationPredictionRunWorker(models, station, train_indices, test_indices, station_name_dict,
                                      sample_type_color_mapping, time_labels, source_path, run):
    n_subplots = 10
    model_station_mean_error = {}
    fig, axes = plt.subplots(n_subplots, figsize=(60, 20), sharey=True)

    for model_idx, model_error_data in enumerate(models):
        experiment_title = model_error_data[0]

        station_error_data = model_error_data[1].sel(station=station).data

        train_station_bias = np.nanmean(station_error_data[train_indices][:, 3])
        train_station_rmse = np.sqrt(np.nanmean(np.square(station_error_data[train_indices][:, 3])))
        train_station_mae = np.nanmean(np.absolute(station_error_data[train_indices][:, 3]))

        test_station_bias = np.nanmean(station_error_data[test_indices][:, 3])
        test_station_rmse = np.sqrt(np.nanmean(np.square(station_error_data[test_indices][:, 3])))
        test_station_mae = np.nanmean(np.absolute(station_error_data[test_indices][:, 3]))

        model_station_mean_error[experiment_title] = [(station_name_dict[station], station,
                                                        train_station_bias, train_station_rmse,
                                                        train_station_mae, test_station_bias,
                                                        test_station_rmse, test_station_mae)]

        N = len(station_error_data)
        split_length = N // n_subplots
        ind = np.arange(N)  # the x locations for the groups
        for i in range(n_subplots):
            # split indexes into slices for each subplot
            index_split = ind[i * split_length:(i + 1) * split_length]

            # for the first model we plot the background coloring
            # denoting train, test and filtered data sections
            if model_idx == 0:
                sampleTypeBackgroundColoring(axes[i], index_split,
                                             sample_type_color_mapping[i * split_length:(i + 1) * split_length])
                axes[i].set_xlim([np.min(index_split), np.max(index_split)])

            # plot network outpu
            axes[i].plot(index_split,
                         station_error_data[i * split_length:(i + 1) * split_length, 0],
                         label=experiment_title, linewidth=0.15, alpha=0.8)

    for i in range(n_subplots):
        # plot cosmo prediction
        axes[i].plot(ind[i * split_length:(i + 1) * split_length],
                     station_error_data[i * split_length:(i + 1) * split_length, 1],
                     label='COSMO-1', linewidth=0.15, alpha=0.8, color='b', linestyle='-.')

        axes[i].plot(ind[i * split_length:(i + 1) * split_length],
                     station_error_data[i * split_length:(i + 1) * split_length, 2],
                     label='Target', linewidth=0.15, alpha=0.8, color='m', linestyle='--')

        tick_step_size = np.maximum(split_length // 30, 1)
        axes[i].set_xticks(ind[i * split_length:(i + 1) * split_length][::tick_step_size])
        axes[i].set_xticklabels(time_labels[i * split_length:(i + 1) * split_length][::tick_step_size])
        axes[i].set_xticks(ind[i * split_length:(i + 1) * split_length], minor=True)
        # And a corresponding grid
        axes[i].grid(which='both')

        # Or if you want different settings for the grids:
        axes[i].grid(which='minor', alpha=0.2)
        axes[i].grid(which='major', alpha=0.5)

        handles, labels = axes[0].get_legend_handles_labels()

    # add error of cosmo prediction
    train_diff_cosmo = station_error_data[train_indices][:, 1] - station_error_data[train_indices][:, 2]
    train_station_bias_cosmo = np.mean(train_diff_cosmo)
    train_station_rmse_cosmo = np.sqrt(np.nanmean(np.square(train_diff_cosmo)))
    train_station_mae_cosmo = np.nanmean(np.absolute(train_diff_cosmo))

    test_diff_cosmo = station_error_data[test_indices][:, 1] - station_error_data[test_indices][:, 2]
    test_station_bias_cosmo = np.nanmean(test_diff_cosmo)
    test_station_rmse_cosmo = np.sqrt(np.nanmean(np.square(test_diff_cosmo)))
    test_station_mae_cosmo = np.nanmean(np.absolute(test_diff_cosmo))

    model_station_mean_error['COSMO-1'] = [(station_name_dict[station], station,
                                             train_station_bias_cosmo, train_station_rmse_cosmo,
                                             train_station_mae_cosmo, test_station_bias_cosmo,
                                             test_station_rmse_cosmo, test_station_mae_cosmo)]

    axes[n_subplots - 1].set_xlabel('Time')
    axes[0].legend(handles, labels)
    plt.tight_layout()

    run_path = source_path + '/plots/prediction_runs/%s' % run
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    fig.savefig(run_path + '/prediction_station_%s_%s' % (
    station, station_name_dict[station].replace("/", "").replace("  ", " ")) + '.png', dpi=300)

    return model_station_mean_error

def generatePredictionResultTable(output_path, results):
    with open(output_path + '/prediction_results_table_data.csv', "w") as csv_file:
        # write csv title
        csv_file.write("Experiment, Train Station Bias, Train Station RMSE, Train Station MAE, "
                       "Test Station Bias, Test Station RMSE, Test Station MAE,"
                       "Filtered Station Bias, Filtered Station RMSE, Filtered Station MAE,\n")

        for model_name, model_result in results.items():
                csv_file.write(",".join([model_name] + [str(result) for result in model_result] + ["\n"]))
            
def generateStationPredictionResultTable(output_path, results):
    with open(output_path + '/prediction_station_results_table_data.csv', "w") as csv_file:
        # write csv title
        csv_file.write("Experiment, Station Name, Station Id, Train Station Bias, Train Station RMSE, Train Station MAE, Test Station Bias, Test Station RMSE, Test Station MAE, \n")

        for model_name, model_result in results.items():
            for station_result_idx, station_result in enumerate(model_result):
                csv_file.write(",".join([model_name] + [str(result) for result in station_result] + ["\n"]))


def sampleTypeBackgroundColoring(ax, index, sample_type):
    assert len(index) == len(sample_type)
    prev_type = None
    coloring_edge_points = []
    for idx, type in enumerate(sample_type):
        if prev_type != type:
            coloring_edge_points += [(index[idx], type)]
            prev_type = type
    coloring_edge_points += [(index[-1], None)]

    last_point = coloring_edge_points[0]
    for edge_point_index in range(1, len(coloring_edge_points)):
        next_point = coloring_edge_points[edge_point_index]
        ax.axvspan(last_point[0], next_point[0], facecolor=type_color_mapping[last_point[1]], alpha=0.05)
        last_point = next_point


if __name__ == '__main__':
    experiment = sys.argv[1]
    if experiment == "network":
        plotExperimentErrorEvaluation(source_path=sys.argv[2])
    elif experiment == "baseline":
        plotBiasCorrectedBaselineResults(source_path=sys.argv[2])
    elif experiment == "plotPredictionRun":
        plotPredictionRun(source_path=sys.argv[2], observation_path=sys.argv[3])
