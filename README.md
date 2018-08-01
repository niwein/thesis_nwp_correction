# Documentation
A short documentation of dependencies, data and structure of the project code for the master thesis "Deep Learning based Error Correction of Numerical Weather Prediction in Switzerland". Additional documentation can be found directly in the code, for specific files, classes or lines.

## Dependencies
The complete code is writetn in Python 3 (v3.5.2). Necessary python packages are:

 - basemap 1.1.0
 - matplotlib 2.2.2  
 - netCDF4 1.3.1
 - numpy  1.14.5
 - pandas 0.22.0
 - pytorch 0.4.0 
 - scikit-learn 0.19.0
 - scipy 1.1.0
 - tensorboard 1.8.0
 - tensorboardX 1.1
 - torch 0.4.0
 - xarray 0.10.2

For the model interpretation additional external libraries are:

 - [SHAP framework](https://github.com/slundberg/shap) 
 - [Influence release](https://github.com/kohpangwei/influence-release)

Even no specific code is provided, the theoretical basis for the uncertainty estimation can be found in this [paper](https://arxiv.org/pdf/1506.02142.pdf) from 2016 by Gal et al.

## Data
Necessary data files for the project are:

 - COSMO-1 outputs
 -  Topographical grid data
 - Observations at the stations

These files are provided by MeteoSwiss (non public) and come as netCDF files (.nc)

To open the files we can recommend "xarray" as a python module or an open-source tool with GUI support called [Panoply](https://www.giss.nasa.gov/tools/panoply/download/) (NASA).

#### COSMO-1 Files
These files contain the time-dependent feature prediction of COSMO-1.
The COSMO-1 files are in folders named by the initialization (year, month, day, hour) like "YYMMDDHH_10X", the last number (X) has no specific meaning to the best of our knowledge. A file itself has a name like "c1ffsurf016.nc", where "16" here stands for the COSMO-1 output predicting the lead time T=16 (16h after the initialization time).
#### Topographical Grid Data File
This file contains topological information for each grid point. The files are either named "c1ffsurf000_timeinvariant.nc" or "c1ffsurf000_timeinvariant_lonlat.nc", depending whether also longitude and latitude information is included.

#### Observations (Station) File
The station observations are in a file named like "meteoswiss_t2m_20151001-20180331.nc". It contains specific information to the station and the temperature observation between the dates in the name.

## Structure
The most important file is the "Runner" script, initiating all executions. The "ModelRun" and "ModelRunError" files contain the training/testing procedure of neural networks. "ModelRunError" is hereby the special case, when the model is trained to predict the COSMO-1 error instead of the direct 2m-temperature.
### Folders
 - **model**: Contains the definition of the nerual networks and the baseline calculation.
 - **preprocessing**: Contains all python files to preprocess the data.
 - **utils**: Contains python files providing utility funtions used in preprocessing, model runs, and plotting.
 - **results**:  Contains files for prediction runs of the baseline and trained models. Additionally, all experiment definitions are put in the "runs" subfolder.
 - **scripts**: Contains jupyter notebook scripts used to generate results in the thesis, i.e. baseline errors, model results, feature interpretation, data valuation and uncertainty estimation.

## Execution
Step-by-step explanation to run different processes, such as preprocessing, model training, model prediction runs and model interpretation.

### Preprocessing
The raw COSMO-1 data has firstly to be preprocessed seperately for the neural network and the baseline runs.

The following parameters have to be adapted, since they will determine the preprocessin:

 - withTopo: Determins if topographical features should be preprocessed
 - ListParam: Lists COSMO-1 parameters to be preprocessed
 - TopoListParam: Lists topographical grid parameters to be preprocessed
 - G: Determines the grid size (patch) around the station to be preprocessed
 - DB: First initialization of data to be preprocessed
 - DE: Last initialization of data to be preprocessed
 - T: Lead times (future prediction) to be preprocessed
 - n_parallel: Number of parallel workers for preprocessing
 - observation_path: Path to station observation file

Important note: For n_parallel > 1, each worker should receive its own observation and topographical grid data file, since we have encountered errors if netCDF (.nc) files are opened simultaneously by multiple workers.
  
#### Preprocessing of Data for Neural Network
The preprocessing can be initiated with the following command:

    python3 Runner.py --script "createStationGrids" --preprocessing "station"
##### Run parameters:

 - **--script**: "createStationGrids" determines the preprocessing of data for the training of neural networks
 - **--preprocessing**: Either "station" or "station_init" can be chosen, to determin how the preprocessed data format should be. For "station" a separate file per station is created and for "station_init" the preprocessed files will have the form "(station_x, init_i)".

Important note: The paths to the data in the preprocessing files ("CreateDataByStation" & "CreateDataByStationAndInit") have to be adapted.
#### Preprocessing of Data for Baseline
The preprocessing can be initiated with the following command:

    python3 Runner.py --script "createBaselineData"
##### Run parameters:

 - **--script**: "createBaselineData" determines the preprocessing of data for the bias-corrected baselines.

Important note: The paths to the data in the preprocessing file ("CreateBaselineData") have to be adapted.

#### Calculation of Feature Statistic (for nerual network only)
Time dependent grid data (COSMO-1 output prediction) is normalized in the pytorch DataLoader when a neural network is run. Therefore, necessary feature statistics (min, max, mean, var) have to be calculated on the preprocessed data in a separate run.
The calculation of the feature statistic can be initiated with the following command:

    python3 Runner.py --script "generateFeatureValues"
##### Run parameters:

 - **--script**: "generateFeatureValues" determines the calculation of the feature statistic.
Important note: The parent folder containing all preprocessed data must be changed in "GenerateFeatureValues". The grid size G will determine, which preprocessed data will be used for the calculation of the feature statistic.

### Baseline Run
To run the baseline, preprocessed baseline data has to be available.
The baseline run can be initiated with the following command:

    python3 Runner.py --script "runBaseline" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA"
##### Run parameters:
 - **--script**: "runBaseline" determines the run of the baseline
 - **--preprocessing**: Type of preprocessing ("station"/"station_init")
 - **--experiment-path**: Path to experiment folder
 - **--input-source**: Parent folder of all preprocessed data

Important note: The experiment folder needs to contain an "experiment_parameters.txt" file. An example can be found in "results/runs/2d_basline".

### Neural Network Run (Training)
To train a neural network to directly predict the 2m-temperature, preprocessed neural network data has to be available.
The neural network training can be initiated with the following command:

    python3 Runner.py --script "runModel" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA" --model-type "network"
##### Run parameters:
 - **--script**: "runBaseline" determines the run of the baseline
 - **--preprocessing**: Type of preprocessing ("station"/"station_init")
 - **--experiment-path**: Path to experiment folder
 - **--input-source**: Parent folder of all preprocessed data
 - **--model-type**: Either "network" to predict the 2m-temperature or "network_error_prediction" to predict the COSMO-1 error.

For the script of uncertainty estimation, the nerural network has to be trained using dropout. To use dropout in training of the network the following property has to be added to the model configuration: "dropout_prop" : p, where p is between 0 and 1.

Important note: The experiment folder needs to contain an "experiment_parameters.txt" file and >=1 model definition file in a subfolder called "models". Examples can be found in "results/runs/neural_network".

### Prediction Runs
For the baselines and trained neural networks, so called "prediction runs" can be executed. Thereby, a prediction is generated for each data points (training and test points). The prediction run generates a error statistic on each data points, what is used in scripts of model error evaluation and also of model interpretation.

#### Baseline Prediction Run
The neural network run can be initiated with the following command:

    python3 Runner.py --script "runModel" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA" --model-type "predictionRun"

All parameters stay the same as for the baseline run, except an additional flag has to be added, i.e. --model-type "predictionRun"
The experiment folder should be the same for an already executed baseline run.

#### Neural Network Prediction Run
For running a neural network prediction, a previously trained network must exist in the experiment path.
The baseline prediction run can be initiated with the following command:

    python3 Runner.py --script "runBaseline" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA" --model-type "predictionRun"

All parameters stay the same as for the baseline run, except an additional flag has to be added, i.e. --model-type "predictionRun" or --model-type "network_error_prediction_run", depending on the label of the trained neural network to be used.

The experiment folder should be the same for an already executed neural network training run.

### Spatial Generalization Experiment
A station cross-validation experiment, where stations are left out for training and are only for testing we provide can be initiated with the following command:

    python3 Runner.py --script "spatialGeneralizationExperiment" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA"

The set-up is the same as for training a neural network. Example experiment folders can be found in /results/runs/spatial_generalization.
The only important change to make, is adapting the "experiment_parameters.txt" file with the following two properties:

 - n_test_stations: Number of stations to be left out from training.
 - first_test_station: First station idx in [0,...,143] to be left out, together with the next "n_test_stations" stations.

Important note: This is only supported for models predicting the 2m-temperature.

### Generate Network Ready Data
For the scripts of model interpretation, especially for feature interpretation, data valuation with influence functions and uncertainty estimation, we need data that can be directly fed into the neural network. Therefore, we provide a method, that simulates a network training run, but instead of feeding the data into the network, the data is directly stored in the desired form.
The generation of network ready data can be initialized with the following command:

    python3 Runner.py --script "runModel" --preprocessing "station" --experiment-path "PATH_TO_EXPERIMENT_FOLDER" --input-source "PATH_TO_PREPROCESSED_DATA" --model-type "generateNetworkReadyTestData"

The experiment folder should be the same for an already executed neural network training run. The network ready data will be generated, such that it would be compliant to the model and experiment configuration in the experiment folder. The final network ready data will be placed in the experiment folder in a separate file for training and test data.

## Jupyter Notebook Scripts
To produce the results and plots for our thesis, we used jupyter notebook scripts. All scripts can be found in the "scripts" folder. The scripts can be differentiated between baseline results, model results and model interpretation.

#### Requirements
The scripts are dependent on previously calculated data, e.g. preprocessed data, model/baseline errors, model/baseline prediction errors and network ready data. We suggest to first run all previous steps, before the scripts can be used.

### Baseline Scripts
Scripts to run a KNN regression baseline and to calculate and plot bias-corrected baseline results.
 - **KNN Regression Baseline**: Used to calculate a KNN baseline.
 - **Plot Baseline Results**: Plot the results from the bias corrected baselines. The script also contains calculations of the forecast skill, whereby also error data from network prediction runs are required.

### Model Scripts
Scripts for the results used in the model results section.
#### Basic Results
Script to calculate and plots basic model results.

 - **Plot Network Result**: Calculate and plot errors of the nerual network models. Therefore, for the models also prediction run is necessary. Additionally, we also use the baseline errors as comparison.

#### Lead Time Prediction
Script to generate the experiment definitions to carry out a lead time prediction experiment, by training a neural network on the desired lead times and a scripts that allows to analyze and plot the results.

 - **Generate Lead Time Experiment**: Convenience script to create runnable experiment folders for desired lead times.  The model used can be specified in parent "experiment_definition" folder and is copied to all folders.
 - **Plot Lead Time Prediction Results**: Script to analyze and plot the results of a lead time prediction experiment.

#### Spatial Generalization
Scripts for the station cross-validation experiment and for predictions of a trained neural network on the complete grid.
##### Station Cross-Validation
Script to set up the experiment and a script to analyze and plot the results.

 - **Generate K-Fold Station Spatial Generalization Experiment**: Convenience script to create runnable experiment folders for desired splits of stations. The model used can be specified in parent "experiment_definition" folder and is copied to all folders.
 - **K-Fold Cross Validation Results**: Script to analyze an plot the results for a station cross-validation experiment.

##### Prediction on complete Grid
Script to make a prediction on a the complete grid using a raw COSMO-1 output.

 - **CreateDataForCompleteGrid**: Script takes a raw COSMO-1 output and pre-processes it. Then a trained model can be loaded to make predictions for each preprocessed grid point.
### Feature Interpretation
Script to calculate the feature importance of a trained neural network for randomly selected test data points and specific splits between summer/winter and 04:00/16:00 with SHAP framework.
 - **Feature Interpretation with SHAP**: With the script the feature importance can be calculated using the SHAP framework. The plotting utilities many come direct from the framework.

For the feature interpretation the code for the SHAP framework is necessary. Additionally, it needs a trained neural network on which the feature importance is calculated. To calculate the feature importance network ready data is necessary, since the framework uses the trained model directly by an interface.

### Data Valuation
Scripts for calculating neural embeddings for the data valuation approaches and scripts for the valuation itself for the two methods used in the thesis, i.e. data value by influence and Shapley value calculation with KNN classifier.

 - **CreateNeuralEmbeddings**: Script to calculate neural embedding of a neural network that has learned to predict the 2m-temperature.
 - **CreateNeuralEmbeddingsErrorPrediction**: Script to calculate neural embedding of a neural network that has learned to predict the COSMO-1 error.
 - **Shapley Values with Influence Functions**: Script to calculate the data value of training points with influence functions. Additionally, it provides methods to plot the results.
 - **Calculate KNN Shapley Values**: Script to calculate the Shapley values of training points based on a KNN classifier.  Additionally, it provides methods to plot the results.

For both scripts calculating the value of data points, first the neural embeddings have to be generated.

### Uncertainty Estimation
To run the script for uncertainty estimation a neural network trained with dropout is necessary.

 - **Uncertainty Estimation**: Script to predict data points with dropout applied also in the prediction. The mean prediction is formed from multiple prediction runs together with the variance. The variance is used as an estimation of model uncertainty.

For this script network ready data is required for generating the multiple prediction runs with active dropout.

