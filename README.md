
[![DOI](https://zenodo.org/badge/648818482.svg)](https://zenodo.org/doi/10.5281/zenodo.11200997)


# Uncertainty Quantification of a Machine Learning Subgrid-Scale Parameterization for Atmospheric Gravity Waves

Code accompanying paper [Mansfield & Sheshadri (2024)](https://doi.org/10.1029/2024MS004292), where we carry out uncertainty quantification of a machine learning subgrd-scale parameterization of atmospheric gravity waves (WaveNet) using deep ensembles. Based on WaveNet that emulates AD99 gravity wave scheme in MiMA ([Espinosa et al., 2022](https://doi.org/10.1029/2022GL098174). PyTorch version of WaveNet has been adapted based on versions written by Minah Yang (https://github.com/yangminah) and Dave Connelly (https://github.com/dsconnelly).

Data is available on the [Stanford Digital Repository](https://doi.org/10.25740/zv875tm6846)

![predictedGWD](./GWD_profile_at_equator.gif)

## Neural network structure
The neural network takes inputs at each grid lon/lat grid cell, some of which are defined as height profiles with dimension=40.

INPUTS (total dim=82):
* wind profile (zonal or meridional). dim=40
* temperature profile. dim=40
* latitude. dim=1
* surface pressure. dim=1

OUTPUTS (total dim=40):
* gravity wave drag profile (zonal or meridional) dim=40

Note, the inputs and outputs are either both zonal or both meridional. See Espinosa et al., 2022 for full details on the neural network.

## Dependencies
* Model of an idealized Moist Atmosphere: https://github.com/DataWaveProject/MiMA-machine-learning.
* FTorch coupling library: https://github.com/Cambridge-ICCS/FTorch.
* For training, testing, plotting, analysis in python: torch, xarray, matplotlib.
GPU resources are not absolutely necessary for training, but are faster. For GPU training, you also need cuda. 

# Source code
`src` containes the most important files for setting up the dataset and defining the neural network architecture.
* `src/GravityWavesDataset.py` defines a custom torch dataset that opens the training/validation/test file(s), concatenates these if necessary, and sets up all transforms if necessary. We use standard scaling transforms. When indexed, the dataset returns the inputs and outputs for a single grid cell and timestep.
* `src/Wavenet.py` defines the neural network architecture for training, validation and testing in python. It assumes inputs and outputs are already scaled.
* `src/WaveNet_for_MiMA.py` defines the neural network architecture for coupling into the climate model, MiMA. It uses the same architecture as Wavenet.py but has been adapted for interoperability with the Fortran-based climate model using FTorch. It assumes inputs and outputs are not scaled (as the climate model will provide raw variables)

`scripts` contain python scripts for training, testing and converting to torchscript. You can edit argument inputs to repeat training for both zonal and meridional components, different random seed initializations for generating enembles, and also for trying different transforms, learning rates, dropout, and so on.
* `scripts/train_wavenet.py` contains the training loop. 
* `scripts/test_wavenet.py` contains a loop to test over new, unseen data.
* `scipts/wavenet_to_torchscript.py` converts the pytorch model into torchscript for use with FTorch and MiMA.

Scripts for analyzing statistics and plotting can be found in `analysis`.

# Quick start
Download files from SDR (link+doi to be added) repository and run the example notebook in `examples`. This provides a quick demo for how to open the data, load the model and start training.

# Using the code
This describes the full process to reproduce results of the paper. Generating the training data and coupling online involves working with the intermediate complexity climate model, MiMA, available here https://github.com/DataWaveProject/MiMA-machine-learning. 

## 1. Generate training data using MiMA
Follow steps here to compile and run MiMA: https://github.com/DataWaveProject/MiMA-machine-learning/tree/master

MiMA should be set up to output gravity wave drag at all timesteps for training. 
The exact version of MiMA used to generate our training data, including all input files
and diagnostic tables, can be found on my branch: https://github.com/lm2612/MiMA/ !!update!!
Note the diagnostic table `diag_table` which must include `"gwfu_cgwd"` and `"gwfv_cgwd"` for `"all"` timesteps.
This will create files named `atmos_all_1.nc`,`atmos_all_2.nc`,`atmos_all_3.nc`,...
which will be used for training, validation and offline testing.

You can skip this step and download the data from SDR repository.


## 2. Train ensemble of neural networks, each seeded with different random number
The script to train an individual NN can be found in `scripts/train_wavenet.py`. It takes in a range of arguments, but the important
ones are to specify the component (zonal or meridional), the transform used to preprocess data (standard or minmax), 
the filenames for training, validation and scaling, and the random seed. Changing the random seed changes the initialization and the 
converged network and is key for deep ensembles.

First, we need to choose training and validation files. We will use file `atmos_all_43.nc` for training and `atmos_all_44.nc` for validation. 

Second, it is recommended to scale all variables. We will use the standard scaler with zero mean and unit standard deviation. 
We need to estimate the mean and standard deviation across the training dataset. 
This can be done quickly using Climate Data Operators (CDO) software (https://code.mpimet.mpg.de/projects/cdo) e.g.,  
```
cdo -f nc4 -selname,ucomp,vcomp,gwfu_cgwd,gwfv_cgwd,temp,ps -timmean atmos_all_43.nc atmos_all_43_mean.nc
cdo -f nc4 -selname,ucomp,vcomp,gwfu_cgwd,gwfv_cgwd,temp,ps -timstd atmos_all_43.nc atmos_all_43_std.nc
```
or you can calculate these in python/xarray. 
If using standard scaling, the means and standard deviations must be saved as `{scaler_filestart}_mean.nc` and `{scaler_filestart}_std.nc` and we will provide the 
script with the `scaler_filestart=atmos_all_43`. If training on more than one year, you may need to average over more years or change this filename.

Then, we can run the script to train wavenet. We will train for 300 epochs. 
```
cd scripts/
python train_wavenet.py --component "zonal" --transform "standard"  --n_epoch 300 --data_dir "path/to/data" --output_dir "path/to/output" --model_name "wavenet_zonal_seed1" --seed 1  --filename "atmos_all_43.nc" --scaler_filestart "atmos_all_43"  --valid_filename "atmos_all_44.nc"
```
`data_dir` is the location of your training, validation and scaler files. All output files will be saved under a new directory `path/to/output/wavenet_zonal_seed1/`. Make sure to save different seeded models under different `model_name`. Output files include checkpoints at every epoch (in case the training needs to be restarted) as `checkpoint_epoch{epoch}.pth` and the neural network weights as `wavenet_weights_epoch{epoch}.pth`.

To train the ensemble, repeat this many times with different random seeds.

## 3. Analyze offline results
Once training is completed, we carry out offline testing. For this example we are testing on daily datasets rather than 3-hourly as these files are smaller (you can find them on the SDR repository). This example tests on `atmos_daily_45.nc`. The script is in `scripts/test_wavenet.py` and uses almost identical arguments to the training script above.
```
python test_wavenet.py --component "zonal" --transform "standard"  --data_dir "path/to/data" --output_dir "path/to/output"  --model_name "wavenet_zonal_seed1" --filename "atmos_daily_45.nc" --scaler_filestart "atmos_all_43"
```
Notice we still provide the scaling file from the training dataset. Also notice we provide the full model output directory `wavenet_zonal_seed1`. This script outputs predictions for zonal gravity wave drag saved as `atmos_daily_45.nc` in the same directory. 

This should be repeated for all ensemble members and for both `zonal` and `meridional` components. Then, profiles of gravity wave drag can be plotted using:
```
python analysis/plot_GWD_profiles.py 
```
This script plots profiles for individual grid cells for each lon, lat, and timestep. You can subsample every n-th lon/lat/timestep if you do not want to create hundreds of plots. Note, you will need to edit filenames and directories.


## 4. Generate online coupled simulations
For online testing, we need to use FTorch to couple MiMA to WaveNet
### 4a. Compile FTorch library 
Follow instructions to set up FTorch library: https://github.com/Cambridge-ICCS/FTorch. Note, FTorch is well-maintained by ICCS and this may have been updated (for these results, FTorch was compiled in June 2023).

### 4b. Compile MiMA with FTorch library
The exact version of MiMA that includes this version of WaveNet can be found here: https://github.com/lm2612/MiMA/tree/ML-laura.
```
git clone https://github.com/lm2612/MiMA
cd MiMA
git checkout ML-laura
```

This is completely based on the `ML` branch in https://github.com/DataWaveProject/MiMA-machine-learning/ but with some minor edits to
`src/atmos_param/cg_drag/cg_drag.f90`. 
Follow instructions to compile MiMA with the FTorch library.

### 4c. Create torchscript version code 
The WaveNet code is edited slightly for compatibility. You can find that in `src/WaveNet_for_MiMA.py`. The key differences are that we carry out scaling at evaluation time, as the climate model passes raw variables to the machine learning model.
We export the pytorch model to torchscript using
```
python wavenet_to_torchscript.py --component "zonal" --transform "standard"   --model_name "wavenet_zonal_seed1" --filename "atmos_all_44.nc" --scaler_filestart "atmos_all_43"
```
The file `atmos_all_44.nc` is used to test the torchscript model. This script checks the validation losses across all epochs and selects the epoch that produces the lowest validation loss. Then, that model is converted into torchscripting, suitable for the FTorch library. The torchscript model is saved as `zonal_weights.pth` or `meridional_weights.pth`

### 4d. Run MiMA with NNs online
Now, we can run MiMA with the NNs. We need to copy our torchscript models for both zonal and meridional components into the directory where we will run MiMA. You can find my scripts for setting up the directories and running MiMA in `mima_scripts/`, however, this will be dependent on architecture and directory names.
We run one 20-year MiMA simulation for each NN ensemble member. Each of these should be initialized from the same point, such as the end of the validation dataset. Be sure to output winds (`ucomp` and `vcomp`) and gravity wave drag (`gwfu_cgwd` and `gwfv_cgwd`) in the `diag_table` file. If you checkout my branch from step 4b, you will find these diagnostics in `input/diag_table` and the relevant restart files in `input/RESTART`. I save output daily to files named `atmos_daily_45.nc`, `atmos_daily_46.nc`, ... `atmos_daily_65.nc`. You can also find these files for all ensemble members in the SDR repository. With a 30 member ensemble and 20 years of simulation data for each member, we can examine statistics of the circulation.

## 5. Analyze online results
We cannot compare online simulations to the AD99 simulations directly, because the simulations diverge with time. But we can plot gravity wave drag and wind distributions using the original AD99 simulation, the offline results from step 3 and the online results from step 4.
```
python analysis/plot_distributions.py
```

In the `analysis` directory, you can also find code for saving, plotting and analyzing QBO and polar vortex winds. This directory contains all scripts used to generate plots in the paper. Note that you will need to edit directory names and possibly also filenames.

# Authors
Laura A. Mansfield 

Please reach out to me if you have any issues: lauraman@stanford.edu

# Citations

### Paper: 
Mansfield, L. A., & Sheshadri, A. (2024). Uncertainty quantification of a machine learning subgrid-scale parameterization for atmospheric gravity waves. Journal of Advances in Modeling Earth Systems, 16, e2024MS004292. [https://doi.org/10.1029/2024MS004292](https://doi.org/10.1029/2024MS004292)

### Code: 
Mansfield, L. A. (2024). lm2612/WaveNet_UQ: WaveNet_UQv0.1.0 (v0.1.0). Zenodo. [https://doi.org/10.5281/zenodo.11200998](https://doi.org/10.5281/zenodo.11200998).

### Data: 
Mansfield, L. A. and Sheshadri, A. (2024). Data for Uncertainty Quantification of a Machine Learning Subgrid-Scale Parameterization for Atmospheric Gravity Waves. Stanford Digital Repository. Available at [https://purl.stanford.edu/zv875tm6846](https://purl.stanford.edu/zv875tm6846). [https://doi.org/10.25740/zv875tm6846](https://doi.org/10.25740/zv875tm6846).





