# WaveNet_UQ
WaveNet with Uncertainty Quantification using deep ensembles. Based on WaveNet that emulates AD99 gravity wave scheme in MiMA (Espinosa et al., 2022). PyTorch version of WaveNet has been adapted based on versions written by Minah Yang (https://github.com/yangminah) and Dave Connelly (https://github.com/dsconnelly).

# Quick start
Download files from zenodo repository and run the example scripts or notebooks.


# Using the code
The full process of generating the training data and coupling online involves working with the intermediate complexity climate model, MiMA, available here https://github.com/DataWaveProject/MiMA-machine-learning. 

## 1. Generate training data using MiMA
Follow steps here to compile and run MiMA: https://github.com/DataWaveProject/MiMA-machine-learning/tree/master

MiMA should be set up to output gravity wave drag at all timesteps for training. 
The exact version of MiMA used to generate our training data, including all input files
and diagnostic tables, can be found on my branch: https://github.com/lm2612/MiMA/ !!update!!
Note the diagnostic table `diag_table` which must include `"gwfu_cgwd"` and `"gwfv_cgwd"` for `"all"` timesteps.
This will create files named `atmos_all_1.nc`,`atmos_all_2.nc`,`atmos_all_3.nc`,...
which will be used for training, validation and offline testing.

You can skip this step and download the files `atmos_all_43.nc` and `atmos_all_44.nc` from zenodo repository.


## 2. Train ensemble of neural networks, each seeded with different random number
The script to train an individual NN can be found in `scripts/train_wavenet.py`. It takes in a range of arguments, but the important
ones are to specify the component (zonal or meridional), the transform used to preprocess data (standard or minmax), 
the filenames for training, validation and scaling, and the random seed. Changing the random seed changes the initialization and the 
converged network and is key for deep ensembles.

First, we need to choose training and validation files. We will use file `atmos_all_43.nc` for training and `atmos_all_44.nc` for testing. 

Second, it is recommended to scale all variables. We will use the standard scaler with zero mean and unit standard deviation. 
We need to estimate the mean and standard deviation across the training dataset. 
This can be done quickly using Climate Data Operators (CDO) software, rather than using python (https://code.mpimet.mpg.de/projects/cdo) e.g.,  
```
cdo -f nc4 -selname,ucomp,vcomp,gwfu_cgwd,gwfv_cgwd,temp,ps -timmean atmos_all_43.nc atmos_all_43_mean.nc
cdo -f nc4 -selname,ucomp,vcomp,gwfu_cgwd,gwfv_cgwd,temp,ps -timstd atmos_all_43.nc atmos_all_43_std.nc
```

If using standard scaling, the means and standard deviations must be saved as `{scaler_filestart}_mean.nc` and `{scaler_filestart}_std.nc` and we will provide the 
script with the `scaler_filestart` (in this case `atmos_all_43`). If training on more than one year, you may need to average over more years or change this filename.

Then, we can run the script to train wavenet. We will train for 300 epochs. 
```
cd scripts/
python -u train_wavenet.py --component "zonal" --transform "standard"  --n_epoch 300 --model_name "wavenet" --seed 1  --filename "atmos_all_43.nc" --scaler_filestart "atmos_all_43"  --valid_filename "atmos_all_44.nc"
```

This example uses the random number seed `1`. We have named the model `wavenet`. All output files will be saved under the directory `wavenet_seed1`. Output files include checkpoints at every epoch (in case the training needs to be restarted) and the neural network weights.


## 3. Test offline
Once training is completed, we carry out offline testing. We will test on `atmos_all_45.nc`.


## 4. Use FTorch to couple MiMA to WaveNet
### 4a. Compile FTorch library 
Follow instructions to set up FTorch library: https://github.com/Cambridge-ICCS/FTorch
Note, FTorch is well-maintained by ICCS and this may have been updated. I compiled FTorch in June 2023.

### 4b. Compile MiMA with FTorch library
The exact version of MiMA that includes this version of WaveNet can be found here: https://github.com/lm2612/MiMA/tree/ML-laura
```
git clone https://github.com/lm2612/MiMA
cd MiMA
git checkout ML-laura
```

This is completely based on the `ML` branch in https://github.com/DataWaveProject/MiMA-machine-learning/ but with some minor edits to
`src/atmos_param/cg_drag/cg_drag.f90`. 
Follow instructions to compile MiMA with the FTorch library.
Run MiMA.
Restart files used in this study can be found in `input/RESTART/`. These initialize the model with the state of the climate at the last 
timestep of the training dataset (`from atmos_all_44.nc`).


### 4c. Create torchscript version code 
The WaveNet code is edited slightly for compatibility. You can find that in `src/WaveNet_for_MiMA.py`

code snipped for pytorch_to_torchscript.

### 4d. Run MiMA online


## 5. Test online
code snippet here

## 6. Create

# Authors
Please reach out to me if you have any issues: lauraman@stanford.edu
Citation: 


