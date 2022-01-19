# Radiative_Response_with_Radiative_Kernel

Use the radiative kernel ([Soden et al. 2008](https://doi.org/10.1175/2007JCLI2110.1)) to diagnose the raditive response <img src="https://render.githubusercontent.com/render/math?math=dR_i">  due to different climate variables i: temperature, water vapor, albedo and cloud. 

Then we can compute the climate feedbacks: 

<img src="https://render.githubusercontent.com/render/math?math=\lambda_i = dR_i/dT">

The example code show how to compute the climate feedback of GFDL-CM4 model (using the abrupt-4xCO2 and piControl experiments). The raw data is availabel at [CMIP6](https://pcmdi.llnl.gov/CMIP6/) data nodes [LLNL](https://esgf-node.llnl.gov/projects/cmip6/). But we have to regrid the original data to the same resolution as the kernel file. The regridded data is avaliable [here](https://drive.google.com/drive/folders/1E66izDrjdOVWYl2nJj32cXSNJPegGQ8q?usp=sharing).

[Numba](https://numba.pydata.org/) is used to parallel and accelerate the computation (recommand for large dataset). 

A version of functions that use xarray and is easier to understand is also provided for understanding and modification (see the example code). 

## r3k_env.yml
The python environment file to run the example.

## Access data for the example code:
https://drive.google.com/drive/folders/1E66izDrjdOVWYl2nJj32cXSNJPegGQ8q?usp=sharing

