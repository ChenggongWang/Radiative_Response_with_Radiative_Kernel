# Radiative_Response_with_Radiative_Kernel

Use the radiative kernel ([Soden et al. 2008](https://doi.org/10.1175/2007JCLI2110.1)) to diagnose the raditive response <img src="https://render.githubusercontent.com/render/math?math=dR_i">  due to different climate variables i: temperature, water vapor, albedo and cloud. 

Then we can compute the climate feedbacks: 

<img src="https://render.githubusercontent.com/render/math?math=\lambda_i = dR_i/dT">

[Numba](https://numba.pydata.org/) is used to parallel and accelerate the computation. 


## r3k_env.yml
The python environment file to run the example.

## Access data for the example code:
https://drive.google.com/drive/folders/1E66izDrjdOVWYl2nJj32cXSNJPegGQ8q?usp=sharing

