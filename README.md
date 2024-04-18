# Radiative_Response_with_Radiative_Kernel

Use the radiative kernel ([Soden et al. 2008](https://doi.org/10.1175/2007JCLI2110.1)) to diagnose the TOA radiative response $dR_i$  due to change in  climate variables i: temperature (ta and ts), water vapor (wv), albedo and cloud. 

Then we can compute the climate feedbacks: 

$$\lambda_i = \frac{dR_i}{dts_{gm}}$$

where ${dts_{gm}}$ is the global mean surface temp. change.

# Usage
### Create the Python environment.
use `r3k_env.yml` file
```
conda env create -f r3k_env.yml
conda activate r3k_env
python -m ipykernel install --user --name r3k_env --display-name "r3k_env" # Register the kernel for jupyter
```
or 
```
conda create -y -n r3k_env python=3.10
conda activate r3k_env
conda install  -y  -c conda-forge numpy numba scikit-learn matplotlib
conda install  -y  -c conda-forge xarray netCDF4 ipykernel 
python -m ipykernel install --user --name r3k_env --display-name "r3k_env" # Register the kernel for jupyter
```

### Download the kernel file and the example data (4G)
```
wget https://tigress-web.princeton.edu/~cw55/share_data/r3k_example_data.tar
wget https://raw.githubusercontent.com/ChenggongWang/Radiative_Response_with_Radiative_Kernel/main/Radiative_Repsonse_with_Raditive_kernel.py -O Radiative_Repsonse_with_Raditive_kernel.py
```
### Run example notebook (requires the example data below).

# Example 

The example notebook ([r3k_example.ipynb](https://github.com/ChenggongWang/Radiative_Response_with_Radiative_Kernel/blob/main/R3k_example.ipynb)) shows how to compute the climate feedback of GFDL-CM4 model (using the abrupt-4xCO2 and piControl experiments). The raw data is available at [CMIP6](https://pcmdi.llnl.gov/CMIP6/) data nodes [LLNL](https://esgf-node.llnl.gov/projects/cmip6/). But we have to regrid the original data to the same resolution as the kernel file (or the other way around). The regridded data can be downloaded from [google drive](https://drive.google.com/drive/folders/1E66izDrjdOVWYl2nJj32cXSNJPegGQ8q?usp=sharing) or [princeton.edu compressed file](https://tigress-web.princeton.edu/~cw55/share_data/r3k_example_data.tar).


# Decription
`decompose_dR_rk_toa_core(var_pert, var_cont,f_RK, forced=True)` is the core function to call. The option `forced=True` will estimate the all-sky forcing using clear-sky forcing. Set `forced=False` for experiments without TOA radiative forcing (e.g. fixed SST experiment).
It will return a xarray dataset that contains variables as follows (also its global-mean, variable name with `_gm`):

>`dR_wv_lw  ` : $\frac{\partial R_{all-sky\ lw}}{\partial wv}\Delta wv ,\qquad lw\ R_{toa}$ change due to `water vapor(wv)` change
>
>`dR_wv_sw  ` : $\frac{\partial R_{all-sky\ sw}}{\partial wv}\Delta wv    $
>
>`dR_wvcs_lw` : $\frac{\partial R_{clr-sky\ lw}}{\partial wv}\Delta wv    $
>
>`dR_wvcs_sw` : $\frac{\partial R_{clr-sky\ sw}}{\partial wv}\Delta wv    $
>
>`dR_ta     ` : $\frac{\partial R_{all-sky\ lw}}{\partial ta}\Delta ta ,\qquad lw\ R_{toa}$ change due to `air temp.(ta)` change (including lapse rate change)
>
>`dR_tacs   ` : $\frac{\partial R_{clr-sky\ lw}}{\partial ta}\Delta ta    $
>
>`dR_lr     ` : $\frac{\partial R_{all-sky\ lw}}{\partial lr}\Delta lr ,\qquad lw\ R_{toa}$ change due to `lapse rate(lr)` change (vertial structure differs from dts)
>
>`dR_lrcs   ` : $\frac{\partial R_{clr-sky\ lw}}{\partial lr}\Delta lr    $
>
>`dR_ts     ` : $\frac{\partial R_{all-sky\ lw}}{\partial ts}\Delta ts ,\qquad lw\ R_{toa}$ change due to `surface temp.(ts)` change (including lapse rate change)
>
>`dR_tscs   ` : $\frac{\partial R_{clr-sky\ lw}}{\partial ts}\Delta ts    $
>
>`dR_alb    ` : $\frac{\partial R_{all-sky\ sw}}{\partial albedo}\Delta albedo  ,\qquad sw\ R_{toa}$ change due to `surface albedo(alb)` change 
>
>`dR_albcs  ` : $\frac{\partial R_{clr-sky\ sw}}{\partial albedo}\Delta albedo$
>
>`dR_cloud_lw   ` : $\frac{\partial R_{all-sky\ sw}}{\partial cloud}\Delta cloud ,\qquad lw\ R_{toa}$ change due to `cloud` change (computed as residual)
>
>`Dcs_lw    ` : $dF_{clr-sky\ lw}$ estimated `clear-sky lw forcing`
>
>`Dcs_sw    ` : $dF_{clr-sky\ sw}$ estimated `clear-sky sw forcing`
>
>`dR_sw     ` : $dR_{all-sky\ sw}$ all-sky TOA sw net flux change
>
>`dR_lw     ` : $dR_{all-sky\ lw}$ all-sky TOA lw net flux change
>
>`dRcs_sw   ` : $dR_{clr-sky\ sw}$ clear-sky TOA sw net flux change
>
>`dRcs_lw   ` : $dR_{clr-sky\ lw}$ clear-sky TOA lw net flux change
>
>`ts        ` : $ts$ surface temperature 
>
>`dts       ` : $dts$ 'surface temperature change'

Decomposition base on RH:
>`dR_rh_lw  ` : $\frac{\partial R_{all-sky\ lw}}{\partial rh}\Delta rh ,\qquad lw\ R_{toa}$ change due to `relative humidity(rh)` change
>
>`dR_rh_sw  ` : $\frac{\partial R_{all-sky\ sw}}{\partial rh}\Delta rh    $
>
>`dR_rhcs_lw` : $\frac{\partial R_{clr-sky\ lw}}{\partial rh}\Delta rh    $
>
>`dR_rhcs_sw` : $\frac{\partial R_{clr-sky\ sw}}{\partial rh}\Delta rh    $
>
>`dR_Ta_rh_lw  ` : $\frac{\partial R_{all-sky\ lw}}{\partial ta}\mid_{RH} \Delta ta ,\qquad lw\ R_{toa}$ change due to `air temp.(ta)` change (including lapse rate change) when `relative humidity(rh)` is fixed
>
>`dR_Ta_rh_sw  ` : $\frac{\partial R_{all-sky\ sw}}{\partial ta}\mid_{RH} \Delta ta    $
>
>`dR_Tacs_rh_lw` : $\frac{\partial R_{clr-sky\ lw}}{\partial ta}\mid_{RH} \Delta ta    $
>
>`dR_Tacs_rh_sw` : $\frac{\partial R_{clr-sky\ sw}}{\partial ta}\mid_{RH} \Delta ta    $

# Note

Everything is common (`xarray` to load/create `netcdf` data, `numpy`, `matplotlib` to show results) except `Numba`.

[`Numba`](https://numba.pydata.org/) is the core package and is used to parallel/accelerate the computation (recommended for large datasets or many datasets/experiments). 

>For jobs that are __not time sensitive__, you should be able to use __numpy or xarray version__ of functions without Numba. 

A version of functions that use only numpy/xarray and is easier to understand is also provided for understanding and modification (see the benchmark code).

>The time for 150 years 2x2.5 [latxlon] data (\~4GB) is ~ 10 seconds on princeton jupyterhub (expect similar time for CPUs>=4).
>
>Numpy version takes ~ 30 seconds (single CPU).
>
>Xarray version takes 1\~2 mins (single CPU).

# To do plan
Add regrid option for convenience
