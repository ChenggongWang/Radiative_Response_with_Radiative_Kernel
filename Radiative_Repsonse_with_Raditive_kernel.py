import numba
from numba import njit
import numpy as np
import xarray as xr

def time_sanity_check(time_v, info):
    month_serise = time_v['time.year']*12+time_v['time.month']
    month_serise = month_serise.values
    if len(time_v.values) != (month_serise[-1] - month_serise[0]+1):
        print('Warn_L2: Data time axis is missing some part.  >>>>: Check Data files ! ')
        print('   >>>>: ' +info)
        return -1
    if np.array_equal(month_serise,np.unique(month_serise)) == False:
        print('Warn_L2: Data time axis is not unique.         >>>>: Check Data files ! ')
        print('   >>>>: ' +info)
        return -1
    return 0

@njit(parallel=True)
def alb_diff_pert_mon_cont_12mon_TLL_fast(pert_rsus,pert_rsds,cont_rsus,cont_rsds):
    """
    Compute albedo anomaly
    ALL inputs have to be numpy array
    """
    alb_diff = np.empty_like(pert_rsus,dtype = np.float32)
    for mon in numba.prange(cont_rsus.shape[0]):
        alb_cont_mon = np.empty_like(cont_rsus[0,:,:],dtype = np.float32)
        alb_cont_mon = cont_rsus[mon,:,:] / cont_rsds[mon,:,:]
        alb_cont_mon = np.where((alb_cont_mon < 1) & (np.isfinite(alb_cont_mon)), alb_cont_mon , 1 )
        for i in numba.prange(pert_rsus.shape[0]/12):
            alb_pert_mon = np.empty_like(cont_rsus[0,:,:],dtype = np.float32)
            alb_pert_mon = pert_rsus[i*12+mon,:,:] / pert_rsds[i*12+mon,:,:]
            alb_pert_mon = np.where((alb_pert_mon < 1) & (np.isfinite(alb_pert_mon)), alb_pert_mon , 1)            
            ## use 100 factor due to kernel file
            alb_diff[i*12+mon,:,:] = (alb_pert_mon  - alb_cont_mon)*100  
    return alb_diff

@njit(parallel=True)
def diff_pert_mon_cont_12mon_TLL_fast(pert_TLL,cont_TLL):
    """
    Compute anomaly between perturbation and control experiment
    For surface variable [Time, Lat, Lon] (TLL)
    ALL inputs have to be numpy array
    """
    diff = np.empty_like(pert_TLL,dtype = np.float32)
    for mon in numba.prange(cont_TLL.shape[0]):
        for i in numba.prange(pert_TLL.shape[0]/12):
            diff[i*12+mon,:,:] = pert_TLL[i*12+mon,:,:] - cont_TLL[mon,:,:]
    return diff

@njit(parallel=True)
def diff_pert_mon_cont_12mon_TPLL_fast(pert_TPLL,cont_TPLL):
    """
    Compute anomaly between perturbation and control experiment
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
    diff = np.empty_like(pert_TPLL,dtype = np.float32)
    for mon in numba.prange(cont_TPLL.shape[0]):
        for i in numba.prange(pert_TPLL.shape[0]/12):
            diff[i*12+mon,:,:,:] = pert_TPLL[i*12+mon,:,:,:] - cont_TPLL[mon,:,:,:]
    return diff


# @njit(parallel=True)
# def omega_wv_fast2(hus_pert_TPLL,hus_cont_TPLL,ta_cont_TPLL):
#     """
#     Convert humidity to omega (the variable used by water vapor kernel)
#     For 3D variable [Time, Plev, Lat, Lon] (TPLL)
#     ALL inputs have to be numpy array
#     """
#     omega_wv = np.empty_like(hus_pert_TPLL,dtype = np.float32)
#     for li in numba.prange(hus_pert_TPLL.shape[2]):
#         for lj in numba.prange(hus_pert_TPLL.shape[3]):
#             for mon in numba.prange(ta_cont_TPLL.shape[0]):
#                 dT_dlnq_mon = _dT_dlnqs_fast(ta_cont_TPLL[mon,:,li,lj])
#                 for i in range(int(hus_pert_TPLL.shape[0]/12)):
#                     omega_wv[i*12+mon,:,li,lj] =  dT_dlnq_mon \
#                                                 *(hus_pert_TPLL[i*12+mon,:,li,lj]-hus_cont_TPLL[mon,:,li,lj]) \
#                                                 /((hus_pert_TPLL[i*12+mon,:,li,lj]+hus_cont_TPLL[mon,:,li,lj])/2)
#     return omega_wv
@njit(parallel=True)
def omega_wv_fast(hus_pert_TPLL,hus_cont_TPLL,ta_cont_TPLL):
    """
    Convert humidity to omega (the variable used by water vapor kernel)
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
    omega_wv = np.empty_like(hus_pert_TPLL,dtype = np.float32)
    for mon in numba.prange(ta_cont_TPLL.shape[0]):
        for pi in range(ta_cont_TPLL.shape[1]):
            dT_dlnq_mon = _dT_dlnqs_fast(ta_cont_TPLL[mon,pi,:,:])
            for i in range(int(hus_pert_TPLL.shape[0]/12)):
                omega_wv[i*12+mon,pi,:,:] =  dT_dlnq_mon \
                                              *(np.log(hus_pert_TPLL[i*12+mon,pi,:,:])\
                                               -np.log(hus_cont_TPLL[mon,pi,:,:]))
    return omega_wv

@njit
def _dT_dlnqs_fast(T):
    # use CC-equation to derive dT/dln(q_s) 
    dTdlnqs = np.empty_like(T,dtype = np.float32)
    dTdlnqs = (T-29.65)**2/4302.64
    return dTdlnqs


@njit(parallel=True)
def RK_compute_TLL_fast(var_mon, rk_mon_cli):
    """    
    Compute radition change due to anomaly of var_mon
    For surface variable [Time, Lat, Lon] (TLL)
    ALL inputs have to be numpy array
    """
    dR_TLL = np.empty_like(var_mon,dtype = np.float32)
    for mon in numba.prange(rk_mon_cli.shape[0]):
        for i in numba.prange(var_mon.shape[0]/12):
            dR_TLL[i*12+mon,:,:] = rk_mon_cli[mon,:,:] * var_mon[i*12+mon,:,:]
    return dR_TLL

@njit(parallel=True)
def RK_compute_TPLL_plev_fast(var_mon, rk_mon_cli, plev_weight):
    """
    Compute radition change due to anomaly of var_mon
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
#     CMIP6 standart pressure level
#     plev_weight = [0.425, 0.75 , 1.125, 1.25 , 1., 1. , 1.,
#                    0.75 , 0.5  ,0.5  , 0.5  , 0.4  , 0.25 ,
#                    0.2  , 0.15 , 0.1  , 0.075, 0.045,0.005]
    
    dR_TLL = np.zeros_like(var_mon[:,0,:,:],dtype = np.float32)
    for mon in numba.prange(rk_mon_cli.shape[0]):
        for i in numba.prange(var_mon.shape[0]/12):
            for pi in range(var_mon.shape[1]):
                for li in range(var_mon.shape[2]):
                    for lj in range(var_mon.shape[3]):
                        if np.isnan(rk_mon_cli[mon,pi,li,lj]) or np.isnan(var_mon[i*12+mon,pi,li,lj]): 
                            continue
                        dR_TLL[i*12+mon,li,lj] += rk_mon_cli[mon,pi,li,lj] \
                                                 * var_mon[i*12+mon,pi,li,lj] \
                                                 * plev_weight[pi]
    return dR_TLL

def RK_plev_weight(plev):
    """
    Create plev weight
    assume the surface pressure is 1.01e5 Pa 
    """
    plev_weight = np.empty_like(plev, dtype = np.float32)
    if ((np.diff(plev) > 0).all()):
#         print('plev increase')
        plev_weight[-1] = (1.01e5-plev[-1])/1e4 +(plev[-1] - plev[-2])/2e4
        plev_weight[0] = plev[0]/1e4 + (plev[1]-plev[0])/2e4
        plev_weight[1:-1] = (plev[2:] - plev[:-2])/20000
    elif ((np.diff(plev) < 0).all()):
#         print('plev decrease')
        plev_weight[0] = (1.01e5-plev[0])/1e4+(plev[0]- plev[1])/2e4
        plev_weight[-1] = (plev[-2]-plev[-1])/2e4 + plev[-1]/1e4
        plev_weight[1:-1] = (plev[:-2] - plev[2:])/2e4
    else:
        raise Exception("Error : plev is not monotonic")
    return plev_weight
    
##  functions with xarray 
def _dT_dlnqs(T):
    # use CC-equation to derive dT/dln(q_s) 
    dTdlnqs = np.empty_like(T,dtype = np.float32)
    dTdlnqs = (T-29.65)**2/4302.64
    return dTdlnqs


def omega_wv_xarray(hus_pert_TPLL,hus_cont_TPLL,ta_cont_TPLL):
    """
    ALL inputs have to be xarray 
    """

    dT_dlnq_mon = _dT_dlnqs(ta_cont_TPLL)
    omega_wv =  dT_dlnq_mon *(np.log(hus_pert_TPLL).groupby('time.month')\
                             -np.log(hus_cont_TPLL)).groupby('time.month')
    return omega_wv


def RK_compute_suf(var, rk):
    dR = var.groupby('time.month') *rk
    return dR

def RK_compute_TPLL(var, rk):
    tmp = var.groupby('time.month')*rk
    
    # weighted by pressure and sum
#     var = ta_anom
#     plev_weight = var.plev.copy().values
#     plev_weight[0] = (1.01e5 - var.plev[1])/20000
#     plev_weight[-1] = (var.plev[-1].values-0)/20000
#     plev_weight[1:-1] = (var.plev[:-2].values - var.plev[2:].values)/20000

    # only correct for cmip6 data
    plev_weight = [0.425, 0.75 , 1.125, 1.25 , 1., 1. , 1., \
                   0.75 , 0.5  ,0.5  , 0.5  , 0.4  , 0.25 ,\
                   0.2  , 0.15 , 0.1  , 0.075, 0.045,0.005]
    plev_weight = xr.DataArray(plev_weight,var.plev.coords,dims=['plev'])
    dR = tmp *plev_weight
    dR = dR.sum(dim='plev',skipna=True)
    return dR