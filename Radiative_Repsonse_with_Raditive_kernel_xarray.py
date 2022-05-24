import numpy as np
import xarray as xr

def global_mean_xarray(ds_XXLL):
    """ 
    Compute the global mean value of the data.
    The data has to have the lat and lon in its dimensions.
    Should not include NaN in Inputs.
    
    Parameters
    ----------
    ds_XXLL   : xarray with lat and lon. ds_XXLL.lat will be 
                used for area weight.

    Returns
    ----------
    tmp_XX    : xarray without lat and lon.
    
    """
    lat = ds_XXLL.coords['lat']        # readin lat
    # global mean
    # compute cos(lat) as a weight function
    weight_lat = np.cos(np.deg2rad(lat))/np.mean(np.cos(np.deg2rad(lat)))
    tmp_XXL = ds_XXLL.mean(dim=['lon'])*weight_lat
    tmp_XX  = tmp_XXL.mean(dim=['lat'])
    return tmp_XX

def decompose_dR_rk_toa_core_xarray(var_pert, var_cont,f_RK ):

    check_dimensions(var_pert, var_cont,f_RK)
    ta_anom = var_pert['ta'].groupby('time.month') - var_cont['ta']
    ts_anom = var_pert['ts'].groupby('time.month') - var_cont['ts']
    ta_anom_lr = ta_anom - ts_anom
    omega_wv = omega_wv_xarray (var_pert['hus'], var_cont['hus'], var_cont['ta'])
    alb_pert = var_pert['rsus']/var_pert['rsds']
    alb_pert = alb_pert.where(np.isfinite(alb_pert),0)
    alb_cont = var_cont['rsus']/var_cont['rsds']
    alb_cont = alb_cont.where(np.isfinite(alb_cont),0)
    alb_anom = alb_pert.groupby('time.month') - alb_cont
    dR_lw    = (- var_pert['rlut'] ).groupby('time.month') \
              -(- var_cont['rlut'] )
    dR_sw    = (var_pert['rsdt'] - var_pert['rsut'] ).groupby('time.month') \
              -(var_cont['rsdt'] - var_cont['rsut'] )
    dRcs_lw  = (- var_pert['rlutcs'] ).groupby('time.month') \
              -(- var_cont['rlutcs'] )
    dRcs_sw  = (var_pert['rsdt'] - var_pert['rsutcs'] ).groupby('time.month') \
              -(var_cont['rsdt'] - var_cont['rsutcs'] )
    
    plev_weight = RK_plev_weight(f_RK.plev)
    dR_wv_lw    = RK_compute_TPLL(omega_wv   ,f_RK.lw_q     , plev_weight)
    dR_wv_sw    = RK_compute_TPLL(omega_wv   ,f_RK.sw_q     , plev_weight)
    dR_wvcs_lw  = RK_compute_TPLL(omega_wv   ,f_RK.lwclr_q  , plev_weight)
    dR_wvcs_sw  = RK_compute_TPLL(omega_wv   ,f_RK.swclr_q  , plev_weight)

    dR_Ta       = RK_compute_TPLL(ta_anom    ,f_RK.lw_ta    , plev_weight)
    dR_Tacs     = RK_compute_TPLL(ta_anom    ,f_RK.lwclr_ta , plev_weight)
    
    dR_LR       = RK_compute_TPLL(ta_anom_lr ,f_RK.lw_ta    , plev_weight)
    dR_LRcs     = RK_compute_TPLL(ta_anom_lr ,f_RK.lwclr_ta , plev_weight)
    
    dR_Ts       = RK_compute_suf (ts_anom     ,f_RK.lw_ts          )
    dR_Tscs     = RK_compute_suf (ts_anom     ,f_RK.lwclr_ts       )
    dR_alb      = RK_compute_suf (alb_anom    ,f_RK.sw_alb   *100  )
    dR_albcs    = RK_compute_suf (alb_anom    ,f_RK.swclr_alb*100  )

    ## dR due to cloud change
    Dcs_lw   = dRcs_lw - dR_Tacs - dR_Tscs - dR_wvcs_lw
    Dcs_sw   = dRcs_sw - dR_albcs - dR_wvcs_sw
    D_lw     = Dcs_lw / 1.16
    D_sw     = Dcs_sw / 1.16
    dR_c_lw  = dR_lw - D_lw - dR_Ta - dR_Ts - dR_wv_lw
    dR_c_sw  = dR_sw - D_sw - dR_alb - dR_wv_sw

    ## write to file
    ds_write = xr.Dataset()
    
    ds_write.coords['time'] = (('time'),var_pert['ts'].coords['time'].values)
    ds_write.coords['lat']  = (('lat'),var_pert['ts'].coords['lat'].values)
    ds_write.coords['lon']  = (('lon'),var_pert['ts'].coords['lon'].values)
    
    ds_write['dR_wv_lw']   = (('time','lat','lon'),dR_wv_lw.values  .astype('float32'))
    ds_write['dR_wv_sw']   = (('time','lat','lon'),dR_wv_sw.values  .astype('float32'))
    ds_write['dR_wvcs_lw'] = (('time','lat','lon'),dR_wvcs_lw.values.astype('float32'))
    ds_write['dR_wvcs_sw'] = (('time','lat','lon'),dR_wvcs_sw.values.astype('float32'))
    ds_write['dR_ta']      = (('time','lat','lon'),dR_Ta.values     .astype('float32'))
    ds_write['dR_tacs']    = (('time','lat','lon'),dR_Tacs.values   .astype('float32'))
    ds_write['dR_lr']      = (('time','lat','lon'),dR_LR.values     .astype('float32'))
    ds_write['dR_lrcs']    = (('time','lat','lon'),dR_LRcs.values   .astype('float32'))
    ds_write['dR_ts']      = (('time','lat','lon'),dR_Ts.values     .astype('float32'))
    ds_write['dR_tscs']    = (('time','lat','lon'),dR_Tscs.values   .astype('float32'))
    ds_write['dR_alb']     = (('time','lat','lon'),dR_alb.values    .astype('float32'))
    ds_write['dR_albcs']   = (('time','lat','lon'),dR_albcs.values  .astype('float32'))
    ds_write['dR_cloud_lw']= (('time','lat','lon'),dR_c_lw.values   .astype('float32'))
    ds_write['dR_cloud_sw']= (('time','lat','lon'),dR_c_sw.values   .astype('float32'))
    ds_write['Dcs_lw']     = (('time','lat','lon'),Dcs_lw.values    .astype('float32'))
    ds_write['Dcs_sw']     = (('time','lat','lon'),Dcs_sw.values    .astype('float32'))
    ds_write['dR_sw']      = (('time','lat','lon'),dR_sw.values     .astype('float32'))
    ds_write['dR_lw']      = (('time','lat','lon'),dR_lw.values     .astype('float32'))
    ds_write['dRcs_sw']    = (('time','lat','lon'),dRcs_sw.values   .astype('float32'))
    ds_write['dRcs_lw']    = (('time','lat','lon'),dRcs_lw.values   .astype('float32'))
    ds_write['ts']         = (('time','lat','lon'),var_pert['ts'].values.astype('float32'))
    ds_write['dts']        = (('time','lat','lon'),ts_anom.values   .astype('float32'))
    
    ds_write['dR_wv_lw_gm']    = (('time'),global_mean_xarray(ds_write.dR_wv_lw    ).values.astype('float32'))
    ds_write['dR_wv_sw_gm']    = (('time'),global_mean_xarray(ds_write.dR_wv_sw    ).values.astype('float32'))
    ds_write['dR_wvcs_lw_gm']  = (('time'),global_mean_xarray(ds_write.dR_wvcs_lw  ).values.astype('float32'))
    ds_write['dR_wvcs_sw_gm']  = (('time'),global_mean_xarray(ds_write.dR_wvcs_sw  ).values.astype('float32'))
    ds_write['dR_ta_gm']       = (('time'),global_mean_xarray(ds_write.dR_ta       ).values.astype('float32'))
    ds_write['dR_tacs_gm']     = (('time'),global_mean_xarray(ds_write.dR_tacs     ).values.astype('float32'))
    ds_write['dR_lr_gm']       = (('time'),global_mean_xarray(ds_write.dR_lr       ).values.astype('float32'))
    ds_write['dR_lrcs_gm']     = (('time'),global_mean_xarray(ds_write.dR_lrcs     ).values.astype('float32'))
    ds_write['dR_ts_gm']       = (('time'),global_mean_xarray(ds_write.dR_ts       ).values.astype('float32'))
    ds_write['dR_tscs_gm']     = (('time'),global_mean_xarray(ds_write.dR_tscs     ).values.astype('float32'))
    ds_write['dR_alb_gm']      = (('time'),global_mean_xarray(ds_write.dR_alb      ).values.astype('float32'))
    ds_write['dR_albcs_gm']    = (('time'),global_mean_xarray(ds_write.dR_albcs    ).values.astype('float32'))
    ds_write['dR_cloud_lw_gm'] = (('time'),global_mean_xarray(ds_write.dR_cloud_lw ).values.astype('float32'))
    ds_write['dR_cloud_sw_gm'] = (('time'),global_mean_xarray(ds_write.dR_cloud_sw ).values.astype('float32'))
    ds_write['Dcs_lw_gm']      = (('time'),global_mean_xarray(ds_write.Dcs_lw      ).values.astype('float32'))
    ds_write['Dcs_sw_gm']      = (('time'),global_mean_xarray(ds_write.Dcs_sw      ).values.astype('float32'))
    ds_write['dR_sw_gm']       = (('time'),global_mean_xarray(ds_write.dR_sw       ).values.astype('float32'))
    ds_write['dR_lw_gm']       = (('time'),global_mean_xarray(ds_write.dR_lw       ).values.astype('float32'))
    ds_write['dRcs_sw_gm']     = (('time'),global_mean_xarray(ds_write.dRcs_sw     ).values.astype('float32'))
    ds_write['dRcs_lw_gm']     = (('time'),global_mean_xarray(ds_write.dRcs_lw     ).values.astype('float32'))
    ds_write['ts_gm']          = (('time'),global_mean_xarray(ds_write.ts          ).values.astype('float32'))
    ds_write['dts_gm']         = (('time'),global_mean_xarray(ds_write.dts         ).values.astype('float32'))
    return ds_write

def check_dimensions(var_pert, var_cont,f_RK):   
    adjust_pressure_units(f_RK) 
    var2d_list = 'ts rlut rsdt  rsut  rlutcs rsutcs rsus  rsds'.split()
    var3d_list = 'ta hus'.split()
    f_RK_2d_shape = f_RK.lw_ts.shape[1:]
    f_RK_3d_shape = f_RK.lw_ta.shape[1:]
    flag = 1
    if f_RK.month.size!=12:
        raise Exception('Error: kernel files are not 12 month climatology')
    for var in var2d_list:
        if not (var_pert[var].shape[1:] == f_RK_2d_shape) :
            raise Exception(f'Error: Dimension is not same: check kernel and data. rk: {f_RK.lw_ts.shape} | data {var}: {var_pert[var].shape}')
        if not (var_cont[var].shape[1:] == f_RK_2d_shape) :
            raise Exception(f'Error: Dimension is not same: check kernel and data. rk: {f_RK.lw_ts.shape} | data {var}: {var_cont[var].shape}')
        if var_pert[var].time.size%12!=0:
            print('Warning: data files are not nx12 month. Results could be wrong due to mismatch between kernal month and data month')
        if var_cont[var].month.size!=12:
            raise Exception('Error: control data files are not 12 month climatology')
    for var in var3d_list:
        adjust_pressure_units(var_pert[var])
        adjust_pressure_units(var_cont[var])
        # check data dimensions # do not guarantee same axis values
        if not (var_pert[var].shape[1:] == f_RK_3d_shape):
            raise Exception(f'Error: Dimension is not same: check kernel and data. rk: {f_RK.lw_ta.shape} | data {var}: {var_pert[var].shape}')
        if not (var_cont[var].shape[1:] == f_RK_3d_shape) :
            raise Exception(f'Error: Dimension is not same: check kernel and data. rk: {f_RK.lw_ta.shape} | data {var}: {var_cont[var].shape}')
        # checking time/month axis
        if var_pert[var].time.size%12!=0:
            print('Warning: data files are not nx12 month. Results could be wrong due to mismatch between kernal month and data month')
        if var_cont[var].month.size!=12:
            raise Exception('Error: control data files are not 12 month climatology')
            # check pressure level values
        if not np.all(var_pert[var].plev.values == f_RK.plev.values):
            raise Exception(f'Error: plev is not same: check kernel and data. rk: {f_RK.plev.values} | data {var}: {var_pert[var].plev.values}')
        if not np.all(var_cont[var].plev.values == f_RK.plev.values) :
            raise Exception(f'Error: plev is not same: check kernel and data. rk: {f_RK.plev.values} | data {var}: {var_cont[var].plev.values}')
    return 

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

def adjust_pressure_units(var):
    if var.plev.attrs['units'] in ['mb','millibars','hPa','hpa',]:
        var['plev'] = var.plev*100
        var.plev.attrs['units'] = 'Pa'
    elif var.plev.attrs['units'] in ['pa', 'Pa']:
        pass
    else:
        raise Exception("Error: please check units of pressure level for dataset: plev not in ['Pa','mb','millibars','hPa','hpa',]")
    return 
    
def RK_plev_weight(plev_rk):
    """
    Create plev weight
    assume the surface pressure is 1.01e3 hPa 
    """
    # units check for plev in kernel file
    try:
        plev_rk.attrs['units'] 
    except:
        raise Exception("Error: Please check plev or its units. Assign with .attrs['units'] if not exist.")
        
    if plev_rk.attrs['units'] in ['mb','millibars','hPa','hpa',]:
        plev = plev_rk.values*100
    elif plev_rk.coords['plev'].attrs['units'] in ['pa', 'Pa']:
        plev = plev_rk.values
    else:
        raise Exception("Error: please check units of pressure level: plev. Not in ['Pa','mb','millibars','hPa','hpa',]")
    # compute weight for kernel files
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
    plev_weight = xr.DataArray(plev_weight,plev_rk.coords,dims=['plev'])

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

def RK_compute_TPLL(var, rk, plev_weight):
    tmp = var.groupby('time.month')*rk
    dR = tmp *plev_weight
    dR = dR.sum(dim='plev',skipna=True)
    return dR
