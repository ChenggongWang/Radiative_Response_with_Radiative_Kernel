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

def decompose_dR_rk_toa_core_numpy(var_pert, var_cont,f_RK ):

    check_dimensions(var_pert, var_cont,f_RK)
    ta_anom = diff_pert_mon_cont_12mon_TPLL_fast(var_pert['ta'].values,var_cont['ta'].values)
    omega_wv = omega_wv_fast(var_pert['hus'].values,\
                             var_cont['hus'].values,\
                             var_cont['ta'].values)
    ts_anom = diff_pert_mon_cont_12mon_TLL_fast(var_pert['ts'].values, \
                                                var_cont['ts'].values)
    ta_anom_lr = ta_anom - np.broadcast_to(ts_anom[:,np.newaxis,:,:],ta_anom.shape)
    alb_anom_100 = alb_diff_pert_mon_cont_12mon_TLL_fast(var_pert['rsus'].values, \
                                                         var_pert['rsds'].values, \
                                                         var_cont['rsus'].values, \
                                                         var_cont['rsds'].values)
    dR_sw   = diff_pert_mon_cont_12mon_TLL_fast((var_pert['rsdt'].values-var_pert['rsut'].values),\
                                                (var_cont['rsdt'].values-var_cont['rsut'].values) )
    dR_lw   = diff_pert_mon_cont_12mon_TLL_fast((-var_pert['rlut'].values),\
                                                (-var_cont['rlut'].values) )
    dRcs_sw = diff_pert_mon_cont_12mon_TLL_fast((var_pert['rsdt'].values-var_pert['rsutcs'].values),\
                                                (var_cont['rsdt'].values-var_cont['rsutcs'].values) )
    dRcs_lw = diff_pert_mon_cont_12mon_TLL_fast((-var_pert['rlutcs'].values),\
                                                (-var_cont['rlutcs'].values) )
    plev_weight = RK_plev_weight(f_RK.plev)
    
    dR_wv_lw    = RK_compute_TPLL_plev_fast(omega_wv    ,f_RK.lw_q.values     , plev_weight)
    dR_wv_sw    = RK_compute_TPLL_plev_fast(omega_wv    ,f_RK.sw_q.values     , plev_weight)
    dR_wvcs_lw  = RK_compute_TPLL_plev_fast(omega_wv    ,f_RK.lwclr_q.values  , plev_weight)
    dR_wvcs_sw  = RK_compute_TPLL_plev_fast(omega_wv    ,f_RK.swclr_q.values  , plev_weight)
    
    dR_Ta       = RK_compute_TPLL_plev_fast(ta_anom     ,f_RK.lw_ta.values    , plev_weight)
    dR_Tacs     = RK_compute_TPLL_plev_fast(ta_anom     ,f_RK.lwclr_ta.values , plev_weight)
    
    dR_LR       = RK_compute_TPLL_plev_fast(ta_anom_lr  ,f_RK.lw_ta.values    , plev_weight)
    dR_LRcs     = RK_compute_TPLL_plev_fast(ta_anom_lr  ,f_RK.lwclr_ta.values , plev_weight)
    
    dR_Ts       = RK_compute_TLL_fast       (ts_anom     ,f_RK.lw_ts.values     )
    dR_Tscs     = RK_compute_TLL_fast       (ts_anom     ,f_RK.lwclr_ts.values  )
    dR_alb      = RK_compute_TLL_fast       (alb_anom_100,f_RK.sw_alb.values    )
    dR_albcs    = RK_compute_TLL_fast       (alb_anom_100,f_RK.swclr_alb.values )

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
    
    ds_write['dR_wv_lw']   = (('time','lat','lon'),dR_wv_lw)
    ds_write['dR_wv_sw']   = (('time','lat','lon'),dR_wv_sw)
    ds_write['dR_wvcs_lw'] = (('time','lat','lon'),dR_wvcs_lw)
    ds_write['dR_wvcs_sw'] = (('time','lat','lon'),dR_wvcs_sw)
    ds_write['dR_ta']      = (('time','lat','lon'),dR_Ta)
    ds_write['dR_tacs']    = (('time','lat','lon'),dR_Tacs)
    ds_write['dR_lr']      = (('time','lat','lon'),dR_LR)
    ds_write['dR_lrcs']    = (('time','lat','lon'),dR_LRcs)
    ds_write['dR_ts']      = (('time','lat','lon'),dR_Ts)
    ds_write['dR_tscs']    = (('time','lat','lon'),dR_Tscs)
    ds_write['dR_alb']     = (('time','lat','lon'),dR_alb)
    ds_write['dR_albcs']   = (('time','lat','lon'),dR_albcs)
    ds_write['dR_cloud_lw']    = (('time','lat','lon'),dR_c_lw)
    ds_write['dR_cloud_sw']    = (('time','lat','lon'),dR_c_sw)
    ds_write['Dcs_lw']     = (('time','lat','lon'),Dcs_lw)
    ds_write['Dcs_sw']     = (('time','lat','lon'),Dcs_sw)
    ds_write['dR_sw']      = (('time','lat','lon'),dR_sw)
    ds_write['dR_lw']      = (('time','lat','lon'),dR_lw)
    ds_write['dRcs_sw']    = (('time','lat','lon'),dRcs_sw)
    ds_write['dRcs_lw']    = (('time','lat','lon'),dRcs_lw)
    ds_write['ts']         = (('time','lat','lon'),var_pert['ts'].values.astype('float32'))
    ds_write['dts']        = (('time','lat','lon'),ts_anom)
    
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
    return plev_weight
    

def alb_diff_pert_mon_cont_12mon_TLL_fast(pert_rsus,pert_rsds,cont_rsus,cont_rsds):
    """
    Compute albedo anomaly
    ALL inputs have to be numpy array
    """
    alb_diff = np.empty_like(pert_rsus,dtype = np.float32)
    for mon in range(cont_rsus.shape[0]):
        alb_cont_mon = np.empty_like(cont_rsus[0,:,:],dtype = np.float32)
        alb_cont_mon = cont_rsus[mon,:,:] / cont_rsds[mon,:,:]
        alb_cont_mon = np.where((alb_cont_mon < 1) & (np.isfinite(alb_cont_mon)), alb_cont_mon , 1 )
        for i in range(int(pert_rsus.shape[0]/12)):
            alb_pert_mon = np.empty_like(cont_rsus[0,:,:],dtype = np.float32)
            alb_pert_mon = pert_rsus[i*12+mon,:,:] / pert_rsds[i*12+mon,:,:]
            alb_pert_mon = np.where((alb_pert_mon < 1) & (np.isfinite(alb_pert_mon)), alb_pert_mon , 1)            
            ## use 100 factor due to kernel file
            alb_diff[i*12+mon,:,:] = (alb_pert_mon  - alb_cont_mon)*100  
    return alb_diff

def diff_pert_mon_cont_12mon_TLL_fast(pert_TLL,cont_TLL):
    """
    Compute anomaly between perturbation and control experiment
    For surface variable [Time, Lat, Lon] (TLL)
    ALL inputs have to be numpy array
    """
    diff = np.empty_like(pert_TLL,dtype = np.float32)
    for i in range(pert_TLL.shape[0]):
        mon = i%12
        diff[i,:,:] = pert_TLL[i,:,:] - cont_TLL[mon,:,:]
    return diff

def diff_pert_mon_cont_12mon_TPLL_fast(pert_TPLL,cont_TPLL):
    """
    Compute anomaly between perturbation and control experiment
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
    diff = np.empty_like(pert_TPLL,dtype = np.float32)
    for i in range(pert_TPLL.shape[0]):
        mon = i%12
        diff[i,:,:,:] = pert_TPLL[i,:,:,:] - cont_TPLL[mon,:,:,:]
    return diff

def omega_wv_fast(hus_pert_TPLL,hus_cont_TPLL,ta_cont_TPLL):
    """
    Convert humidity to omega (the variable used by water vapor kernel)
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
    omega_wv = np.zeros_like(hus_pert_TPLL,dtype = np.float32)
    dT_dlnq_12mon = _dT_dlnqs_fast(ta_cont_TPLL)
    for i in range(hus_pert_TPLL.shape[0]):
        mon = i%12
        pert = hus_pert_TPLL[i,:,:,:]
        cont = hus_cont_TPLL[mon,:,:,:]
        omega_wv[i,:,:,:] =  dT_dlnq_12mon[mon,:,:,:]*np.log(pert/cont)
    return omega_wv

def _dT_dlnqs_fast(T):
    # use CC-equation to derive dT/dln(q_s) 
    dTdlnqs = np.empty_like(T,dtype = np.float32)
    dTdlnqs = (T-29.65)**2/4302.64
    return dTdlnqs


def RK_compute_TLL_fast(var_mon, rk_mon_cli):
    """    
    Compute radition change due to anomaly of var_mon
    For surface variable [Time, Lat, Lon] (TLL)
    ALL inputs have to be numpy array
    """
    dR_TLL = np.empty_like(var_mon,dtype = np.float32)
    for i in range(var_mon.shape[0]):
        mon = i%12
        dR_TLL[i,:,:] = rk_mon_cli[mon,:,:] * var_mon[i,:,:]
    return dR_TLL

def RK_compute_TPLL_plev_fast(var_mon, rk_mon_cli, plev_weight):
    """
    Compute radition change due to anomaly of var_mon
    For 3D variable [Time, Plev, Lat, Lon] (TPLL)
    ALL inputs have to be numpy array
    """
    dR_TLL = np.zeros_like(var_mon[:,0,:,:],dtype = np.float32)  
    rk_mon_cli_nonan = np.where(np.isnan(rk_mon_cli),0,rk_mon_cli)
    for i in range(var_mon.shape[0]):
        mon = i%12
        for pi in range(var_mon.shape[1]):
            # skip NaN values (empty location due to topography)
#             if np.isnan(rk_mon_cli[mon,pi,:,:]) or np.isnan(var_mon[i,pi,:,:]): 
#                 continue
            # set to zero in numpy functions
            var_tmp = np.where(np.isnan(var_mon[i,pi,:,:]),0,var_mon[i,pi,:,:])
            var_tmp = np.where(np.isnan(rk_mon_cli[mon,pi,:,:]),0,var_tmp)
            dR_TLL[i,:,:] += rk_mon_cli_nonan[mon,pi,:,:]* var_tmp * plev_weight[pi]
    return dR_TLL
