import platform, h5py, time, yaml, os
import numpy as np
import PyLab as PL
try:
    import py21cmfast as p21c # This is the p21f version!!!
except:
    # print("Cannot load p21c")
    pass

def copy_H5_att(f1, f2, skip = 1):
    dataset_names = ['astro_params', 'cosmo_params', 'flag_options', '_globals', 'user_params']
    skipped_names = ['F_STAR10', 'F_ESC10', 'L_X', 'm_chi', 'sigma_SDM']
    
    h1 = h5py.File(f1, 'r')
    h2 = h5py.File(f2, 'w')
    # General attrs
    h2.attrs['random_seed'] = h1.attrs['random_seed']
    h2.attrs['redshift'] = h1.attrs['redshift']
    for dataset_name in dataset_names:
        h2.create_group(dataset_name)
        for att_name in h1[dataset_name].attrs.keys():
            if not ((att_name in skipped_names) and skip):
                h2[dataset_name].attrs[att_name] = h1[dataset_name].attrs[att_name]
    h1.close()
    h2.close()
    
def save_lightcone(lc, datafile, global_quantities, lightcone_quantities, save_lc = 1):
    # I cannot save lc params directly as h5 attrs, write them to a swap yml file first and then re-read it to write it to h5
    swap_file = datafile + '_tmp.yml'
    fid = open(swap_file, 'w')
    for p in [lc.user_params, lc.cosmo_params, lc.flag_options, lc.astro_params]:
        print(p, file = fid)
    print('GlobalParams:', file = fid)
    for k in lc.global_params.keys():
        print('    ' + k + ':' + '  '+str(lc.global_params[k]), file=fid)
    fid.close()
    # Read params
    with open(swap_file, 'r') as f:
        params = yaml.safe_load(f)
    os.remove(swap_file)
    master_keys = ['AstroParams', 'CosmoParams', 'FlagOptions', 'GlobalParams', 'UserParams']
    aliases = ['astro_params', 'cosmo_params', 'flag_options', '_globals', 'user_params']
    keys_to_skip = [
        'external_table_path', 'wisdoms_path', 'LOG_SDGF_BARYONS', 'LOG_SDGF_CDM', 'LOG_SDGF_SDM', 'SIGMA_MZ',
        'LOG_Z_ARR', 'LOG_T_k', 'LOG_x_e', 'LOG_SIGF', 'LOG_T_chi', 'LOG_V_chi_b', 'LOG_K_ARR_FOR_TRANSFERS', 'T_M0_TRANSFER',
        'T_VCB_KIN_TRANSFER', 'T_V_CHI_B_ZHIGH_TRANSFER', 'LOG_K_ARR_FOR_SDGF', 'LOG_M_ARR', 'Z_ARRAY_FOR_SIGMA']

    # Write params
    with h5py.File(datafile, 'w') as h5:
        h5.attrs['random_seed'] = lc.random_seed
        h5.attrs['redshift'] = lc.redshift
        
        for idx in np.arange(0, len(master_keys)):
            master_key, alias = master_keys[idx], aliases[idx]
            h5.create_group(alias)
            dict_here = params[master_key]
            for k in dict_here.keys():
                if not k in keys_to_skip:
                    h5[alias].attrs[k] = dict_here[k]
                #else:
                #    h5[alias].attrs[k] = np.nan
        # saving global quantities
        h5.create_dataset('global_quantities/brightness_temp', data = lc.global_brightness_temp, dtype = 'float64')
        h5.create_dataset('node_redshifts', data = lc.node_redshifts, dtype = 'float64')
        
        if 'Tk_box' in global_quantities:
            h5.create_dataset('global_quantities/Tk_box', data = lc.global_Tk, dtype = 'float64')
        if 'Ts_box' in global_quantities:
            h5.create_dataset('global_quantities/Ts_box', data = lc.global_Ts, dtype = 'float64')
        if 'xH_box' in global_quantities:
            h5.create_dataset('global_quantities/xH_box', data = lc.global_xH, dtype = 'float64')
        if 'T_chi_box' in global_quantities:
            h5.create_dataset('global_quantities/T_chi_box', data = lc.global_T_chi, dtype = 'float64')
        if save_lc:
            if 'brightness_temp' in lightcone_quantities:
                h5.create_dataset('lightcones/brightness_temp', data = lc.brightness_temp, dtype = 'float32')

def Run_21cmFirstCLASS_MINI(
        F_STAR10 = -1.25,
        F_ESC10 = -1.35,
        L_X = 40.5,
        ALPHA_STAR = 0.5,
        ALPHA_ESC = -0.3,
        F_STAR7_MINI = -2.5,
        L_X_MINI = 40.5,
        NU_X_THRESH = 500,
        F_ESC7_MINI = -1.35,
        ALPHA_STAR_MINI = 0.0,
        A_LW = 2.0,
        mdm = -3,
        sigma = -200,
        HII_DIM = 50,
        BOX_LEN = 100,
        zmin = 10,
        SDM_TARGET_TYPE = 'BARYONS',
        verbose = 1,
        # MANY_Z_SAMPLES_AT_COSMIC_DAWN = False,
        N_THREADS = 1,
        datafile = None,
        save_lc = 0
    ):
    '''
    An interface to 21cmFirstCLASS, all params are entered in log10
    mdm : DM mass in GeV
    sigma : sigma / (v/c)^{-4}, in cm^2
    zmin : minimum z in simulation
    SDM_TARGET_TYPE: particles that DM scatters with
        BARYONS (all the baryons)
        IONIZED (free protons and electrons)
        HYDROGEN (hydrogen nuclei, neutralized or not)
        PROTONS (free protons)
        ELECTRONS (free electrons)
    verbose: print out useful debug and dev info (run time, etc)?
    N_THREADS: number of cpu per run
    datafile: if not None, this is the data file to save results. lc.save doesn't seem to be working but it's not very disk efficient anyway if we don't want lc
    '''
    if NU_X_THRESH < 100.0:
        raise Exception("NU_X_THRESH cannot be below 100, NU_X_THRESH = {:.3E}".format(NU_X_THRESH))
    ncpu = 1 if platform.system() == 'Darwin' else N_THREADS # M-chip Mac cannot do mpi
    user_params = {
        'BOX_LEN' : BOX_LEN, # size of the simulated box (in comoving Mpc) 
        'HII_DIM' : HII_DIM, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
        'N_THREADS' : ncpu,
        'SDM_TARGET_TYPE' : SDM_TARGET_TYPE,
        'SCATTERING_DM' : True,
        'FUZZY_DM' : False,
        'FAST_FCOLL_TABLES': True,
        'USE_RELATIVE_VELOCITIES': True,
        # 'MANY_Z_SAMPLES_AT_COSMIC_DAWN': MANY_Z_SAMPLES_AT_COSMIC_DAWN,
        # Other background params
        'USE_INTERPOLATION_TABLES': True}
    flag_options = {
        'USE_MINI_HALOS': True, 
        'USE_MASS_DEPENDENT_ZETA': True,
        'INHOMO_RECO': True,
        'USE_TS_FLUCT': True}
    # Cosmo params with PLK18
    cosmo_params = {
        'hlittle': 0.6766, # hubble parameter
        'OMb': 0.04897468161, # baryon density
        'OMm': 0.30964168161, # matter (CDM+baryon) density
        'A_s': 2.105209331337507e-09, # amplitude of the primordial fluctuations
        'POWER_INDEX': 0.9665, # spectral index of the primordial spectrum
        'tau_reio': 0.0561, # optical depth to reionization
        'm_chi' : mdm + 9, # log10 GeV to log10 eV
        'sigma_SDM' : -sigma, # -log10[sigma / (v/c)^{-4} / cm^2]
        'f_chi' : 0, # fraction of SDM. This is in fact -log_10(f_chi), thus we simulate here SDM with a fraction of 100%
        'SDM_INDEX' : -4, # index of the SDM cross-section. Note that it doesn't have to be an integer!
    }
    astro_params = {
        'F_STAR10': F_STAR10, # star formation efficiency (atomic cooling galaxies) for pivot mass 1e10 Msun (log10)
        'ALPHA_STAR': ALPHA_STAR, # slope of the dependency of star formation efficiency on the host halo mass 
        'F_ESC10': F_ESC10, # escape fraction of Lyman photons into the IGM for pivot mass 1e10 Msun (log10)
        'ALPHA_ESC': ALPHA_ESC, # slope of the dependency of escape fraction on the host halo mass 
        'L_X': L_X,
        'F_ESC7_MINI': F_ESC7_MINI,
        'ALPHA_STAR_MINI': ALPHA_STAR_MINI,
        'A_LW': A_LW,
        'F_STAR7_MINI': F_STAR7_MINI,
        'L_X_MINI': L_X_MINI,
        'NU_X_THRESH': NU_X_THRESH,
        't_STAR': 0.5,
        'X_RAY_SPEC_INDEX': -0.5,# X-ray luminosity (log10)
    }
    
    global_quantities = (
        'brightness_temp', # brightness temperature
        'J_Lya_box', # Lyman alpha flux 
        'Tk_box', # baryon kinetic temperature
        'Ts_box', # baryon spin temperature
        'xH_box', 
        'T_chi_box', # SDM temperature
        'V_chi_b_box') # Relative velocity between baryons and SDM
    lightcone_quantities = ("brightness_temp",) # brightness temperature
    t1 = time.time()
    lc = p21c.run_lightcone(
        redshift = zmin, # minimum redshift at which the simulation will stop
        random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
        regenerate = True, # create new data even if cached are found
        write = False, # whether or no to save cached files
        user_params = user_params,
        astro_params = astro_params,
        flag_options = flag_options,
        cosmo_params = cosmo_params,
        global_quantities = global_quantities,
        lightcone_quantities = lightcone_quantities)
    t2 = time.time()
    if verbose: print("Time used: {:.2f} s".format(t2 - t1))

    result = {
        'lc': lc,
        'z': np.array(lc.node_redshifts),
        'Tb': lc.global_brightness_temp,
        'Tk' :  lc.global_Tk,
        'Ts' :  lc.global_Ts,
        'Tx' : lc.global_T_chi, 
        'xH' :  lc.global_xH,
    }
    z_lc = lc.lightcone_redshifts
    Tb_lc = lc.brightness_temp
    mask = z_lc < 40.1
    z_lc = z_lc[mask]
    Tb_lc = Tb_lc[:,:,mask]
    Tb_lc = np.array(Tb_lc).astype(np.float32)
                
    if not datafile is None:
        # For some reasons lc.save doesn't work?
        if datafile[-1] == 'z':
            if save_lc:
                raise Exception("To save LC please use h5")
            else:
                np.savez(
                    datafile,
                    z = lc.node_redshifts,
                    Tb = lc.global_brightness_temp,
                    Tk = lc.global_Tk,
                    Ts = lc.global_Ts,
                    Tx = lc.global_T_chi,
                    xH = lc.global_xH,
                    F_STAR10 = F_STAR10,
                    F_ESC10 = F_ESC10,
                    L_X = L_X,
                    ALPHA_STAR = ALPHA_STAR,
                    ALPHA_ESC = ALPHA_ESC,
                    F_STAR7_MINI = F_STAR7_MINI,
                    L_X_MINI = L_X_MINI,
                    NU_X_THRESH = NU_X_THRESH,
                    F_ESC7_MINI = F_ESC7_MINI,
                    ALPHA_STAR_MINI = ALPHA_STAR_MINI,
                    A_LW = A_LW,
                    mdm = mdm,
                    sigma = sigma,
                    HII_DIM = HII_DIM,
                    BOX_LEN = BOX_LEN)
        else:
            save_lightcone(lc = lc, datafile=datafile, global_quantities=global_quantities,lightcone_quantities=lightcone_quantities,save_lc=save_lc)
    return result

# ---- OPTIONAL ----
# For Park18 astro and MC inference

def Run_21cmFirstCLASS(
        F_STAR10 = -1.301,
        F_ESC10 = -1.0,
        L_X = 40.5,
        M_TURN = 8.7,
        ALPHA_STAR = 0.5,
        ALPHA_ESC = -0.5,
        mdm = -3,
        sigma = -200,
        HII_DIM = 25,
        BOX_LEN = 100,
        zmin = 10,
        SDM_TARGET_TYPE = 'BARYONS',
        Z_HEAT_MAX = 35.0,
        verbose = 1,
        # MANY_Z_SAMPLES_AT_COSMIC_DAWN = False,
        N_THREADS = 1,
        datafile = None,
        save_lc = 0
    ):
    '''
    An interface to 21cmFirstCLASS, all params are entered in log10
    mdm : DM mass in GeV
    sigma : sigma / (v/c)^{-4}, in cm^2
    zmin : minimum z in simulation
    SDM_TARGET_TYPE: particles that DM scatters with
        BARYONS (all the baryons)
        IONIZED (free protons and electrons)
        HYDROGEN (hydrogen nuclei, neutralized or not)
        PROTONS (free protons)
        ELECTRONS (free electrons)
    verbose: print out useful debug and dev info (run time, etc)?
    N_THREADS: number of cpu per run
    datafile: if not None, this is the data file to save results. lc.save doesn't seem to be working but it's not very disk efficient anyway if we don't want lc
    '''
    ncpu = 1 if platform.system() == 'Darwin' else N_THREADS # M-chip Mac cannot do mpi
    user_params = {
        'BOX_LEN' : BOX_LEN, # size of the simulated box (in comoving Mpc) 
        'HII_DIM' : HII_DIM, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
        'N_THREADS' : ncpu,
        'SDM_TARGET_TYPE' : SDM_TARGET_TYPE,
        'SCATTERING_DM' : True,
        'FUZZY_DM' : False,
        # 'MANY_Z_SAMPLES_AT_COSMIC_DAWN': MANY_Z_SAMPLES_AT_COSMIC_DAWN,
        # Other background params
        'USE_INTERPOLATION_TABLES': True}
    flag_options = {
        'USE_MINI_HALOS': False, 
        'USE_MASS_DEPENDENT_ZETA': True,
        'INHOMO_RECO': True,
        'USE_TS_FLUCT': True}
    # Cosmo params with PLK18
    cosmo_params = {
        'hlittle': 0.6766, # hubble parameter
        'OMb': 0.04897468161, # baryon density
        'OMm': 0.30964168161, # matter (CDM+baryon) density
        'A_s': 2.105209331337507e-09, # amplitude of the primordial fluctuations
        'POWER_INDEX': 0.9665, # spectral index of the primordial spectrum
        'tau_reio': 0.0561, # optical depth to reionization
        'm_chi' : mdm + 9, # log10 GeV to log10 eV
        'sigma_SDM' : -sigma, # -log10[sigma / (v/c)^{-4} / cm^2]
        'f_chi' : 0, # fraction of SDM. This is in fact -log_10(f_chi), thus we simulate here SDM with a fraction of 100%
        'SDM_INDEX' : -4, # index of the SDM cross-section. Note that it doesn't have to be an integer!
    }
    astro_params = {
        'F_STAR10': F_STAR10, # star formation efficiency (atomic cooling galaxies) for pivot mass 1e10 Msun (log10)
        'ALPHA_STAR': ALPHA_STAR, # slope of the dependency of star formation efficiency on the host halo mass 
        'F_ESC10': F_ESC10, # escape fraction of Lyman photons into the IGM for pivot mass 1e10 Msun (log10)
        'ALPHA_ESC': ALPHA_ESC, # slope of the dependency of escape fraction on the host halo mass 
        'L_X': L_X,
        'NU_X_THRESH': 500.0,
        't_STAR': 0.5,
        'M_TURN': M_TURN,
        'X_RAY_SPEC_INDEX': -0.5,# X-ray luminosity (log10)
    }
    
    global_quantities = (
        'brightness_temp', # brightness temperature
        'J_Lya_box', # Lyman alpha flux 
        'Tk_box', # baryon kinetic temperature
        'Ts_box', # baryon spin temperature
        'xH_box', 
        'T_chi_box', # SDM temperature
        'V_chi_b_box') # Relative velocity between baryons and SDM
    global_params = {
        'Z_HEAT_MAX': Z_HEAT_MAX
    }
    lightcone_quantities = ("brightness_temp",) # brightness temperature
    t1 = time.time()
    lc = p21c.run_lightcone(
        redshift = zmin, # minimum redshift at which the simulation will stop
        random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
        regenerate = True, # create new data even if cached are found
        write = False, # whether or no to save cached files
        user_params = user_params,
        global_params = global_params,
        max_redshift = Z_HEAT_MAX,
        astro_params = astro_params,
        flag_options = flag_options,
        cosmo_params = cosmo_params,
        global_quantities = global_quantities,
        lightcone_quantities = lightcone_quantities)
    t2 = time.time()
    if verbose: print("Time used: {:.2f} s".format(t2 - t1))

    result = {
        'lc': lc,
        'z': np.array(lc.node_redshifts),
        'Tb': lc.global_brightness_temp,
        'Tk' :  lc.global_Tk,
        'Ts' :  lc.global_Ts,
        'Tx' : lc.global_T_chi, 
        'xH' :  lc.global_xH,
    }
    z_lc = lc.lightcone_redshifts
    Tb_lc = lc.brightness_temp
    mask = z_lc < 40.1
    z_lc = z_lc[mask]
    Tb_lc = Tb_lc[:,:,mask]
    Tb_lc = np.array(Tb_lc).astype(np.float32)
                
    if not datafile is None:
        # For some reasons lc.save doesn't work?
        if datafile[-1] == 'z':
            if save_lc:
                np.savez(
                    datafile,
                    z = lc.node_redshifts,
                    Tb = lc.global_brightness_temp,
                    Tk = lc.global_Tk,
                    Ts = lc.global_Ts,
                    Tx = lc.global_T_chi,
                    xH = lc.global_xH,
                    Tb_lc = Tb_lc,
                    z_lc = z_lc,
                    F_STAR10 = F_STAR10,
                    F_ESC10 = F_ESC10,
                    L_X = L_X,
                    mdm = mdm,
                    sigma_dm = sigma,
                    BOX_LEN = BOX_LEN,
                    HII_DIM = HII_DIM)
            else:
                np.savez(
                    datafile,
                    z = lc.node_redshifts,
                    Tb = lc.global_brightness_temp,
                    Tk = lc.global_Tk,
                    Ts = lc.global_Ts,
                    Tx = lc.global_T_chi,
                    xH = lc.global_xH,
                    F_STAR10 = F_STAR10,
                    F_ESC10 = F_ESC10,
                    L_X = L_X,
                    mdm = mdm,
                    sigma_dm = sigma,
                    BOX_LEN = BOX_LEN,
                    HII_DIM = HII_DIM)
        else:
            save_lightcone(lc = lc, datafile=datafile, global_quantities=global_quantities,lightcone_quantities=lightcone_quantities,save_lc=save_lc)
    return result

def Find_datafile(fst, fesc, LX, mx, sx, use_h5 = 0):
    if platform.system()=='Darwin':
        data_path = '/Volumes/X320/SDM_DataSet/'
    else:
        data_path = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/DataSet/'
    sim_name = "{:.3f}_".format(fst) + "{:.3f}_".format(fesc) + "{:.3f}_".format(LX) + "{:.3f}_".format(mx) + "{:.3f}".format(sx)
    suffix = '.h5' if use_h5 else '.npz'
    r = data_path + 'LC_' + sim_name + suffix
    return r

def Load_DataSet():
    def apply_redundancy(x):
        new_x = x
        redundancy_factor = 0.001
        new_x[0] = x[0] - redundancy_factor
        new_x[-1] = x[-1] + redundancy_factor
        return new_x
    if platform.system()=='Darwin':
        DataSet_File = '/Users/cangtao/Desktop/datasets/SDM_lite/DataSet_Sumaries.h5'
    else:
        DataSet_File = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/DataSet_Sumaries.h5'
    with h5py.File(DataSet_File, 'r') as h5f:
        '''
        r = {
            'z': h5f['z'][:],
            'Tb': h5f['Tb'][:],
            'fst': apply_redundancy(h5f['params/fst'][:]),
            'fesc': apply_redundancy(h5f['params/fesc'][:]),
            'LX': apply_redundancy(h5f['params/LX'][:]), 
            'mx': apply_redundancy(h5f['params/mx'][:]),
            'sx': apply_redundancy(h5f['params/sx'][:])}
        '''
        r = {
            'z': h5f['z'][:].astype(np.float64),
            'Tb': h5f['Tb'][:].astype(np.float64),
            'Tk': h5f['Tk'][:].astype(np.float64),
            'fst': apply_redundancy(h5f['params/fst'][:]).astype(np.float64),
            'fesc': apply_redundancy(h5f['params/fesc'][:]).astype(np.float64),
            'LX': apply_redundancy(h5f['params/LX'][:]).astype(np.float64), 
            'mx': apply_redundancy(h5f['params/mx'][:]).astype(np.float64),
            'sx': apply_redundancy(h5f['params/sx'][:]).astype(np.float64)}
        
    return r

try:
    Interp_Dataset = Load_DataSet()
except:
    # print("Cannot load Interp_Dataset")
    Interp_Dataset = None

def Read_dataset(fst, fesc, LX, mx, sx):
    def find_idx(x, xax):
        dif = np.abs(x - xax)
        if np.min(dif) > 1E-3:
            raise Exception("Cannot find a match in dataset, input = {:.3E}, {:.3E}, {:.3E}, {:.3E}, {:.3E}".format(fst, fesc, LX, mx, sx))
        idx = np.argmin(dif)
        return idx
    idx_fst = find_idx(fst, Interp_Dataset['fst'])
    idx_fesc = find_idx(fesc, Interp_Dataset['fesc'])
    idx_LX = find_idx(LX, Interp_Dataset['LX'])
    idx_mx = find_idx(mx, Interp_Dataset['mx'])
    idx_sx = find_idx(sx, Interp_Dataset['sx'])
    r = Interp_Dataset['Tb'][idx_fst, idx_fesc, idx_LX, idx_mx, idx_sx, :]
    return r

def Interpolate_DataSet(fst, fesc, LX, mx, sx, type = 'Tb'):
    '''
    Screw scipy, let's use my own interpolation code
    '''

    input_params = [fst, fesc, LX, mx, sx]
    param_axis = [Interp_Dataset[name] for name in ['fst', 'fesc', 'LX', 'mx', 'sx']]
    def find_indexes(param_idx):
        x = input_params[param_idx]
        xax = param_axis[param_idx]
        id1 = PL.Find_Index(x = x, x_axis = xax)
        id2 = id1 + 1
        return id1, id2, xax[id1], xax[id2], input_params[param_idx]
    
    def linear_interp_fun(x, x1, x2, y1, y2):
        r = (y2 - y1) * (x - x1) / (x2 - x1) + y1
        if True in np.isnan(r):
            raise Exception("interpolation result is nan")
        return r
    
    # fst
    idx = 0
    id1, id2, x1, x2, x = find_indexes(idx)
    y1 = Interp_Dataset[type][id1, :, :, :, :, :]
    y2 = Interp_Dataset[type][id2, :, :, :, :, :]
    Tab = linear_interp_fun(x, x1, x2, y1, y2)
    
    # fesc
    idx = 1
    id1, id2, x1, x2, x = find_indexes(idx)
    y1 = Tab[id1, :, :, :, :]
    y2 = Tab[id2, :, :, :, :]
    Tab = linear_interp_fun(x, x1, x2, y1, y2)
    
    # LX
    idx = 2
    id1, id2, x1, x2, x = find_indexes(idx)
    y1 = Tab[id1, :, :, :]
    y2 = Tab[id2, :, :, :]
    Tab = linear_interp_fun(x, x1, x2, y1, y2)
    
    # mx
    idx = 3
    id1, id2, x1, x2, x = find_indexes(idx)
    y1 = Tab[id1, :, :]
    y2 = Tab[id2, :, :]
    Tab = linear_interp_fun(x, x1, x2, y1, y2)
    
    # sx, takes a bit of work
    if sx >= Interp_Dataset['sx'][1]:
        idx = 4
        id1, id2, x1, x2, x = find_indexes(idx)
    else:
        id1, id2 = 0, 1
        x1, x2 = 0.0, 10.0**Interp_Dataset['sx'][1]
        x = 10.0**sx
    y1 = Tab[id1, :]
    y2 = Tab[id2, :]

    r = linear_interp_fun(x, x1, x2, y1, y2)
    return r
