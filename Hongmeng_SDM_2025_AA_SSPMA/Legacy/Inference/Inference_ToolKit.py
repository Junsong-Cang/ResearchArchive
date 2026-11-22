import numpy as np
import os, sys, platform, time
import PyLab as PL
if platform.system() == 'Darwin':
    sys.path.insert(1, '/Users/cangtao/FileVault/Projects/SDM_lite/ToolKit/')
else:
    sys.path.insert(1, '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/ToolKit/')
from DataSetTools import Interpolate_DataSet, Interp_Dataset

def T_FG_LinLog(linear_params, freq):
    '''
    Get foreground temp in mK for LinLog Model
    ---- inputs ----
    params: FG param, in [p0, p1, p2, ...]
    freq: ferquency in MHz
    '''
    b = -2.53
    N_FG = len(linear_params)
    f0 = 75
    t = 0
    for idx in np.arange(0, N_FG):
        t = t + linear_params[idx] * (np.log(freq/f0))**(idx)
    t = t* (freq/f0)**b * 1000 # in mK
    return t

def T_rcv(v):
    '''
    Receiver temp in mK
    -- input --
    v : frequency in MHz
    '''
    T0 = 200
    vc = 75
    p = (v/vc - 0.4)**1.277
    p = -p/0.0808
    p = 16.809 * np.exp(p) + 1.034
    T = T0 * p
    T = T * 1000
    return T

def Get_HM_Data():
    astro_params = {
        'fst': -1.301,
        'fesc': -1.0,
        'LX': 40.5,
        'mx': 2,
        'sx': -100.0
    }
    freq = np.arange(30, 120, 0.061)
    redshifts = 1420/freq - 1
    T_ax = Interpolate_DataSet(
        fst = astro_params['fst'],
        fesc = astro_params['fesc'],
        LX = astro_params['LX'],
        mx = astro_params['mx'],
        sx = astro_params['sx'])
    z_ax = Interp_Dataset['z']
    T_cosmic = np.interp(x = redshifts, xp = z_ax[::-1], fp = T_ax[::-1], left = np.nan, right = np.nan)
    if True in np.isnan(T_cosmic):
        raise Exception('T_cosmic has nan')
    FG_params = [
        1697.8076928220983,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0]
    T_FG = T_FG_LinLog(FG_params, freq)
    Tsky = T_cosmic + T_FG
    Trcv = T_rcv(freq)
    data = {
        'freq':freq,
        'Tsky':Tsky,
        'Trcv': Trcv,
        'redshifts':redshifts, 
        'FG_params': FG_params,
        'astro_params': astro_params,
        'z_ax': z_ax,
        'T_FG': T_FG,
        'T_ax': T_ax}
    return data

HM_Data = Get_HM_Data()
if platform.system() == 'Darwin':
    chain_path = '/Users/cangtao/FileVault/Projects/SDM_lite/data/chains/'
else:
    chain_path = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/chains/'

def compute_noise(tobs_yr = 5, block_sun = 0):
    '''
    Compute HM noise in mK
    '''
    eff = 0.1 if block_sun else 0.15
    Tsky = HM_Data['Tsky']
    Trcv = HM_Data['Trcv']
    Tsys = Tsky + Trcv
    dv = 0.4E6 # Hz
    seconds_in_year = 365.25 * 24 * 3600
    Tobs = tobs_yr * seconds_in_year
    Tobs_eff = eff * Tobs # in s
    sigma = Tsys / np.sqrt(dv * Tobs_eff)
    return sigma

def Log_Like_HM(
        fst,
        fesc,
        LX,
        mx,
        sx,
        LinLog_FG_Params,
        Tobs_yr,
        block_sun):
    '''
    Get HM Likelihood, all astro params are in log10
    '''
    # Get cosmic signal
    T_ax = Interpolate_DataSet(fst = fst, fesc = fesc, LX = LX, mx = mx, sx = sx)
    z_ax = Interp_Dataset['z']
    T_cosmic = np.interp(x = HM_Data['redshifts'], xp = z_ax[::-1], fp = T_ax[::-1], left = np.nan, right = np.nan)
    
    # FG
    T_FG = T_FG_LinLog(LinLog_FG_Params, HM_Data['freq'])

    # Now get Chi2
    T_theory = T_cosmic + T_FG
    T_sky = HM_Data['Tsky']
    sigma = compute_noise(Tobs_yr, block_sun)
    sigma2 = sigma**2.0

    Chi2 = ((T_theory - T_sky)**2)/sigma2
    Chi2 = np.sum(Chi2)
    
    # Normalise properly for Bayesian factor
    PreFactor = 1/np.sqrt(2 * np.pi * sigma2)
    PreFactor = np.sum(np.log(PreFactor))
    LnL = -0.5 * Chi2 + PreFactor
    return LnL

def Log_Like_kernel(
        theta,
        Tobs_yr,
        block_sun,
        sx_unit,
        fixed_log10_mx,
        use_log10_sx):
    ndim = len(theta)
    if fixed_log10_mx is None:
        n_astro = 5
        mx = theta[3]
        sid = 4
    else:
        n_astro = 4
        mx = fixed_log10_mx
        sid = 3
    if use_log10_sx:
        sx = theta[sid]
    else:
        sx = np.log10(theta[sid] * sx_unit)
    NFG = ndim - n_astro
    if NFG == 0:
        FG_Params = HM_Data['FG_params'] # Fixing FG
    elif NFG == 1:
        FG_Params = np.array([theta[-1], 0.0, 0.0]) # My FG function cannot handle scalar yet
    else:
        FG_Params = theta[n_astro:n_astro+NFG]
    LnL = Log_Like_HM(
        fst = theta[0],
        fesc = theta[1],
        LX = theta[2],
        mx = mx,
        sx = sx,
        LinLog_FG_Params = FG_Params,
        Tobs_yr = Tobs_yr,
        block_sun = block_sun)
    return LnL

# ---- Automate ----
ParamInfos = {
        'Astro' : 
            [{'name': 'fst', 'min': -2.5, 'max':0.0, 'latex':'\log_{10}f_{\star, 10}'},
            {'name': 'fesc', 'min': -2.5, 'max': 0.0, 'latex': '\log_{10}f_{\mathrm{esc}, 10}'},
            {'name': 'LX', 'min': 38.0, 'max': 42.0, 'latex': '\log_{10}L_{\mathrm{x}}'},
            {'name': 'mx', 'min': -4.0, 'max': 2, 'latex': '\log_{10}m_{\mathrm{x}} [{\mathrm{GeV}}]'},
            {'name': 'sx', 'min': 0, 'max': 100, 'latex': '\sigma_0 [10^x{\mathrm{cm^2}}]'},
            {'name': 'sx', 'min': -46, 'max': -39, 'latex': 'log10 \sigma_0 [10^x{\mathrm{cm^2}}]'},
            ],
        'FG':
            [{'name': 'FG1', 'min': 1000, 'max': 3000, 'latex': 'p_{\mathrm{FG,1}}'},
            {'name': 'FG2', 'min': -2000, 'max': 2000, 'latex': 'p_{\mathrm{FG,2}}'},
            {'name': 'FG3', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,3}}'},
            {'name': 'FG4', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,4}}'},
            {'name': 'FG5', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,5}}'},
            {'name': 'FG6', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,6}}'},
            {'name': 'FG7', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,7}}'},
            {'name': 'FG8', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,8}}'},
            {'name': 'FG9', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,9}}'},
            {'name': 'FG10', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,10}}'},
            {'name': 'FG11', 'min': -1000, 'max': 1000, 'latex': 'p_{\mathrm{FG,11}}'}]}

def Get_Param_Infos(
        NFG, 
        use_log10_sx, 
        fixed_log10_mx):
    infos = []
    # Astro
    if fixed_log10_mx is None:
        if use_log10_sx:
            astro_idx = [0, 1, 2, 3, 5]
        else:
            astro_idx = [0, 1, 2, 3, 4]
    else:
        if use_log10_sx:
            astro_idx = [0, 1, 2, 5]
        else:
            astro_idx = [0, 1, 2, 4]
    for idx in astro_idx:
        infos.append(ParamInfos['Astro'][idx])
    # FG
    for idx in np.arange(0, NFG):
        infos.append(ParamInfos['FG'][idx])
    return infos

def Set_FlatPrior(
        theta, 
        NFG, 
        use_log10_sx, 
        fixed_log10_mx):
    infos = Get_Param_Infos(NFG = NFG, use_log10_sx = use_log10_sx, fixed_log10_mx = fixed_log10_mx)
    r = PL.Set_Uniform_Multinest_Prior(theta=theta, infos=infos)
    return r

# ---- Automated Inference, can only use 1 CPU ----
def Run_Fit(
        NFG,
        Tobs,
        block_sun,
        use_log10_sx,
        sx_unit,
        fixed_log10_mx,
        root_info,
        sampling_efficiency = 0.8,
        n_live_per_dim = 50,
        overwrite_fileroot = None):
    from pymultinest.solve import solve
    # Prepare data path
    if not overwrite_fileroot is None:
        FileRoot = overwrite_fileroot
    else:
        data_path = chain_path + root_info[0] + '/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        FileRoot = data_path + root_info[1]
    infos = Get_Param_Infos(NFG=NFG, use_log10_sx=use_log10_sx, fixed_log10_mx=fixed_log10_mx)
    ndim = len(infos)
    if not fixed_log10_mx is None:
        PL.SaySomething(File=FileRoot+'sx_unit.txt', MSG = "{:.4E}".format(sx_unit))
    
    # likelihood and priors
    def LogLike(theta):
        LnL = Log_Like_kernel(
            theta = theta,
            Tobs_yr = Tobs,
            block_sun = block_sun,
            sx_unit = sx_unit,
            fixed_log10_mx = fixed_log10_mx, 
            use_log10_sx = use_log10_sx)
        return LnL
    
    def FlatPrior(theta):
        return Set_FlatPrior(theta, NFG, use_log10_sx, fixed_log10_mx)
    
    # run MultiNest
    Start_time = time.time()
    r = solve(
        LogLikelihood = LogLike,
        Prior = FlatPrior,
        n_dims = ndim,
        n_live_points = n_live_per_dim * ndim,
        resume = True,
        outputfiles_basename = FileRoot, 
        verbose = False,
        sampling_efficiency = sampling_efficiency,
        n_iter_before_update = 10)
    End_time = time.time()
    print("Run time: {:.1} s".format(End_time - Start_time))
    
    # Removing unneeded things
    PL.clean_multinest_cache(FileRoot=FileRoot)
    
    # Print run info to indicate that run is finished
    PL.print_mcmc_info(FileRoot, infos)
