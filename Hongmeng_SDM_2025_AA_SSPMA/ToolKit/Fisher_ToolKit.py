import numpy as np
import h5py, getdist
'''
Timestep precision notes
    fst    0.1
    fesc   0.1
    LX     0.1
    ast    0.1
    fst3   0.1
    fesc3  0.1
    LX3    0.1
    ast3   0.1
    aesc   0.1
    E0     0.1
    ALW    0.1
    ---- SDM ----
        mid      sx42
        0       0.01
        1       0.01
        2       0.01
        3       0.01
        4       0.01
        5       0.01
        6       0.01
        7       0.01
        8       0.01
        9       0.01
        10      0.01
        11      0.01
        12      0.01
        13      0.01
        14      0.01
    ---- FG ----
        FG0     0.01
        FG1     0.01
        FG2     0.01
        FG3     0.01
        FG4     0.01
'''

def Load_Fisher_DataSet():
    data_file = '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/data/Fisher.h5'
    #data_file = '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/data/Fisher_LargeBox.h5';print("Using Large box")
    with h5py.File(data_file, 'r') as h5:
        z = h5['z'][:]
        params_fid = h5['params_fiducial'][:]
        params_ax = h5['params'][:]
        Tb = h5['Tb'][:]
        
    result = {
        'z': z,
        'params_fid': params_fid,
        'params_ax': params_ax,
        'Tb': Tb,
    }
    return result

Fisher_DataSet = Load_Fisher_DataSet()

def check_inputs(param):
    param_fid = Fisher_DataSet['params_fid']
    count = 0
    for idx in np.arange(0, len(param_fid)):
        if not idx == 11: # u can vary sdm mass only when varying sx
            if np.abs(param[idx] - param_fid[idx]) > 1E-10:
                count = count + 1
    if count > 1:
        raise Exception("You cannot vary more than one params relative to fiducial!")
    
def Read_Fisher_DataSet(fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx42, type = 'Tb'):
    '''
    Entered in log10:
        fst, fesc, LX, mt, mx
    Entered in linear:
        ast, aesc, sx (1E-42)
    '''
    if sx42 < 1E-30:
        sx = -100.0
    else:
        sx = np.log10(sx42) - 42.0
    data_len = len(Fisher_DataSet['params_ax'])
    dist = np.zeros(data_len)
    param = np.array([fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx])
    check_inputs(param)
    for idx in np.arange(0, data_len):
        dist[idx] = np.sum((Fisher_DataSet['params_ax'][idx, :] - param)**2.0)**0.5
    
    # Find how many elements has dist > 1E-10, if more than one we have a trouble
    count = 0
    for d in dist:
        if d < 1E-10:
            count += 1
    if not count == 1:
        print('input = ', param)
        print('best match:', Fisher_DataSet['params_ax'][np.argmin(dist), :])
        raise Exception("Cannot find match, or more than one match was found!")
    
    idx_closest = np.argmin(dist)
    result = Fisher_DataSet[type][idx_closest, :]
    return result

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

def T_sky(fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx42, FG_params, freq):
    '''
    Find sky temp in mK
    '''
    # freq = np.arange(30, 120, 0.061)
    redshifts = 1420/freq - 1
    z_ax = Fisher_DataSet['z']
    T_ax = Read_Fisher_DataSet(fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx42)
    T_cosmic = np.interp(x = redshifts, xp = z_ax[::-1], fp = T_ax[::-1], left = np.nan, right = np.nan)
    T_FG = T_FG_LinLog(FG_params, freq)
    Tsky = T_cosmic + T_FG
    results = {
        'redshifts': redshifts,
        'Tsky': Tsky,
        'T_FG': T_FG,
        'T_cosmic': T_cosmic
    }
    return results

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
    T = T * 1000 # in mK
    return T

def Get_HM_Data(
        tobs_yr = 5,
        block_sun = 1):
    # -- Tsys --
    freq = np.arange(30, 120, 0.061)
    astro_params = {
        'fst': -1.25,
        'fesc': -1.35,
        'LX': 40.5,
        'ast': 0.5,
        'fst3': -2.5,
        'fesc3': -1.35,
        'LX3': 40.5,
        'ast3' : 0.0,
        'aesc': -0.3,
        'E0': 0.5,
        'ALW' : 2.0,
        'mx': 0,
        'sx42': 0.0
    }
    FG_params = [
        1697.8076928220983,
        0.0,
        0.0,
        0.0,
        0.0, # 1st version only goes to NFG=5
        0.0, # adding other FG terms
        0.0,
        0.0,
        0.0,
        0.0
        ]
    Tsky_stats = T_sky(
        fst = astro_params['fst'],
        fesc = astro_params['fesc'],
        LX = astro_params['LX'],
        ast = astro_params['ast'],
        fst3 = astro_params['fst3'],
        fesc3 = astro_params['fesc3'],
        LX3 = astro_params['LX3'],
        ast3 = astro_params['ast3'],
        aesc = astro_params['aesc'],
        E0 = astro_params['E0'],
        ALW = astro_params['ALW'],
        mx = astro_params['mx'],
        sx42 = astro_params['sx42'],
        FG_params = FG_params,
        freq = freq)
    Tsky = Tsky_stats['Tsky']
    T21 = Tsky_stats['T_cosmic']
    TFG = Tsky_stats['T_FG']
    
    Trcv = T_rcv(freq)
    Tsys = Tsky + Trcv
    
    # -- noise --
    eff = 0.1 if block_sun else 0.15
    dv = 0.4E6 # Hz
    seconds_in_year = 365.25 * 24 * 3600
    Tobs = tobs_yr * seconds_in_year
    Tobs_eff = eff * Tobs # in s
    sigma = Tsys / np.sqrt(dv * Tobs_eff)
    result = {
        'astro_params': astro_params,
        'FG_params': FG_params,
        'Tsky': Tsky,
        'sigma': sigma,
        'Trcv': Trcv,
        'T21': T21,
        'TFG': TFG,
        'freq': freq,
        'z': 1420/freq - 1
    }
    return result

def compute_derivative(
        pid,
        mx = 0.0,
        dx = 0.01, # For sx this is in unit of 1E-42 (in linear scale)
        NFG = 5,
        verbose = 0
        ):
    '''
    Find derivative dT_sky/ dp
    pid =      0    1      2     3     4      5      6     7      8     9       10    11      12     13    14    15    16
    params =  fst  fesc    LX   ast  fst3   fesc3   LX3   ast3   aesc   E0      ALW   sx42    FG0    FG1   FG2   FG3   FG4
    scale =   log  log    log   lin   log    log    log   lin    lin  Lin_KeV   lin   lin     lin    lin   lin   lin   lin
    '''
    mx_scaled = mx if pid == 11 else 0.0 # mx is only used when varying sx, if not varying sx fix mx to fiducial which is 0
    HM_data = Get_HM_Data() # Tobs and block_sun really doesn't matter here
    names = []
    
    # -- STEP 1: Get fiducial params --
    params_fid = []
    
    # astro
    astro_names = ['fst', 'fesc', 'LX', 'ast', 'fst3', 'fesc3', 'LX3', 'ast3', 'aesc', 'E0', 'ALW', 'sx42']
    for name in astro_names:
        params_fid.append(HM_data['astro_params'][name])
        names.append(name)
    
    # FG
    for idx in np.arange(0, NFG):
        params_fid.append(HM_data['FG_params'][idx])
        names.append("FG_{:.0f}".format(idx))
    
    params_fid = np.array(params_fid)
    ndim = len(params_fid)

    # All done with params_fid, print debug MSG to avoid screw-ups
    if verbose:
        for idx in np.arange(0, ndim):
            MSG = "{:.0f}   ".format(idx) + names[idx] + "    {:.2f}".format(params_fid[idx])
            print(MSG)
    
    # -- STEP 2: Get derivative
    def Get_Tsky_kernel(p):
        fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, sx42 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]
        n_astro = 12
        if not ndim == n_astro + NFG:
            MSG = "Wrong input dimension, detected dimension = {:.0f}, input dimension = {:.0f}".format(ndim, n_astro + NFG)
            raise Exception(MSG)
        FG_params = p[n_astro: ndim]
        T = T_sky(
                fst = fst,
                fesc = fesc,
                LX = LX,
                ast = ast,
                fst3 = fst3,
                fesc3 = fesc3,
                LX3 = LX3,
                ast3 = ast3,
                aesc = aesc,
                E0 = E0,
                ALW = ALW,
                sx42 = sx42,
                mx = mx_scaled,
                FG_params = FG_params,
                freq = HM_data['freq']
            )['Tsky']
        return T
    # Can only go towards positive with sx
    if pid == 11:
        params_Right = params_fid.copy()
        params_Right[pid] = params_fid[pid] + dx
        TR = Get_Tsky_kernel(params_Right)
        T0 = HM_data['Tsky']
        dTdp = (TR - T0)/dx
    else:
        # Use double-sided derative for all others
        params_Left = params_fid.copy()
        params_Right = params_fid.copy()
        params_Left[pid] = params_fid[pid] - dx
        params_Right[pid] = params_fid[pid] + dx
        TL = Get_Tsky_kernel(params_Left)
        TR = Get_Tsky_kernel(params_Right)
        dTdp = (TR - TL)/(2.0*dx)
    return dTdp

def Cut_Fish(
        Fish,
        params_to_exclude = None):
    '''
    -- outputs --
    Fish
    names
    Latex
    Truth
    '''
    def add_dollar_latex(name):
        return '$'+name+'$'
    
    infos_template = [
            {'name': 'fst', 'truth': -1.25, 'latex': '\log_{10}f_{\star, 10}'},                 # 0
            {'name': 'fesc', 'truth': -1.35, 'latex': '\log_{10}f_{\mathrm{esc}, 10}'},         # 1
            {'name': 'LX', 'truth': 40.5, 'latex': '\log_{10}\mathcal{L}_{\mathrm{x, II}}'},    # 2
            {'name': 'ast', 'truth': 0.5, 'latex': '\\alpha_{\star}'},                          # 3
            {'name': 'fst3', 'truth': -2.5, 'latex': '\log_{10}f_{\star,7}'},                   # 4
            {'name': 'fesc3', 'truth': -1.35, 'latex': '\log_{10}f_{{\mathrm{esc}},7}'},        # 5
            {'name': 'LX3', 'truth': 40.5, 'latex': '\log_{10}\mathcal{L}_{\mathrm{x, III}}'},  # 6
            {'name': 'ast3', 'truth': 0.0, 'latex': '\\alpha_{\star, {\\rm III}}'},             # 7
            {'name': 'aesc', 'truth': -0.3, 'latex': '\\alpha_{\mathrm{esc}}'},                 # 8
            {'name': 'E0', 'truth': 0.5, 'latex': 'E_0[{\mathrm{keV}}]'},                       # 9
            {'name': 'ALW', 'truth': 2.0, 'latex': 'A_{\mathrm{LW}}'},                          # 10
            {'name': 'sx42', 'truth': 0.0, 'latex':'\sigma_0 [10^{-42}{\mathrm{cm^2}}]'},       # 11  ----  SDM
            {'name': 'FG0', 'truth': 1697.8076928220983, 'latex':'p_{0}-p_{0,{\mathrm{fid}}}'}, # 12
            {'name': 'FG1', 'truth': 0.0, 'latex':'p_1'},                                       # 13
            {'name': 'FG2', 'truth': 0.0, 'latex':'p_2'},                                       # 14
            {'name': 'FG3', 'truth': 0.0, 'latex':'p_3'},                                       # 15
            {'name': 'FG4', 'truth': 0.0, 'latex':'p_4'},                                       # 16
            {'name': 'FG5', 'truth': 0.0, 'latex':'p_5'},                                       # 17
            {'name': 'FG6', 'truth': 0.0, 'latex':'p_6'},                                       # 18
            {'name': 'FG7', 'truth': 0.0, 'latex':'p_7'},                                       # 19
            {'name': 'FG8', 'truth': 0.0, 'latex':'p_8'},                                       # 20
            {'name': 'FG9', 'truth': 0.0, 'latex':'p_9'},                                       # 21
            {'name': 'FG10', 'truth': 0.0, 'latex':'p_{10}'}]                                   # 22
    
    # A more compact version
    infos_template = [
            {'name': 'fst', 'truth': -1.25, 'latex': 'f_{\star, 10}'},                 # 0
            {'name': 'fesc', 'truth': -1.35, 'latex': 'f_{\mathrm{esc}, 10}'},         # 1
            {'name': 'LX', 'truth': 40.5, 'latex': '\mathcal{L}_{\mathrm{x, II}}'},    # 2
            {'name': 'ast', 'truth': 0.5, 'latex': '\\alpha_{\star}'},                          # 3
            {'name': 'fst3', 'truth': -2.5, 'latex': 'f_{\star,7}'},                   # 4
            {'name': 'fesc3', 'truth': -1.35, 'latex': 'f_{{\mathrm{esc}},7}'},        # 5
            {'name': 'LX3', 'truth': 40.5, 'latex': '\mathcal{L}_{\mathrm{x, III}}'},  # 6
            {'name': 'ast3', 'truth': 0.0, 'latex': '\\alpha_{\star, {\\rm III}}'},             # 7
            {'name': 'aesc', 'truth': -0.3, 'latex': '\\alpha_{\mathrm{esc}}'},                 # 8
            {'name': 'E0', 'truth': 0.5, 'latex': 'E_0[{\mathrm{keV}}]'},                       # 9
            {'name': 'ALW', 'truth': 2.0, 'latex': 'A_{\mathrm{LW}}'},                          # 10
            {'name': 'sx42', 'truth': 0.0, 'latex':'\sigma_0'},       # 11  ----  SDM
            {'name': 'FG0', 'truth': 1697.8076928220983, 'latex':"p'_0"}, # 12
            {'name': 'FG1', 'truth': 0.0, 'latex':'p_1'},                                       # 13
            {'name': 'FG2', 'truth': 0.0, 'latex':'p_2'},                                       # 14
            {'name': 'FG3', 'truth': 0.0, 'latex':'p_3'},                                       # 15
            {'name': 'FG4', 'truth': 0.0, 'latex':'p_4'},                                       # 16
            {'name': 'FG5', 'truth': 0.0, 'latex':'p_5'},                                       # 17
            {'name': 'FG6', 'truth': 0.0, 'latex':'p_6'},                                       # 18
            {'name': 'FG7', 'truth': 0.0, 'latex':'p_7'},                                       # 19
            {'name': 'FG8', 'truth': 0.0, 'latex':'p_8'},                                       # 20
            {'name': 'FG9', 'truth': 0.0, 'latex':'p_9'},                                       # 21
            {'name': 'FG10', 'truth': 0.0, 'latex':'p_{10}'}]                                   # 22
    
    dim_0 = np.shape(Fish)[0]
    param_infos = infos_template[0: dim_0]
    names_tot = []
    latex_tot = []
    latex_dollar_tot = []
    truth_tot = []
    for idx in np.arange(0, dim_0):
        info_here = param_infos[idx]
        names_tot.append(info_here['name'])
        latex_tot.append(info_here['latex'])
        latex_dollar_tot.append(add_dollar_latex(info_here['latex']))
        truth_tot.append(info_here['truth'])
    if params_to_exclude is None:
        result = {
            'Fish_new': Fish,
            'names': names_tot,
            'latex': latex_tot,
            'latex_dollar': latex_dollar_tot,
            'truth': truth_tot
        }
        return result
    if not isinstance(params_to_exclude, list):
        MSG = "You must enter list for params_to_exclude. If you only want to exclude only one param then do params_to_exclude = [your_param]"
        raise Exception(MSG)
    
    # Check that param names makes sense
    for name in params_to_exclude:
        if not name in names_tot:
            raise Exception("Unknown param name")
    
    # Set up mask
    masks = []
    for name in names_tot:
        if name in params_to_exclude:
            masks.append(False)
        else:
            masks.append(True)
    
    # Get new stats
    dim_new = dim_0 - len(params_to_exclude)
    Fish_new = np.zeros([dim_new, dim_new])
    names_new = []
    latex_new = []
    latex_dollar_new = []
    truth_new = []
    count = 0
    for idx in np.arange(0, dim_0):
        mask_here = masks[idx]
        if mask_here:
            names_new.append(names_tot[idx])
            latex_new.append(latex_tot[idx])
            latex_dollar_new.append(latex_dollar_tot[idx])
            truth_new.append(truth_tot[idx])
            Fish_here_old = Fish[idx, :]
            Fish_here_new = Fish_here_old[masks]
            Fish_new[count, :] = Fish_here_new[:]
            count = count + 1
    result = {
        'Fish_new': Fish_new,
        'names': names_new,
        'latex': latex_new,
        'latex_dollar': latex_dollar_new,
        'truth': truth_new
    }
    return result

def check_symmetry(mat, raise_error):
    '''
    Numpy inv can induce numerical issues, can generally be solved
    '''
    n = np.shape(mat)[0]
    red_flag = 0
    for id1 in np.arange(0, n):
        for id2 in np.arange(0, n):
            f12 = mat[id1, id2]
            f21 = mat[id2, id1]
            dif = np.abs(f12 - f21)
            if dif > 1E-20:
                red_flag = 1
    if raise_error and red_flag:
        raise Exception("Your matrx is not symmetric")
    return red_flag

def Compute_Fisher_Matrix(
        mx = 0.0,
        dx_ax = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        NFG = 5,
        Tobs = 5,
        block_sun = 1,
        params_to_exclude = None,
        zmax = 200.0,
        zmin = 0,
        n_samples = 50000,
        get_samples = 0,
        Irreducible_noise_mK = 0.0,
        ):
    '''
    -- outputs --
    getdist_samples
    nD Gaussian samples
    latex_dollar
    '''
    HM_data = Get_HM_Data(tobs_yr = Tobs, block_sun = block_sun)
    sigma = np.sqrt(HM_data['sigma']**2.0 + Irreducible_noise_mK**2.0) 
    if NFG < 1:
        MSG = "Best do this through params_to_exclude"
        raise Exception(MSG)

    # mask = HM_data['z'] < zmax
    mask = []
    for redshift in HM_data['z']:
        if (redshift > zmin) and (redshift < zmax):
            mask.append(True)
        else:
            mask.append(False)
    
    nf = len(sigma)
    n_astro = 12
    ndim = n_astro + NFG
    if not len(dx_ax) == ndim:
        raise Exception("Dimension mismatch between dx_ax and dim")
    dTdp_ax = np.zeros([ndim, nf])
    for pid in np.arange(0, ndim):
        dTdp = compute_derivative(
            pid = pid,
            mx = mx,
            dx = dx_ax[pid],
            NFG = NFG,
            verbose = 0)
        dTdp_ax[pid, :] = dTdp[:]
    
    Fish = np.zeros([ndim, ndim])
    for id1 in np.arange(0, ndim):
        for id2 in np.arange(0, ndim):
            d1 = dTdp_ax[id1, :]
            d2 = dTdp_ax[id2, :]
            Fish_strip = d1*d2/sigma**2.0
            Fish_strip = Fish_strip[mask]
            Fish[id1, id2] = np.sum(Fish_strip)
    
    # Want to cut the fish?
    FishPlate = Cut_Fish(Fish=Fish, params_to_exclude = params_to_exclude)
    Fish = FishPlate['Fish_new']
    param_names = FishPlate['names']
    param_latex = FishPlate['latex']
    param_latex_dollar = FishPlate['latex_dollar']
    truth = FishPlate['truth']
    if 'sx42' in param_names:
        sx_idx = param_names.index('sx42')
    else:
        sx_idx = None
    ndim = np.shape(Fish)[0]
    
    CovMat = np.linalg.inv(Fish) # Covariance Matrix

    # Ensure that the matrix is symmetric, numerical issues can give non-symmetric CovMat, e.g., this happended for mx = -2, Tob = 1. purely numerical issue, nothing to worry about
    check_symmetry(Fish, 1) # Fish should be symmetric, if not raise error
    red_flag = check_symmetry(CovMat, 0)
    if red_flag:
        CovMat = (CovMat + CovMat.T)/2.0
    params_sigma = np.zeros(ndim)
    for idx in np.arange(0, ndim):
        params_sigma[idx] = CovMat[idx, idx]**0.5
    if 'sx42' in param_names:
        sx42_limit = 2.0*params_sigma[sx_idx] # 95% C.I.for sigma_0 in 1E-42cm^2
    else:
        sx42_limit = None

    result = {
        'Fish': Fish,
        'CovMat': CovMat,
        'params_sigma': params_sigma,
        'sx42_limit': sx42_limit,
        'names': param_names,
        'sx_idx': sx_idx,
        'latex': param_latex,
        'latex_dollar': param_latex_dollar, # corner and matplotlib needs dollar sign
        'truth': truth,
    }
    if get_samples:
        # scaling FG0 otherwise it would be impossible to plot it!
        samples = np.random.multivariate_normal(truth, CovMat, size = n_samples)
        # FG0 needs to be scaled, so param here is in fact FG0 - FG0_fid
        FG0_idx = param_names.index('FG0')
        samples[:, FG0_idx] = samples[:, FG0_idx] - truth[FG0_idx]
        result['truth'][FG0_idx] = 0
        getdist_samples = getdist.mcsamples.MCSamples(samples = samples, names = param_names, labels = param_latex)
        result['samples'] = samples
        result['getdist_samples'] = getdist_samples
    return result
