reload = 1
params_fid = [-1.50, -1.20, 40.5, 0.5, -1.75, -2.25, 40.5, 0.0, -0.3, 0.2, 2.0, 0.0, -100.0]
#               fst  fesc    LX   ast  fst3   fesc3  LX3  ast3  aesc  E0   ALW  mx    sx
# pid            0    1      2     3     4      5     6    7     8    9    10   11    12
# scale         log  log    log   lin   log    log   log  lin   lin Lin_KeV   lin  log  log

setting_mac = {
    'nm': 8,
    'dx': [1E-3, 1E-2, 0.09],
    'dsx': [1E-3, 1E-2]
}

setting_HPC = {
    'nm': 15,
    'dx': [1E-4, 1E-3, 1E-2, 0.09], # With the new fiducial E0_KeV cannot go below 0.1
    'dsx': [1E-4, 1E-3, 1E-2, 0.1, 1.0, 10.0, 100.0]
    # 'dx': [1E-3, 1E-2, 1E-1, 0.4],
    # 'dsx': [1E-3, 1E-2, 1E-1, 1.0]
}

import PyLab as PL
# from joblib import Parallel, delayed
import numpy as np
import sys, platform, os, time, h5py, tqdm
from mpi4py import MPI
if platform.system() == 'Darwin':
    sys.path.insert(1, '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/ToolKit/')
    settings = setting_mac
    # settings = setting_HPC
    data_path = '/Volumes/X320/备份/SDM_HM_2512.05056/Fisher_DataSet_OPT/'
    data_file = '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/data/Fisher_OPT.h5'
else:
    sys.path.insert(1, '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/ToolKit/')
    settings = setting_HPC
    data_path = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/Fisher_DataSet_OPT/'
    data_file = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/Fisher_OPT.h5'
from DataSetTools import Run_21cmFirstCLASS_MINI
dim = len(params_fid)

def Find_filename(params):
    # fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx = params
    name = data_path + 'LC_'
    for p in params:
        name = name + "{:.7f}_".format(p)
    name = name + '.h5'
    return name

def Find_params_astro(pid=0):
    dx = settings['dx']
    pax = []
    for dx_ in dx:
        pax.append(params_fid[pid] + dx_)
        pax.append(params_fid[pid] - dx_)
    params_ax = []
    for p in pax:
        params = params_fid.copy()
        params[pid] = p
        params_ax.append(params)
    params_ax = np.array(params_ax)
    return params_ax

def Find_params_SDM():
    sx_unit = 1E-42
    dsx = np.array(settings['dsx'])
    nm = settings['nm']
    nsx = len(dsx)
    mx_ax = np.linspace(-4, 3, nm)
    sx_ax = sx_unit * dsx
    params_ax = np.zeros([int(nm*nsx), dim])
    idx = 0
    for mid in np.arange(0, nm):
        for sid in np.arange(0, nsx):
            mx = mx_ax[mid]
            sx = np.log10(sx_ax[sid])
            params_ax[idx, :] = params_fid.copy()
            params_ax[idx, -1] = sx
            params_ax[idx, -2] = mx
            idx += 1
    return params_ax

def Find_ALL_DataSet_params():
    # First determine the number of sampels we need to run
    param_astro_template = Find_params_astro(0)
    param_SDM_template =  Find_params_SDM()
    n_astro = len(param_astro_template)
    n_sdm = len(param_SDM_template)
    n = int(1 + n_astro * (dim - 2) + n_sdm)
    
    params_tot = np.zeros([n, dim])
    params_tot[0, :] = params_fid[:] # Fiducial
    count = 1
    for pid in np.arange(0, dim-2):
        params_ax = Find_params_astro(pid)
        size = len(params_ax)
        for idx in np.arange(0, size):
            params_tot[count, :] = params_ax[idx, :]
            count += 1
    # SDM
    params_ax = Find_params_SDM()
    size = len(params_ax)
    for idx in np.arange(0, size):
        params_tot[count, :] = params_ax[idx, :]
        count += 1
    return params_tot

def Find_run_params():
    params_tot = Find_ALL_DataSet_params()
    params_ax = []
    n = len(params_tot)
    for idx in np.arange(0, n):
        params = params_tot[idx, :]
        datafile = Find_filename(params)
        if not os.path.exists(datafile):
            params_ax.append(params)
    params_ax = np.array(params_ax)
    return params_ax

params_ax = Find_run_params()
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank == 0:
    for idx in np.arange(0, len(params_ax)):
        MSG = ' '
        for pid in np.arange(0, dim):
            p = params_ax[idx, pid]
            MSG += "  {:.4f}".format(p)
        print(MSG)
    print("Samples to run:", len(params_ax))

def Run(idx):
    params = params_ax[idx, :]
    datafile = Find_filename(params)
    fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0_KeV, ALW, mx, sx = params
    E0_eV = E0_KeV*1000.0
    if E0_KeV < 0.1000001:
        # We cannot really run E0_KeV < 0.1
        return 0
    Run_21cmFirstCLASS_MINI(
        F_STAR10 = fst,
        F_ESC10 = fesc,
        L_X = LX,
        ALPHA_STAR = ast,
        ALPHA_ESC = aesc,
        F_STAR7_MINI = fst3,
        L_X_MINI = LX3,
        NU_X_THRESH = E0_eV,
        F_ESC7_MINI = fesc3,
        ALPHA_STAR_MINI = ast3,
        A_LW = ALW,
        mdm = mx,
        sigma = sx,
        HII_DIM = 50,
        BOX_LEN = 200,
        datafile = datafile,
        save_lc = 0)
    return 0

def Read_Data(file):
    full_name = data_path + file
    with h5py.File(full_name, 'r') as h5:
        fst = h5['astro_params'].attrs['F_STAR10']
        fesc = h5['astro_params'].attrs['F_ESC10']
        LX = h5['astro_params'].attrs['L_X']
        ast = h5['astro_params'].attrs['ALPHA_STAR']
        fst3 = h5['astro_params'].attrs['F_STAR7_MINI']
        fesc3 = h5['astro_params'].attrs['F_ESC7_MINI']
        LX3 = h5['astro_params'].attrs['L_X_MINI']
        ast3 = h5['astro_params'].attrs['ALPHA_STAR_MINI']
        aesc = h5['astro_params'].attrs['ALPHA_ESC']
        E0 = h5['astro_params'].attrs['NU_X_THRESH']/1.0E3 # in KeV
        ALW = h5['astro_params'].attrs['A_LW']
        mx = h5['cosmo_params'].attrs['m_chi']
        sx = h5['cosmo_params'].attrs['sigma_SDM']
        z = h5['node_redshifts'][:]
        Tb = h5['global_quantities/brightness_temp'][:]
        # Tk = h5['global_quantities/Tk_box'][:]
        # xH = h5['global_quantities/xH_box'][:]
    
    fst = np.log10(fst)
    fesc = np.log10(fesc)
    try:
        LX = np.log10(LX)
    except:
        LX = np.float64(LX)
        LX = np.log10(LX)
    mx = mx - 9.0 # eV to GeV in log10
    sx = -sx
    fst3 = np.log10(fst3)
    fesc3 = np.log10(fesc3)
    LX3 = np.log10(LX3)

    param = [fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0, ALW, mx, sx]
    #names = ['fst', 'fesc', 'LX', 'ast', 'fst3', 'fesc3', 'LX3', 'ast3', 'aesc', 'E0', 'ALW', 'mx', 'sx']
    #for idx in np.arange(0, len(names)):
    #    print(names[idx], param[idx])
    mask = z < 50

    results = {
        'param': param,
        'Tb': Tb[mask],
        #'Tk': Tk,
        #'xH': xH,
        'z': z[mask]}
    # print(file)
    return results

# ---- Run dataset ----

if reload:
    params_len = len(params_ax)
    indexs = PL.distribute_MPI_Jobs(n_jobs = params_len, mpi_size = mpi_size, mpi_rank = mpi_rank)
    time.sleep(15.0)
    for idx in indexs:
        Run(idx)
else:
    if mpi_size > 1:
        raise Exception("Dont use MPI when running post-processing")
    files = os.listdir(data_path)

    nf = len(files)
    z = Read_Data(files[0])['z']
    nz = len(z)
    
    Tb = np.zeros([nf, nz])
    # Tk = np.zeros([nf, nz])
    # xH = np.zeros([nf, nz])

    params = np.zeros([nf, 13])

    for idx in tqdm.tqdm(range(nf), desc = 'doing cool stuff'):
        data = Read_Data(files[idx])
        Tb[idx, :] = data['Tb'][:]
        # Tk[idx, :] = data['Tk'][:]
        # xH[idx, :] = data['xH'][:]
        params[idx, :] = data['param'][:]

    params_fiducial = np.array(params_fid)

    with h5py.File(data_file, 'w') as h5:
        h5.create_dataset('z', data = z, dtype = 'float64')
        h5.create_dataset('params_fiducial', data = params_fiducial, dtype = 'float64')
        h5.create_dataset('params', data = params, dtype = 'float64')
        h5.create_dataset('Tb', data = Tb, dtype = 'float64')
        # h5.create_dataset('Tk', data = Tk, dtype = 'float64')
        # h5.create_dataset('xH', data = xH, dtype = 'float64')
