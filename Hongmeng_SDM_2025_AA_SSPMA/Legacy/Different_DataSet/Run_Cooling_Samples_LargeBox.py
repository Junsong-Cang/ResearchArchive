import numpy as np

zmax = 50
summary_file = '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/data/cooling_rate_LargeBox.h5'
reload = 1
mx_ax = np.linspace(-4, 3, 40)
sx_ax = np.linspace(-46, -39, 40)

import PyLab as PL
import sys, platform, os, time, h5py
from mpi4py import MPI
if platform.system() == 'Darwin':
    sys.path.insert(1, '/Users/cangtao/FileVault/GitHub/ResearchArchive/Hongmeng_SDM_2025_AA_SSPMA/ToolKit/')
else:
    sys.path.insert(1, '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/ToolKit/')
from DataSetTools import Run_21cmFirstCLASS_MINI

def Find_filename(mx, sx):
    if platform.system() == 'Darwin':
        data_path = '/Volumes/X320/备份/SDM_HM_2512.05056/CoolingDataSet_LargeBox/'
    else:
        data_path = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/CoolingDataSet_LargeBox/'
    name = data_path + "LC_{:.7f}_{:.7f}.h5".format(mx, sx)
    return name

def Run_p21f(mx, sx):
    params_fid = [-1.25, -1.35, 40.5, 0.5, -2.5,  -1.35, 40.5, 0.0, -0.3, 500.0, 2.0]
    fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0_eV, ALW = params_fid
    datafile = Find_filename(mx, sx)
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
        HII_DIM = 100,
        BOX_LEN = 630,
        zmin = 9.5,
        datafile = datafile,
        save_lc = 0)

params = []
for mx in mx_ax:
    for sx in sx_ax:
        df = Find_filename(mx, sx)
        if not os.path.exists(df):
            params.append([mx, sx])
params = np.array(params)
param_len = len(params)
print('Samples to run:', param_len)

def Run(idx):
    mx, sx = params[idx, 0], params[idx, 1]
    Run_p21f(mx, sx)

if reload:
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    indexs = PL.distribute_MPI_Jobs(n_jobs = param_len, mpi_size = mpi_size, mpi_rank = mpi_rank)
    time.sleep(15.0)
    for idx in indexs:
        Run(idx)
    comm.Barrier()
else:
    raise Exception("This bit is not ready")
    with h5py.File(Find_filename(mx_ax[0], sx_ax[0]), 'r') as h5:
        z = h5['node_redshifts'][:]
    mask = z < zmax
    z = z[mask]
    nz = len(z)
    nm = len(mx_ax)
    ns = len(sx_ax)
    Tk = np.zeros([nm, ns, nz])
    Tb = np.zeros([nm, ns, nz])

    for mid in np.arange(0, nm):
        for sid in np.arange(0, ns):
            df = Find_filename(mx = mx_ax[mid], sx = sx_ax[sid])
            with h5py.File(df, 'r') as h5:
                Tk_ = h5['global_quantities/Tk_box'][:]
                Tb_ = h5['global_quantities/brightness_temp'][:]
                Tk[mid, sid, :] = Tk_[:][mask]
                Tb[mid, sid, :] = Tb_[:][mask]
    # add fiducial as well
    if platform.system() == 'Darwin':
        path = '/Volumes/X320/备份/SDM_HM_2512.05056/Fisher_DataSet/'
    else:
        path = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/data/Fisher_DataSet/'
    df = path + 'LC_-1.2500000_-1.3500000_40.5000000_0.5000000_-2.5000000_-1.3500000_40.5000000_0.0000000_-0.3000000_0.5000000_2.0000000_0.0000000_-100.0000000_.h5'
    with h5py.File(df, 'r') as h5:
        Tk_fid = h5['global_quantities/Tk_box'][:][mask]

    with h5py.File(summary_file, 'w') as h5f:
        h5f.create_dataset('z', data = z, dtype = 'float64')
        h5f.create_dataset('sx', data = sx_ax, dtype = 'float64')
        h5f.create_dataset('mx', data = mx_ax, dtype = 'float64')
        h5f.create_dataset('Tk', data = Tk, dtype = 'float64')
        h5f.create_dataset('Tk_fiducial', data = Tk_fid, dtype = 'float64')
        h5f.create_dataset('Tb', data = Tb, dtype = 'float64')
