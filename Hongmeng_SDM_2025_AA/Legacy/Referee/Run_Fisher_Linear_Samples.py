params_fid = [-1.25, -1.35, 40.5, 0.5, -2.5,  -1.35, 40.5, 0.0, -0.3, 0.5, 2.0, 0.0, -100.0]
#               fst  fesc    LX   ast  fst3   fesc3  LX3  ast3  aesc  E0   ALW  mx    sx
# pid            0    1      2     3     4      5     6    7     8    9    10   11    12
# scale         log  log    log   lin   log    log   log  lin   lin Lin_KeV   lin  log  log


import PyLab as PL
import numpy as np
import sys, time
from mpi4py import MPI
sys.path.insert(1, '/Users/cangtao/FileVault/GitHub/ResearchArchive_Private/HM_SDM25/ToolKit/')
import DataSetTools as DST

# sx_ax is the one-sigma error in sx at different masses
mx_ax = np.linspace(-4, 3, 15)
sx_ax = [-42.67, -42.67, -42.67, -42.67, -42.67, -42.67, -42.66, -42.64, -42.16, -41.85, -41.43, -41.08, -40.88, -40.79, -40.70]

def Run(idx):
    sigma = sx_ax[idx]
    mass = mx_ax[idx]
    params_astro = np.array([-1.25, -1.35, 40.5, 0.5, -2.5,  -1.35, 40.5, 0.0, -0.3, 0.5, 2.0])
    #                         fst  fesc    LX   ast  fst3   fesc3  LX3  ast3  aesc  E0   ALW  mx    sx
    datafile = "/Users/cangtao/Desktop/SDM_HM_cache/LC_{:.2f}_{:.2f}.h5".format(sigma, mass)
    fst, fesc, LX, ast, fst3, fesc3, LX3, ast3, aesc, E0_KeV, ALW = params_astro
    E0_eV = E0_KeV*1000.0
    DST.Run_21cmFirstCLASS_MINI(
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
        mdm = mass,
        sigma = sigma,
        HII_DIM = 50,
        BOX_LEN = 200,
        datafile = datafile,
        save_lc = 0)
    return 0

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
params_len = len(sx_ax)
indexs = PL.distribute_MPI_Jobs(n_jobs = params_len, mpi_size = mpi_size, mpi_rank = mpi_rank)
time.sleep(15.0)
for idx in indexs:
    Run(idx)
comm.Barrier()

# Run(1)
