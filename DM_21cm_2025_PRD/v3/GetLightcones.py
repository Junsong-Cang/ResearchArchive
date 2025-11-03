reload = 0
dev_mode = 1
nk = 40
PS_File = 'PowerSpectrums.h5'

params = {'Pann27': [0, 1.0, 1.0],
  'INHOMO_HALO_BOOST': [False, False, True],
  'FileName': ['Fiducial', 'HMG', 'IHM']}

HII_DIM = 300
HII_DIM_mac = 200
redshift = 11.0
DM_Channel = 1
mdm = 0.1
Z_HEAT_MAX = 60
GLB_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box', 'Boost_box', 'density')
if dev_mode:
  data_path_mac = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_v3_tmp_HII_200/'
else:
  data_path_mac = '/Users/cangtao/Desktop/21cmFAST-data/BoostFactor_v3_HiRes/'
data_path_HPC = '/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/BoostFactor/data/'

import py21cmfast as p21c
import platform, time, os, h5py, shutil
import cosmo_tools as cosmo
import PyLab as PL
import numpy as np
from joblib import Parallel, delayed

if platform.system() == 'Darwin':
  HII_DIM = HII_DIM_mac
  data_path = data_path_mac
else:
  data_path = data_path_HPC
    
def RunP21c(sample_idx):
  if sample_idx == 2:
    LC_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box','Boost_box', 'density')
  else:
    LC_Quantities = ('brightness_temp','Ts_box','xH_box','Tk_box')
  write = True
  cache_loc = data_path + params['FileName'][sample_idx]+'/'
  if os.path.exists(cache_loc): shutil.rmtree(cache_loc)
  os.makedirs(cache_loc)
    
  INHOMO_HALO_BOOST = params['INHOMO_HALO_BOOST'][sample_idx]
  FileName = data_path + params['FileName'][sample_idx]+'.h5'
  Pann27 = params['Pann27'][sample_idx]
  
  user_params = p21c.UserParams(
    HII_DIM = HII_DIM,
    N_THREADS = 1,
    USE_RELATIVE_VELOCITIES = False,
    USE_INTERPOLATION_TABLES = True,
    FAST_FCOLL_TABLES = False,
    HMF = 1,
    POWER_SPECTRUM = 2,
    DM_Dep_Method = 1,
    DM_ANN_Channel = DM_Channel,
    BOX_LEN = 500)
  
  astro_params = p21c.AstroParams(
    Pann27 = Pann27,
    mdm = mdm,
    L_X = 40.5,
    F_STAR10 = -1.3,
    ALPHA_STAR = 0.5,
    F_ESC10 = -1.0,
    ALPHA_ESC = -0.5,
    M_TURN = 8.7,
    t_STAR = 0.5,
    NU_X_THRESH = 500.0)
  
  flag_options = p21c.FlagOptions(
    USE_MINI_HALOS = False,
    USE_MASS_DEPENDENT_ZETA = True,
    INHOMO_RECO = True,
    USE_TS_FLUCT = True,
    USE_HALO_BOOST = True,
    PHOTON_CONS = True,
    INHOMO_HALO_BOOST = INHOMO_HALO_BOOST)

  start_time = time.time()
  time.sleep(sample_idx * 5.0) # HyRec is not mpi-compatible
  InitialCondition = PL.HyRec(Pann = Pann27*1E-27, Use_SSCK = 0, mdm = mdm, DM_Channel=DM_Channel)
  z = InitialCondition['z'][::-1]
  xe = InitialCondition['xe'][::-1]
  Tk = InitialCondition['Tk'][::-1]

  XION_at_Z_HEAT_MAX = np.interp(x = Z_HEAT_MAX, xp = z, fp = xe)
  XION_at_Z_HEAT_MAX = XION_at_Z_HEAT_MAX/(1+0.08112582781456953) # Convert to p21c format assuming shared xe, 0.08 is fHe
  TK_at_Z_HEAT_MAX = np.interp(x = Z_HEAT_MAX, xp = z, fp = Tk)
  with p21c.global_params.use(Z_HEAT_MAX = Z_HEAT_MAX, XION_at_Z_HEAT_MAX = XION_at_Z_HEAT_MAX, TK_at_Z_HEAT_MAX = TK_at_Z_HEAT_MAX):
    lc = p21c.run_lightcone(
      redshift=redshift, 
      max_redshift=Z_HEAT_MAX,
      astro_params=astro_params, 
      flag_options=flag_options,
      user_params = user_params,
      lightcone_quantities=LC_Quantities,
      global_quantities=GLB_Quantities,
      random_seed = 42,
      write = write,
      direc = cache_loc)
  if os.path.exists(FileName):os.remove(FileName)
  lc.save(FileName)
  end_time = time.time()
  print("Run time: {:.2f}".format(end_time - start_time))

if reload:
  if platform.system() == 'Darwin':
    if HII_DIM > 201:
      for idx in [0, 1, 2]: RunP21c(idx)
    else:
      swap = Parallel(n_jobs = 3)(delayed(RunP21c)(idx) for idx in [0, 1, 2])
  else:
    swap = Parallel(n_jobs = 3)(delayed(RunP21c)(idx) for idx in [0, 1, 2])
  
# Now get all power spectrums for everything but radio temp (which we don't have)
# if platform.system() == 'Darwin':raise Exception('About to start PS, can NOT be run on mac')

FieldNames = ['Tb', 'Tk', 'xH', 'xe', 'Tr', 'Boost', 'density']

# f = h5py.File('/Users/cangtao/FileVault/LaTex/BoostFactor/Codes/v3/data/' + PS_File, 'w')
f = h5py.File(PS_File, 'w')
def Get_PS_Summaries(SimIdx, field):
  path = data_path + params['FileName'][SimIdx] + '/'
  SimName = params['FileName'][SimIdx]
  FieldName = FieldNames[field]
  dataset_prefix = SimName+'/'+FieldName
  print('Status :', dataset_prefix)
  data_name_z = dataset_prefix+'/z'
  data_name_k = dataset_prefix+'/k'
  data_name_ps = dataset_prefix+'/ps'
  r = cosmo.Get_P21c_Coeval_cache_PS(path=path, cleanup=0, nk=nk, output_file='tmp.npz', field=field, show_status=1)
  z = r['z']
  k = r['k']
  ps = r['ps']
  f.create_dataset(data_name_z, data = z)
  f.create_dataset(data_name_k, data = k)
  f.create_dataset(data_name_ps, data = ps)

'''
def SaveRedshifts():
  FileName = data_path + params['FileName'][2]+'.h5'
  lc = p21c.LightCone.read(FileName)
  z_glb = lc.node_redshifts
  z_lc = lc.lightcone_redshifts
  f.create_dataset('global/z', data = z_glb)
  f.create_dataset('lc2D/z', data = z_lc)

def Save_Global_Tb(idx):
  FileName = data_path + params['FileName'][idx]+'.h5'
  lc = p21c.LightCone.read(FileName)
  Tb = lc.global_brightness_temp
  f.create_dataset('global/Tb_'+params['FileName'][idx], data = Tb)

def Save_Global_B():
  FileName = data_path + params['FileName'][2]+'.h5'
  lc = p21c.LightCone.read(FileName)
  Boost = lc.global_Boost
  f.create_dataset('global/B', data = Boost)

def Save_2D_Slices(idx):
  FileName = data_path + params['FileName'][idx]+'.h5'
  lc = p21c.LightCone.read(FileName)
  Tb = lc.global_brightness_temp[1,:,:]
  Tk = lc.Tk_box[1,:,:]
  xe = 1-lc.xH_box[1,:,:]
'''

# Runing post-processing
for SimIdx in [0, 1]: Get_PS_Summaries(SimIdx, 0)
for field in [0, 5, 6]: Get_PS_Summaries(2, field)

f.close()
# os.system('touch /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/BoostFactor/finished_status.txt')

# Dev: non - PS stats
'''
Fiducial:  
  Global Tb
HMG: 
  LC slices for: Tk, xe, Tb
IHM: 
  LC slices for: Tk, xe, Tb, density, Boost
  Global Tb, B
'''
