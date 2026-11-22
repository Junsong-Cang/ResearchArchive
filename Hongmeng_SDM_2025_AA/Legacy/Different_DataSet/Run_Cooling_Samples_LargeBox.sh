#!/bin/bash
#SBATCH --job-name=CoolFish
#SBATCH --time=96:00:00
#SBATCH --output=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Cooling_Samples_LargeBox.out
#SBATCH --error=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Cooling_Samples_LargeBox.err
#SBATCH --nodes=4
#SBATCH --mem=220GB
#SBATCH --ntasks-per-node=62
#SBATCH --cpus-per-task=1
#SBATCH --partition=ali

mpirun --map-by ppr:62:node /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs//soft/anaconda3/envs/SDM/bin/python3 /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/SDM_lite/DataSet/Run_Cooling_Samples_LargeBox.py
