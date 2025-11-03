#!/bin/bash
#SBATCH --job-name=DM21cm
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/BoostFactor/GetLightcones.out
#SBATCH --error=/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/BoostFactor/GetLightcones.err
#SBATCH --mem 200G
#SBATCH --partition=ali

/afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/soft/anaconda3/envs/p21c/bin/python3 /afs/ihep.ac.cn/users/z/zhangzixuan/work/cjs/BoostFactor/GetLightcones.py
