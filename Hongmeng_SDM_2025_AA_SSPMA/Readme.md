# ---- Codes for Hongmeng-SDM forecast ----

# Papers:
    Ref1: J. Cang, Y. Gao and Y-Z. Ma, Sensitivity of the Hongmeng 21cm experiment to scattering dark matter, Astron.Astrophys. 714 (2026), A40 [arXiv: 2512.05056]
    Ref2: B. Li, et al., Investigating the nature of dark matter and early-universe black hole formation based on the Hongmeng measurement of the global 21-cm spectrum, SCIENTIA SINICA Physica, Mechanica & Astronomica (2026)

# Pipeline
0. Install 21cmFirstCLASS
    url: https://github.com/Junsong-Cang/21cmFirstCLASS.git
    Any commits around 2025/11 should be fine.
1. Build Fisher DataSet
    1.1 - First run Run_Fisher_DataSet.py with MPI and reload=1
    1.2 - Run it again but with reload=0 and no MPI
    1.3 - scp Fisher.h5 produced by Run_Fisher_DataSet.py to mac
2. Run cooling/heating samples
    2.1 - Run Run_Cooling_Samples.py with MPI & reload
    2.2 Run it again without MPI & reload
    2.3 scp cooling_rate.h5 to mac
3. Final Steps: 
    main_AA.ipynb runs everything needed by the AA paper
    main_SSPMA.ipynb runs everything needed by the SSPMA paper

# Optional deappreciated pipeline - Park18 interpolation table
4. Build interpolation dataset
    4.1 Run Run_DataSet.py multiple times with MPI
    4.2 Before 4.1 finishes, keep running Find_Broken_Runs.py to look for broken runs
    4.3 When 4.1 finishes and 4.2 reports no broken runs, run Build_summaries.py
    4.4 Optionally you can also run Compress_things.sh to compress everything that can be scp'ed to mac, but this takes a lot of space
5. Build testset to check interpolation precision
    5.1 - Run_TestSets.py
    5.2 - scp TestSets.h5 to mac
6. Some inference codes are in 43_FG5.py, there are different versions but i am keeping only the last one
7. There are also a bunch of tests for interpolation precision etc, but I removed them after we decided to go with Fisher

# Trouble-shooting logs
Initially we tried to use MC rather than Fisher, with this we vary 3 astro params (fst, LX, fesc) in addition to mx and sigma. The idea for the pipeline is to use FG in my previous EDGES paper + Park18 astro and then:
1. Run 5D (mx-sigma) for multiple NFG, check that NFG=7 has the highest evidence;
2. Run multiple inferences with different masses, then get 95%C.I. upper limits on sigma;
3. Check that all above gives good truth recovery.
However we keep getting stucked in local maximum near mx=2.4 GeV, following are tried and failed:
    1. Use different fiducial astro
    2. Use different fiducial FG - NFG = 5
    3. Use different FG model (from Sims)
    4. Use different sampler (PolyChord)
    5. Use different sampling efficiency & live points
    6. Use different Tobs and Trcv
    7. Jointly sample mx and sigma
So in the end we decided to give up MC and use Fisher.
