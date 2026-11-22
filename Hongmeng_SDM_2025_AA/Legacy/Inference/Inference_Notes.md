# 本来要做Inference的，但是1GeV附近总是有无法消除的bump。以下记录了一些之前做过的测试，最后还是用Fisher了。Inference的数据和代码都不用保留了，记录一下问题就行了

HM issue

1. 砍掉 m > 2 GeV
2. 用joint constraints - 有可能有island，并且不一定能扫到小质量
3. 完美FG
4. 简化FG-高阶0，NFG5 - 29测试一下不同精度，能行的话那就用这个
5. 简化FG分别扫不同质量 - # 20在干这个好像有戏但是需要跑完后再分析一下，有必要的话更新FG&noise重新跑

# ----需要做的----
2. 重新跑Model Selection, log sx
    # 31_Model_Selection
3. 重新跑几个变化m的线性sx，有啥算啥
    # LowRes: 32_T5_vary_mx, 33_T1_vary_mx, 34_T3_vary_mx
    # HiRes: 35_T5_HiRes, 36_T1_HiRes, 37_T3_HiRes

# --更换FG FID---

5. 跑几个完美FG
6. 继续跑27_T5_S1_FG5，如果有问题就增加精度，可以的话就用这个

Do a HiRes run of my FG7 model and see whether my previous runs were stuck in local minimum
Do some HiRes runs for FG5

#20好像有戏，需要换FG Fiducial并且只用NFG5

# ---- Failed Runs ----
5 换个非常高的livepoint跑FID 2&3
    好像没用
6 prior上限设置为1E-42，避免bogus
    有用但是不物理，并且贝叶斯证据降低了，算了吧
7 换一些质量重新跑
    结果很奇怪，算了
8 换一组astro fiducial
    没跑完算了
9&10 减小sampling efficiency，增加live points
    没用
11 - 类似于9&10换个FID
    没用
12 - 换PolyChord
    没用还超级慢
13 - 4D测试一下PolyChord，没用
14 - 4D试试增加mc精度和实验精度
    有用
15 - 在14上+FG
    不行
16 - 在增加实验和mc精度
    没跑完，初步看来不行
17 - 换Sims的FG
    不行，增加精度可能性但难说
18 - SimsFG, 增加mc和实验精度
    没跑完，初步看来不行
19 - 换个FG fiducial用不同的NFG跑，固定质量
    默认精度不行，但是20可能可以！！！
26 - 换个FG Fiducial，默认精度，五年，联合拟合mx, NFG0
    不太行，低质量那边完全没有posterior
28 - 换个FG Fiducial，默认精度，五年，联合拟合mx, NFG3
    不太行，有孤立岛
29_T5_S1_FG5_mx8 - 固定mx，dif FG FID, 中等精度
    不行
27_T5_S1_FG5 - FG5联合拟合mx
    不行，有岛
38_T5_S0_FG5_mx8_HiRes：更换FG，相当于超高精度的20_HM_FG5
    不行，岛更明显了
39_T5_S0_vary_m
    类似于高精度版的27_T5_S1_FG5
    不行，有岛
30_Run_DM_Limits
    不行，重建全跑偏了
    
# ----正在跑的----
21 - 24: 换了Tsky model，用不同的观测时间（1&5yr）和sun block，联合拟合mx
27_T5_S1_FG5: 默认mc精度，dif FG FID, 联合mx跑

# ----最新版本----
30-37都是用EDGES FG FID, updated Trcv, Tobs
