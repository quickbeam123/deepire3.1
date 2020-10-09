#!/bin/bash

ulimit -Sn 10000

./multi_inf_parallel.py smt4vamp_avOff > run96_lr001_p9n1.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat > run48_lr001_p9n1_swout01.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat smt4vamp_defaultStrat/model-epoch40.pt > 2_run39dropout_lr001_p85n15_swout00.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat smt4vamp_defaultStrat/model0_40_Tanh_CatLayBIGGER_EvalLayNONLIN_LayerNormOFF_Dropout0.5.pt > run40dropout_lr0001_p85n15_swout00.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitp447 smt4vamp_aitp447/initial_59_Tanh_CatLayBIGGER_EvalLayNONLIN_LayerNormOFF.pt > run59_lr001_p85n15_swout02.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitp447 > run59_lr001_p85n15_swout00.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitp447 smt4vamp_aitp447/initial_60_Tanh_CatLayBIGGER_EvalLayNONLIN_LayerNormOFF.pt > run60_lr001_p85n15_swout01.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitp447 > run60_lr001_p85n15.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitpUnion > run57_lr001_p85n15.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat smt4vamp_defaultStrat/initial_56_Tanh_CatLaySMALL_EvalLayNONLIN_LayerNormOFF.pt > run56_lr001.txt 2>&1

# ./multi_inf_parallel.py enigma_tptp_culled/training_data.pt enigma_tptp_culled/model_55_Tanh.pt enigma_tptp_culled > run55tptp.txt 2>&1

# ./multi_inf_parallel.py enigma_smt_union/training_data.pt enigma_smt_union/model0_99_Tanh.pt enigma_smt_union > run99smt.txt 2>&1
