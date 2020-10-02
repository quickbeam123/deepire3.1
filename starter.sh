#!/bin/bash

ulimit -Sn 10000

./multi_inf_parallel.py smt4vamp_aitp447 > run60_lr001_p85n15.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_aitpUnion > run57_lr001_p85n15.txt 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat smt4vamp_defaultStrat/initial_56_Tanh_CatLaySMALL_EvalLayNONLIN_LayerNormOFF.pt > run56_lr001.txt 2>&1

# ./multi_inf_parallel.py enigma_tptp_culled/training_data.pt enigma_tptp_culled/model_55_Tanh.pt enigma_tptp_culled > run55tptp.txt 2>&1

# ./multi_inf_parallel.py enigma_smt_union/training_data.pt enigma_smt_union/model0_99_Tanh.pt enigma_smt_union > run99smt.txt 2>&1
