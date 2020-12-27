#!/bin/bash

ulimit -Sn 1000

./multi_inf_parallel_files_continuous.py mizar_gsd/thax2000/ mizar_gsd/thax2000/runX/ 2>&1

# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax500/ mizar_strat1nacc/thax500/run4c/ mizar_strat1nacc/thax500/run4b/check-epoch395.pt 500 2>&1

# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax500/ mizar_strat1nacc/thax500/run4b/ mizar_strat1nacc/thax500/run4/check-epoch185.pt 2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax500/ mizar_strat1nacc/thax500/run4/ 2>&1

# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax2000/ mizar_strat1nacc/thax2000/run3b/ mizar_strat1nacc/thax2000/run3/check-epoch116.pt 2>&1

# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run2d/ mizar_strat1nacc/thax1000/run2c/check-epoch433.pt 2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run2c/ mizar_strat1nacc/thax1000/run2b/check-epoch408.pt 2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run2/ 2>&1

# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run1e/ mizar_strat1nacc/thax1000/run1d/check-epoch199.pt  2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run1d/ mizar_strat1nacc/thax1000/run1c/check-epoch153.pt  2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run1c/ mizar_strat1nacc/thax1000/run1b/check-epoch100.pt  2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run1b/ mizar_strat1nacc/thax1000/run1/check-epoch50.pt  2>&1
# ./multi_inf_parallel_files_continuous.py mizar_strat1nacc/thax1000/ mizar_strat1nacc/thax1000/run1/ 2>&1

# ./multi_inf_parallel_files.py mizar_strat1/ mizar_strat1/run2e mizar_strat1/run2d/check-epoch910.pt 2>&1 &
# ./multi_inf_parallel_files.py mizar_strat2/ mizar_strat2/run2e mizar_strat2/run2d/check-epoch772.pt 2>&1

# ./multi_inf_parallel_files.py mizar_strat2/ mizar_strat2/run2d mizar_strat2/run2c/check-epoch500.pt 2>&1

# ./multi_inf_parallel_files.py mizar_strat3/ mizar_strat3/run2b mizar_strat3/run2/check-epoch500.pt 2>&1

# ./multi_inf_parallel_files.py mizar_strat3/ mizar_strat3/run2 2>&1

# vzdy kdyz poustis znovu mizar_strat4 runs (run4x a run5), tak chces POS_WEIGHT_EXTRA = 3.0, at se to nemeni!
# ./multi_inf_parallel_files.py mizar_strat4/loop2_recalibrated/ mizar_strat4/run5 2>&1

# ./multi_inf_parallel_files.py mizar_strat3/ mizar_strat3/run1b mizar_strat3/run1/check-epoch193.pt 2>&1

# ./multi_inf_parallel_files.py mizar_strat2/ mizar_strat2/run1 2>&1

# ./multi_inf_parallel_files.py mizar_strat1/ mizar_strat1/run2_withSine 2>&1

# ./multi_inf_parallel_files.py mizar_strat1/ mizar_strat1/run3_temp 2>&1

# ./multi_inf_parallel.py smt4vamp_avOff/abstractOnlyBal smt4vamp_avOff/abstractOnlyBal/run128_p2.0_overfit_deep 2>&1

# ./multi_inf_parallel.py mizar_strat1/ mizar_strat1/frun128_p5n1_do 2>&1

# ./multi_inf_parallel.py smt4vamp_defaultStrat smt4vamp_defaultStrat/erun256_p3n1_do_overfit 2>&1

# ./multi_inf_parallel.py smt4vamp_avOff > run144_lr001_p9n1_swout00.txt 2>&1

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
