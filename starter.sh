#!/bin/bash

ulimit -Sn 10000
./multi_inf_parallel.py smt4vamp_defaultStrat/training_data.pt smt4vamp_defaultStrat/validation_data.pt smt4vamp_defaultStrat/model0_55_Tanh.pt > run55.txt 2>&1

# ./multi_inf_parallel.py enigma_tptp_culled/training_data.pt enigma_tptp_culled/model_55_Tanh.pt enigma_tptp_culled > run55tptp.txt 2>&1

# ./multi_inf_parallel.py enigma_smt_union/training_data.pt enigma_smt_union/model0_99_Tanh.pt enigma_smt_union > run99smt.txt 2>&1
