python optimise.py --fewshot --param_set 0 --backend cuda --prd_sr_loss --prd_rs_loss --emd_pr_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 0 --seed 2 --emd_pr_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 1 --seed 1 --emd_p_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 1 --seed 2 --emd_p_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 2 --seed 2 --pd_rs_loss --pd_sr_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 3 --seed 0 --prd_rs_loss --prd_sr_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 3 --seed 1 --prd_rs_loss --prd_sr_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 3 --seed 2 --prd_rs_loss --prd_sr_loss