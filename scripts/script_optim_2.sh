python optimise.py --fewshot --param_set 1 --backend cuda --n_run 4 --seed 0 --hm_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 4 --seed 1 --hm_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 4 --seed 2 --hm_loss
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 5 --seed 0 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 5 --seed 1 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 5 --seed 2 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 9 --seed 0 --emd_pr_loss --exp_dist
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 9 --seed 1 --emd_pr_loss --exp_dist
python optimise.py --fewshot --param_set 1 --backend cuda --n_run 9 --seed 2 --emd_pr_loss --exp_dist