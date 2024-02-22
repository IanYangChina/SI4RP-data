python optimise.py --fewshot --param_set 0 --backend cuda --hm_loss --emd_pr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 0 --seed 0 --pd_rs_loss --pd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 0 --seed 1 --pd_rs_loss --pd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 0 --seed 2 --pd_rs_loss --pd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 1 --seed 0 --prd_rs_loss --prd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 1 --seed 1 --prd_rs_loss --prd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 1 --seed 2 --prd_rs_loss --prd_sr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 2 --seed 0 --emd_p_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 2 --seed 1 --emd_p_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 2 --seed 2 --emd_p_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 3 --seed 0 --emd_pr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 3 --seed 1 --emd_pr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 3 --seed 2 --emd_pr_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 4 --seed 0 --hm_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 4 --seed 1 --hm_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 4 --seed 2 --hm_loss
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 5 --seed 0 --pd_rs_loss --pd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 5 --seed 1 --pd_rs_loss --pd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 5 --seed 2 --pd_rs_loss --pd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 6 --seed 0 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 6 --seed 1 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 6 --seed 2 --prd_rs_loss --prd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 7 --seed 0 --emd_p_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 7 --seed 1 --emd_p_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend cuda --n_run 7 --seed 2 --emd_p_loss --exp_dist