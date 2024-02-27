python optimise.py --oneshot --param_set 1 --backend opengl --n_run 4 --seed 2 --hm_loss
python optimise.py --oneshot --param_set 1 --backend opengl --n_run 5 --seed 0 --pd_rs_loss --pd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend opengl --n_run 5 --seed 1 --pd_rs_loss --pd_sr_loss --exp_dist
python optimise.py --oneshot --param_set 1 --backend opengl --n_run 5 --seed 2 --pd_rs_loss --pd_sr_loss --exp_dist
python loss_landscape_mfgf.py --fewshot --backend opengl --exp_dist