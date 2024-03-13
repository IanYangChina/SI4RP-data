# python loss_landscape_mfgf.py --oneshot --backend cuda --param_set 1
# python loss_landscape_rhoys.py --oneshot --backend cuda --param_set 1
# python optimise.py --oneshot --n_run 12 --seed 0 --backend cuda --param_set 1 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035
# python optimise.py --oneshot --n_run 12 --seed 1 --backend cuda --param_set 1 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035
# python optimise.py --oneshot --n_run 12 --seed 2 --backend cuda --param_set 1 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035
python optimise.py --realoneshot --n_run 12 --seed 1  --backend cuda --param_set 1 --realoneshot_agent_id 1 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035 --device_mem 5
python optimise.py --realoneshot --n_run 12 --seed 2  --backend cuda --param_set 1 --realoneshot_agent_id 1 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035 --device_mem 5
python optimise.py --realoneshot --n_run 12 --seed 0  --backend cuda --param_set 1 --realoneshot_agent_id 2 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035 --device_mem 5
python optimise.py --realoneshot --n_run 12 --seed 1  --backend cuda --param_set 1 --realoneshot_agent_id 2 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035 --device_mem 5
python optimise.py --realoneshot --n_run 12 --seed 2  --backend cuda --param_set 1 --realoneshot_agent_id 2 --pd_rs_loss --pd_sr_loss --ptcl_d 6e7 --dsvs 0.0035 --device_mem 5
