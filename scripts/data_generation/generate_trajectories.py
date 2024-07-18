import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
tr_path = os.path.join(script_path, '..', 'trajectories')

motion = 'validation'
agent = '4'
dim = 2

for dt in ['0.01', '0.02', '0.03', '0.04', 'avg']:
    tr = np.load(os.path.join(tr_path, f'tr_{agent}_v_dt_{dt}.npy'))
    plt.plot(tr[:, dim], label=f'dt: {dt} T: {tr.shape[0]}')
tr_ = np.load(os.path.join(script_path, '..', f'data-motion-{agent}', 'tr_eef_v.npy'))
plt.plot(tr_[:, dim], label=f'old T: {tr_.shape[0]}')
plt.legend()
plt.show()
exit()

tr = np.load(os.path.join(tr_path, f'tr_{agent}_eef_v_moveit.npy'))
timestamps = np.load(os.path.join(tr_path, f'tr_{agent}_timestamps_moveit.npy'))
tds = np.diff(timestamps)

px = [0.0]
pz = [0.0]

for n in range(tr.shape[0] - 1):
    px.append(px[-1] + tr[n][0] * tds[n])
    pz.append(pz[-1] + tr[n][2] * tds[n])
    print(n, px[-1], pz[-1])

inds = [0, 10, 25]
for i in inds:
    print(i, timestamps[i], px[i+1], pz[i+1])
print(tr.shape[0] - 1, timestamps[-1], px[-1], pz[-1])
plt.plot(px, label='px')
plt.plot(pz, label='pz')
# plt.legend()
# plt.show()
# exit()

dt = timestamps[-1] / tr.shape[0]
print(f'dt: {dt}, dt_sub:{dt / 50}')
inds.append(-1)
new_tr = np.zeros(shape=(600, 6), dtype=np.float32)

n_0 = np.ceil(timestamps[inds[1]] / dt).astype(np.int32)
new_tr[:n_0, 2] = -0.02 / (dt * n_0)
n_1 = np.ceil((timestamps[inds[2]] - timestamps[inds[1]]) / dt).astype(np.int32)
new_tr[n_0:n_0+n_1, 0] = 0.03 / (dt * n_1)
n_2 = np.ceil((timestamps[inds[3]] - timestamps[inds[2]]) / dt).astype(np.int32)
new_tr[n_0+n_1:n_0+n_1+n_2, 2] = 0.03 / (dt * n_2)

new_tr = new_tr[:n_0+n_1+n_2, :]
print(new_tr.shape[0])

px_new = [0.0]
pz_new = [0.0]
for n in range(new_tr.shape[0]):
    px_new.append(px_new[-1] + new_tr[n][0] * dt)
    pz_new.append(pz_new[-1] + new_tr[n][2] * dt)

plt.plot(px_new, label='px_new')
plt.plot(pz_new, label='pz_new')
plt.legend()
plt.show()

plt.plot(tr, label='tr')
plt.plot(new_tr, label='new_tr')
plt.legend()
plt.show()

np.save(os.path.join(tr_path, f'tr_{agent}_v_dt_avg.npy'), new_tr)
np.save(os.path.join(tr_path, f'tr_{agent}_dt_avg.npy'), dt)
