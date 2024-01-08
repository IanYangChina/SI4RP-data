import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))

motion = 'validation'
agent = 'cyldr'
tr = np.load(os.path.join(script_path, '..', f'data-motion-{motion}', f'tr_{agent}_valid_eef_v_1.npy'))
timestamps = np.load(os.path.join(script_path, '..', f'data-motion-{motion}', f'tr_{agent}_valid_timestamps_1.npy'))
tds = np.diff(timestamps)

px = [0.0]
pz = [0.0]

for n in range(tr.shape[0] - 1):
    px.append(px[-1] + tr[n][0] * tds[n])
    pz.append(pz[-1] + tr[n][2] * tds[n])
    print(n, px[-1], pz[-1])
print(0, timestamps[0], px[0], pz[0])
print(26, timestamps[26], px[26], pz[26])
print(51, timestamps[51], px[51], pz[51])
print(100, timestamps[100], px[100], pz[100])
print(149, timestamps[149], px[149], pz[149])
print(175, timestamps[175], px[175], pz[175])
print(226, timestamps[226], px[226], pz[226])
print(tr.shape[0] - 1, timestamps[-1], px[-1], pz[-1])
plt.plot(px, label='px')
plt.plot(pz, label='pz')

dt = timestamps[26] / 26
new_tr = np.zeros(shape=(600, 6), dtype=np.float32)

print(f'dt: {dt}')
n_0 = np.ceil(timestamps[26] / dt).astype(np.int32)
new_tr[:n_0, 2] = -0.025 / (dt * n_0)
n_1 = np.ceil((timestamps[51] - timestamps[26]) / dt).astype(np.int32)
new_tr[n_0:n_0+n_1, 0] = 0.025 / (dt * n_1)
n_2 = np.ceil((timestamps[100] - timestamps[51]) / dt).astype(np.int32)
new_tr[n_0+n_1:n_0+n_1+n_2, 2] = 0.025 / (dt * n_2)
n_3 = np.ceil((timestamps[149] - timestamps[100]) / dt).astype(np.int32)
new_tr[n_0+n_1+n_2:n_0+n_1+n_2+n_3, 0] = -0.025 / (dt * n_3)
n_4 = np.ceil((timestamps[175] - timestamps[149]) / dt).astype(np.int32)
new_tr[n_0+n_1+n_2+n_3:n_0+n_1+n_2+n_3+n_4, 2] = -0.025 / (dt * n_4)
n_5 = np.ceil((timestamps[226] - timestamps[175]) / dt).astype(np.int32)
new_tr[n_0+n_1+n_2+n_3+n_4:n_0+n_1+n_2+n_3+n_4+n_5, 0] = -0.025 / (dt * n_5)
n_6 = np.ceil((timestamps[-1] - timestamps[226]) / dt).astype(np.int32)
new_tr[n_0+n_1+n_2+n_3+n_4+n_5:n_0+n_1+n_2+n_3+n_4+n_5+n_6, 2] = 0.025 / (dt * n_6)

new_tr = new_tr[:n_0+n_1+n_2+n_3+n_4+n_5+n_6, :]

px_new = [0.0]
pz_new = [0.0]
for n in range(new_tr.shape[0]):
    px_new.append(px_new[-1] + new_tr[n][0] * dt)
    pz_new.append(pz_new[-1] + new_tr[n][2] * dt)

print(0, 0.0, px_new[0], pz_new[0])
print(26, 26*dt, px_new[26], pz_new[26])
print(n_0+n_1, (n_0+n_1)*dt, px_new[n_0+n_1], pz_new[n_0+n_1])
print(n_0+n_1+n_2, (n_0+n_1+n_2)*dt, px_new[n_0+n_1+n_2], pz_new[n_0+n_1+n_2])
print(n_0+n_1+n_2+n_3, (n_0+n_1+n_2+n_3)*dt, px_new[n_0+n_1+n_2+n_3], pz_new[n_0+n_1+n_2+n_3])
print(n_0+n_1+n_2+n_3+n_4, (n_0+n_1+n_2+n_3+n_4)*dt, px_new[n_0+n_1+n_2+n_3+n_4], pz_new[n_0+n_1+n_2+n_3+n_4])
print(n_0+n_1+n_2+n_3+n_4+n_5, (n_0+n_1+n_2+n_3+n_4+n_5)*dt, px_new[n_0+n_1+n_2+n_3+n_4+n_5], pz_new[n_0+n_1+n_2+n_3+n_4+n_5])
print(n_0+n_1+n_2+n_3+n_4+n_5+n_6, (n_0+n_1+n_2+n_3+n_4+n_5+n_6)*dt, px_new[n_0+n_1+n_2+n_3+n_4+n_5+n_6], pz_new[n_0+n_1+n_2+n_3+n_4+n_5+n_6])
print(new_tr.shape[0] - 1, timestamps[-1], px_new[-1], pz_new[-1])
plt.plot(px_new, label='px_new')
plt.plot(pz_new, label='pz_new')
plt.legend()
plt.show()

plt.plot(tr)
plt.plot(new_tr)
plt.show()

np.save(os.path.join(script_path, '..', f'data-motion-{motion}', f'tr_{agent}_v.npy'), new_tr)
np.save(os.path.join(script_path, '..', f'data-motion-{motion}', f'tr_{agent}_dt.npy'), dt)