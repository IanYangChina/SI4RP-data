import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
# trajectory = np.zeros(shape=(800, 6), dtype=np.float32)
#
# trajectory[:100, 2] = -0.075
# trajectory[100:200, 0] = 0.075
# trajectory[200:400, 0] = -0.075
# trajectory[400:500, 0] = 0.075
# trajectory[500:600, 1] = -0.075
# trajectory[600:800, 1] = 0.075

# np.save(os.path.join(script_path, '..', 'demo_files', 'eef_v_trajectory_test.npy'), trajectory)

tr = np.load(os.path.join(script_path, '..', 'data-motion-2', 'tr2_eef_v_0.npy'))
timestamps = np.load(os.path.join(script_path, '..', 'data-motion-2', 'tr_timestamps.npy'))
tds = np.diff(timestamps)
d = 2
p = [0.0]

for n in range(tr.shape[0] - 1):
    p.append(p[-1] + tr[n][d] * tds[n])
    print(n, p[-1])
print(0, timestamps[0], p[0])
print(40, timestamps[40], p[40])
print(tr.shape[0] - 1, timestamps[-1], p[-1])
plt.plot(p)
plt.show()

dt = timestamps[40] / 40
end_point = np.ceil(timestamps[-1] / dt).astype(np.int32)
new_tr = np.zeros(shape=(end_point, 6), dtype=np.float32)

print(f'dt: {dt}')
new_tr[:40, d] = -0.02 / timestamps[40]
new_tr[40:, d] = 0.03 / (dt * (new_tr.shape[0] - 40))

d = 2
p = [0.0]
for n in range(new_tr.shape[0]):
    p.append(p[-1] + new_tr[n][d] * dt)

print(0, timestamps[0], p[0])
print(40, timestamps[40], p[40])
print(new_tr.shape[0] - 1, timestamps[-1], p[-1])
plt.plot(p)
plt.show()

# d = 0
# p = [0.0]
# for n in range(tr.shape[0] - 1):
#     p.append(p[-1] + tr[n][d] * tds[n])
#     # print(n, p[-1])
# print(0, timestamps[0], p[0])
# for n in [40, 70]:
#     print(n, timestamps[n], p[n])
# print(tr.shape[0] - 1, timestamps[-1], p[-1])
# plt.plot(p)
# plt.show()
#
# new_tr[40:70, d] = 0.03 / (dt * (70 - 40))
#
# d = 0
# p = [0.0]
# for n in range(new_tr.shape[0]):
#     p.append(p[-1] + new_tr[n][d] * dt)
#
# print(0, timestamps[0], p[0])
# for n in [40, 70]:
#     print(n, timestamps[n], p[n])
# print(new_tr.shape[0] - 1, timestamps[-1], p[-1])
# plt.plot(p)
# plt.show()

# plt.plot(tr)
# plt.show()
# plt.plot(new_tr)
# plt.show()
#
np.save(os.path.join(script_path, '..', 'data-motion-2', 'tr_eef_v.npy'), new_tr)
np.save(os.path.join(script_path, '..', 'data-motion-2', 'tr_dt.npy'), dt)