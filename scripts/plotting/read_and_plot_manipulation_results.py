import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 30})

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
