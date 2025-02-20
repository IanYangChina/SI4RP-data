import taichi as ti
import numpy as np
import torch
import math


NUM_MATERIAL = 5
# particle materials
SAND = 0
ELASTIC = 1
CLAY = 2
RIGID = 3
WATER = 4

# other materials
GROUND = 51
TABLE = 52
CONTAINER = 53
MANIPULATOR = 63

FRAME = 100
TARGET = 101
EFFECTOR = 102
XAXIS = 103
YAXIS = 104
ZAXIS = 105

MAT_ELASTIC = 200
MAT_PLASTO_ELASTIC = 201
MAT_RIGID = 202
MAT_SAND = 203
MAT_LIQUID = 204

# material name
MAT_NAME = {
    SAND: 'sand',
    ELASTIC: 'elastic',
    CLAY: 'clay',
    RIGID: 'rigid',
    WATER: 'water'
}

# material class
MAT_CLASS = {
    SAND: MAT_SAND,
    ELASTIC: MAT_ELASTIC,
    CLAY: MAT_PLASTO_ELASTIC,
    RIGID: MAT_RIGID,
    WATER: MAT_LIQUID
}

# default color
COLOR = {
    # SAND: (51/255, 34/255, 20/255, 0.5),
    SAND: (162/255, 107/255, 59/255, 1.0),
    ELASTIC: (0, 0, 0, 1.0),
    CLAY: (210/255, 105/255, 30/255, 1.0),
    RIGID: (1.0, 0.5, 0.5, 1.0),
    WATER: (0.0, 0.0, 1.0, 0.5),

    TABLE: (0.1, 0.1, 0.1, 1.0),
    CONTAINER: (5/255, 5/255, 0, 1.0),
    MANIPULATOR: (0.95, 0.95, 0.95, 1.0),

    FRAME: (1.0, 0.2, 1.0, 1.0),
    TARGET: (0.2, 0.9, 0.2, 0.4),
    EFFECTOR: (0.95, 0.95, 0.95, 1.0),
    XAXIS: (1.0, 0.0, 0.0, 1.0),
    YAXIS: (0.0, 1.0, 0.0, 1.0),
    ZAXIS: (0.0, 0.0, 1.0, 1.0)
}

# initial parameters
# E: [1e4, 3e5]
# nu: [0.01, 0.48]
# yield_stress: [1e3, 1e6]
# rho: [1000, 2000]
# friction: [0.01, 2.0]

# Frictions
# [0, 2], 0: no friction, 2: max friction, >2: sticky
FRICTION = {
    GROUND: 2,
    TABLE: 0.5,
    CONTAINER: 0.5,
    MANIPULATOR: 0.5
}

SAND_FRICTION_ANGLE = 10 # in degree
YIELD_STRESS = 1e4
THETA_C = 0.025
THETA_S = 0.0045

E = {
    SAND: 120000,
    ELASTIC: 40000,
    CLAY: 40000,
    RIGID: 10,
    WATER: 10000
}

NU = {
    SAND: 0.2,
    ELASTIC: 0.4,
    CLAY: 0.4,
    RIGID: 0.1,
    WATER: 0.1
}

RHO = {
    SAND: 1500,
    ELASTIC: 1000,
    CLAY: 1000,
    RIGID: 100,
    WATER: 1000
}

# dtype
dprecision = 32
DTYPE_TI = eval(f'ti.f{dprecision}')
DTYPE_NP = eval(f'np.float{dprecision}')
DTYPE_TC = eval(f'torch.float{dprecision}')

EPS = 1e-12

# misc
NOWHERE = [-100.0, -100.0, -100.0]
