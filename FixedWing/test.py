import sys
sys.path.append('..')

from cost import CostFunctor
from nempc import NEMPC
import numpy as np
import matplotlib.pyplot as plt
from time import time

from tools import Euler2Quaternion
from tools import boxminus

np.set_printoptions(precision=4, suppress=True)

cost_fn = CostFunctor(use_penalty=False)
u_eq = np.array([0,0,0.5,0])
phi0 = 0.0  # roll angle
theta0 =  0.0  # pitch angle
psi0 = 0.0  # yaw angle

e = Euler2Quaternion(phi0, theta0, psi0, 1)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)

x_des = np.array([[0],  # (0)
                   [0],   # (1)
                   [-100],   # (2)
                   [15],    # (3)
                   [0],    # (4)
                   [0],    # (5)
                   [e0],    # (6)
                   [e1],    # (7)
                   [e2],    # (8)
                   [e3],    # (9)
                   [0],    # (10)
                   [0],    # (11)
                   [0]])   # (12)

pop_sizes = [5000,1000,500,300,100]
gen_sizes = [500, 300, 200, 100]

for gs in gen_sizes:
    plt.figure()
    for ps in pop_sizes:
        ctrl = NEMPC(cost_fn, 9, 4, cost_fn.u_min, cost_fn.u_max, u_eq,
            horizon=10, population_size=ps, num_parents=10, num_gens=gs,
            mode='tournament')

        start = time()
        u_traj = ctrl.solve(x_des)
        end = time()

        print(f'PS: {ps} GS: {gs}')
        print(f'Elapsed time: {end-start}')
        print(u_traj.reshape(-1,4))

        plt.plot(ctrl.cost_hist, label='$N_p = $'+str(ps))

    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('cost')
    plt.pause(1.0)

plt.show()
