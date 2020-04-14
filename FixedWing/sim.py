import sys
sys.path.append('..')

from cost import CostFunctor
from nempc import NEMPC
import numpy as np
import matplotlib.pyplot as plt
from time import time

cost_fn = CostFunctor(use_penalty=False)
u_eq = np.array([0.5,0,0,0])
ctrl = NEMPC(cost_fn, 9, 4, cost_fn.u_min, cost_fn.u_max, u_eq, horizon=10,
        population_size=500, num_parents=10, num_gens=100, mode='tournament')

x_des = np.array([0,1,-5.0,0,0,0,0,0,0])

start = time()
u_traj = ctrl.solve(x_des)
end = time()

print('Elapsed time: ', end - start)

np.set_printoptions(precision=4, suppress=True)
print(u_traj.reshape(-1,4))

plt.figure()
plt.plot(ctrl.cost_hist)

plt.show()
