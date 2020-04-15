import sys
sys.path.append('..')

from cost import CostFunctor
from nempc import NEMPC
import numpy as np
import matplotlib.pyplot as plt
from time import time
from IPython.core.debugger import set_trace

from tools import Euler2Quaternion
from tools import boxminus

np.set_printoptions(precision=4, suppress=True)

cost_fn = CostFunctor(use_penalty=False)
u_eq = np.array([0.0, -1.5, 1.0, 0.0])
phi0 = 0.0 #np.pi/4  # roll angle
theta0 =  np.radians(5) #0.0  # pitch angle
psi0 = 0.0 #np.pi/2  # yaw angle

e = Euler2Quaternion(phi0, theta0, psi0, 1)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)

x_des = np.array([[0],  # (0)
                   [0],   # (1)
                   [-100],   # (2)
                   [25],    # (3)
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
pop_sizes = [100]
# gen_sizes = [500, 300, 200, 100]
gen_sizes = [100]

horizon = 10

for gs in gen_sizes:
    all_u = np.zeros((len(pop_sizes),horizon,4))
    i = 0
    plt.figure()
    for ps in pop_sizes:
        ctrl = NEMPC(cost_fn, 9, 4, cost_fn.u_min, cost_fn.u_max, u_eq,
            horizon=horizon, population_size=ps, num_parents=10, num_gens=gs,
            mode='tournament')

        start = time()
        u_traj = ctrl.solve(x_des)
        end = time()

        print(f'PS: {ps} GS: {gs}')
        print(f'Elapsed time: {end-start}')
        print(u_traj.reshape(-1,4))

        all_u[i,:,:] = u_traj.reshape(-1,4)
        i += 1

        plt.plot(ctrl.cost_hist, label='$N_p = $'+str(ps))


    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('cost')
    # plt.pause(1.0)

    fig = plt.figure()
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    cost_fn.return_states = True
    for j in range(all_u.shape[0]):
        all_traj, _ = cost_fn(all_u[j,:,:].flatten())
        ax1.plot(all_traj[0], all_traj[1], label='$N_p = $'+str(pop_sizes[j]))
        # set_trace()
        t = np.linspace(cost_fn.dt,horizon*cost_fn.dt,horizon)
        ax2.plot(t, all_traj[2], label='$N_p = $'+str(pop_sizes[j]))
    cost_fn.return_states = False

    ax1.legend()



    plt.pause(1.0)
    # set_trace()

plt.show()
