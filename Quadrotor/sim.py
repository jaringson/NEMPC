import sys
sys.path.append('..')

from cost import CostFunctor
from nempc import NEMPC
from dynamics import Dynamics
import numpy as np
import matplotlib.pyplot as plt

cost_fn = CostFunctor(use_penalty=False)
cost_fn.Q = np.array([10,10,100,2.5,2.5,2,1,1,1])
u_eq = np.array([0.5,0,0,0])
ctrl = NEMPC(cost_fn, 9, 4, cost_fn.u_min, cost_fn.u_max, u_eq, horizon=10,
        population_size=500, num_parents=10, num_gens=200, mode='tournament',
        warm_start=True)
dyn = Dynamics()

t = 0.0
tf = 10.0
ts = 0.02
x = np.array([0,0,-5.0,0,0,0,0,0,0])
x_des = np.array([0,1,-6.0,0,0,0,0,0,0])

state_hist = []
input_hist = []
time_hist = []

while t < tf:
    print(t, end='\r')
    time_hist.append(t)
    state_hist.append(x)

    cost_fn.x0 = x
    cost_fn.x_des = x_des
    u_traj = ctrl.solve(x_des)
    u_star = u_traj[:4]

    input_hist.append(u_star)

    x = dyn.rk4(x, u_star)
    t += ts

state_hist = np.array(state_hist)
input_hist = np.array(input_hist)

fig1, ax1 = plt.subplots(3,3)
# position plots
ax1[0,0].plot(time_hist, state_hist[:,0])
ax1[0,0].set_xlabel('pos_n')
ax1[0,1].plot(time_hist, state_hist[:,1])
ax1[0,1].set_xlabel('pos_e')
ax1[0,2].plot(time_hist, state_hist[:,2])
ax1[0,2].set_xlabel('pos_d')
# attitude plots
ax1[1,0].plot(time_hist, state_hist[:,3])
ax1[1,0].set_xlabel('roll')
ax1[1,1].plot(time_hist, state_hist[:,4])
ax1[1,1].set_xlabel('pitch')
ax1[1,2].plot(time_hist, state_hist[:,5])
ax1[1,2].set_xlabel('yaw')
# velocity plots
ax1[2,0].plot(time_hist, state_hist[:,6])
ax1[2,0].set_xlabel('vel_n')
ax1[2,1].plot(time_hist, state_hist[:,7])
ax1[2,1].set_xlabel('vel_e')
ax1[2,2].plot(time_hist, state_hist[:,8])
ax1[2,2].set_xlabel('vel_d')

fig2, ax2 = plt.subplots(4)
# throttle plot
ax2[0].plot(time_hist, input_hist[:,0])
ax2[0].set_xlabel('throttle')
# angular velocties
ax2[1].plot(time_hist, input_hist[:,1])
ax2[1].set_xlabel('wx')
ax2[2].plot(time_hist, input_hist[:,2])
ax2[2].set_xlabel('wy')
ax2[3].plot(time_hist, input_hist[:,3])
ax2[3].set_xlabel('wz')

plt.show()
