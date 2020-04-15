import sys
sys.path.append('..')

from cost import CostFunctor
from nempc import NEMPC
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from IPython.core.debugger import set_trace

from FixedWing import FixedWing

from tools import Euler2Quaternion
from tools import boxminus
from tools import Quaternion2Euler

from NonlinearEMPC import NonlinearEMPC

cost_fn = CostFunctor(use_penalty=False)
u_eq = np.array([0,-1.5,1.0,0])
ctrl = NEMPC(cost_fn, 9, 4, cost_fn.u_min, cost_fn.u_max, u_eq, horizon=10,
        population_size=70, num_parents=10, num_gens=200, mode='tournament',
        warm_start=False)

fw = FixedWing()
x = deepcopy(fw._start)

t = 0.0
tf = 10.0
ts = 0.01

phi0 = np.pi/4  # roll angle
theta0 =  np.radians(2.5) #0.0  # pitch angle
psi0 = np.pi/2 #0.0  # yaw angle

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

state_hist = []
input_hist = []
time_hist = []

ctrl.solve(x_des)
ctrl.warm_start = True
ctrl.num_gens = 10
start = time()
while t < tf:
    print(t, end='\r')
    time_hist.append(t)

    phi, theta, psi = Quaternion2Euler(deepcopy(x[6:10]))
    x_in = np.zeros((12,1))
    x_in[0:6] = deepcopy(x[0:6])
    x_in[6:9] = np.array([[phi[0]],[theta[0]], [psi[0]]])
    x_in[9:12] = deepcopy(x[10:13])
    # print(x_in[2])
    state_hist.append(x_in)

    cost_fn.x0 = deepcopy(x)
    cost_fn.x_des = x_des
    u_traj = ctrl.solve(x_des)
    u_star = deepcopy(u_traj[:4])

    # set_trace()
    input_hist.append(u_star)

    x = fw.forward_simulate_dt(x, u_star.reshape((4,1)), ts)
    t += ts
end = time()
print(f'Time: {end-start}')

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
ax1[1,0].plot(time_hist, state_hist[:,6])
ax1[1,0].set_xlabel('roll')
ax1[1,1].plot(time_hist, state_hist[:,7])
ax1[1,1].set_xlabel('pitch')
ax1[1,2].plot(time_hist, state_hist[:,8])
ax1[1,2].set_xlabel('yaw')
# velocity plots
ax1[2,0].plot(time_hist, state_hist[:,3])
ax1[2,0].set_xlabel('vel_n')
ax1[2,1].plot(time_hist, state_hist[:,4])
ax1[2,1].set_xlabel('vel_e')
ax1[2,2].plot(time_hist, state_hist[:,5])
ax1[2,2].set_xlabel('vel_d')

fig2, ax2 = plt.subplots(4)
# throttle plot
ax2[0].set_title("Fixed Wing Inputs")
ax2[0].plot(time_hist, input_hist[:,0])
ax2[0].set_ylabel('ael')
# angular velocties
ax2[1].plot(time_hist, input_hist[:,1])
ax2[1].set_ylabel('ele')
ax2[2].plot(time_hist, input_hist[:,2])
ax2[2].set_ylabel('thr')
ax2[3].plot(time_hist, input_hist[:,3])
ax2[3].set_ylabel('rud')
ax2[3].set_xlabel('Time (sec)')

fig = plt.figure()
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0:2, 0])
plt.tight_layout()




ax1.set_title("Fixed Wing Positions")
ax1.plot(state_hist[:,0], -state_hist[:,1], label='Actual')
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
# ax1.set_ylim(-15,5)
ax1.set_xlim(-10,225)

ax1.plot([-100,300], [0,0], label='Commanded')

rho = 20
x_t = np.linspace(-rho, rho, 50)
y_t = np.sqrt(rho**2 - x_t**2)
x_t = np.append(x_t,np.linspace(rho, -rho, 50))
# print(y_t.shape)
y_t = np.append(y_t,-y_t)

# ax1.plot(x_t,y_t)
ax1.legend()

ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(time_hist, -state_hist[:,2])
ax2.set_ylabel('Z (m)')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(98,101)
plt.tight_layout()

plt.show()
