import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.core.debugger import set_trace
import time
from copy import deepcopy

from FixedWing import *


u_pen = np.array([ 1.99968847, -1.99937726,  0.99963555, -1.99974936, -1.85635716,
       -1.23882894,  0.99955841, -2.00000374,  0.18832958,  1.06771522,
        0.99962614,  1.260342  , -0.53428481, -2.00000832,  0.9997045 ,
        0.02068783, -0.28880833, -0.6296831 ,  0.99978117, -0.42219638,
       -0.24789309, -1.16855602,  0.99984204, -0.32701288, -0.29349161,
       -1.04229027,  0.99988301, -0.54846785, -0.22825965, -1.09610842,
        0.99997923, -0.39563048, -0.32289351, -1.35541227,  0.99995413,
       -0.41654092, -0.21861447, -0.89845965,  0.99999614, -0.41061412])

u_sci = np.array([ 1.        , -1.        ,  1.        , -1.        ,  0.35224218,
       -1.        ,  1.        , -1.        , -1.        , -0.99560284,
        1.        , -1.        , -0.8298546 , -0.51014836,  1.        ,
       -0.25476999,  0.05618059, -0.84389363,  1.        ,  0.87874051,
       -0.40577091, -1.        ,  1.        , -1.        , -0.20094489,
       -1.        ,  1.        , -0.20273986, -0.21962922, -1.        ,
        1.        , -0.64607618, -0.12951911, -1.        ,  1.        ,
       -0.39834518, -0.39998068, -1.        ,  1.        , -0.15620109])

# u_pen = np.array([ 1.87363681e-04, -2.00006705e+00,  1.00002660e+00, -2.33882312e-04,
#        -3.17333717e-04, -1.10245442e+00,  1.00003213e+00,  8.27633534e-05,
#         2.57947815e-04,  1.07423706e+00,  1.00002629e+00,  1.58974966e-03,
#        -3.14965417e-04, -1.70142671e+00,  1.00001652e+00, -3.92501427e-03,
#         4.49127749e-04,  3.24570761e-01,  1.00001007e+00,  5.22999769e-03,
#        -4.62517403e-04, -9.14842233e-01,  1.00000702e+00, -6.10806468e-03,
#        -3.50761633e-05, -1.11973168e-01,  1.00000570e+00,  7.45938596e-03,
#         7.29422455e-04, -2.06469285e-01,  9.99995137e-01, -7.54311938e-03,
#        -6.12713791e-04,  7.72797519e-02,  1.00000036e+00,  4.82927089e-03,
#         5.42388529e-04, -3.54129731e-01,  1.86100734e-02, -2.16677653e-04])
#
#
# u_sci = np.array([ 6.14669381e-04, -9.99999996e-01,  9.99999999e-01, -2.05823018e-03,
#        -6.13579804e-04, -1.00000000e+00,  9.99999999e-01,  5.24138728e-03,
#         1.11727898e-04, -1.00000000e+00,  9.99999999e-01, -5.59979887e-03,
#        -4.79853153e-04, -1.59228654e-01,  1.00000000e+00,  2.83511273e-03,
#         1.06615507e-03, -9.99999890e-01,  1.00000000e+00,  3.05383687e-04,
#        -1.35143616e-03, -1.03345335e-01,  1.00000000e+00, -3.77528113e-03,
#         1.27016587e-03, -7.67814052e-02,  1.00000000e+00,  8.51543590e-03,
#        -5.83279918e-04, -3.70966284e-01,  1.00000000e+00, -1.17545768e-02,
#         4.95564007e-05, -2.21164060e-02,  1.00000000e+00,  8.09421382e-03,
#        -2.88587635e-04, -2.93187969e-01,  7.67200137e-01,  2.50612842e-03])



def obj_func(u, dt, fw):
    global num_pts, horizon
    cost = 0
    # u = u.reshape((4,2))
    fw._state = deepcopy(fw._start)

    Q = 0.1*np.diag([0,0,5, 5,1,1, 50,50,50, 0,0,0])
    Qx = np.zeros((fw.numStates-1,fw.numStates))

    spot = 0
    counter = 0
    u_now = u[4*spot: 4*spot+4]

    all_x = np.array([])
    all_u = np.array([])
    time = 0
    all_time = []

    for i in range(horizon):
        counter += 1
        if counter >= horizon // num_pts and i != horizon-1:
            spot += 1
            # set_trace()
            u_now = u[4*spot: 4*spot+4]
            counter = 0
        # print(4*i,4*i+4)
        # u_now = u[4*i:4*i+4]
        x = fw.forward_simulate_dt(fw._state, u_now, dt)
        if all_x.size == 0:
            all_x = deepcopy(x)
            all_u = deepcopy(np.atleast_2d(np.array(u_now))).T
        else:
            all_x = np.append(all_x,deepcopy(x),axis=1)
            all_u = np.append(all_u,deepcopy(np.atleast_2d(np.array(u_now))).T,axis=1)
        all_time.append(time)
        time += dt
    return all_x, all_u, all_time


horizon = 100
num_pts = 10
dt = 0.01

fw_pen = FixedWing()
fw_sci = FixedWing()

all_x_pen, all_u_pen, all_time_pen = obj_func(u_pen, dt, fw_pen)
all_x_sci, all_u_sci, all_time_sci = obj_func(u_sci, dt, fw_sci)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(all_x[0,:], -all_x[1,:], -all_x[2,:])
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")
# ax.set_zlim3d(90,110)


fig = plt.figure()
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0:2, 0])
plt.tight_layout()

ax1.set_title("Fixed Wing Positions")
ax1.plot(all_x_pen[0,:], -all_x_pen[1,:], label="Penalty")
ax1.plot(all_x_sci[0,:], -all_x_sci[1,:], label="Scipy")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_ylim(-4,1)
ax1.legend()

ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(all_time_pen, -all_x_pen[2,:])
ax2.plot(all_time_sci, -all_x_sci[2,:])
ax2.set_ylabel('Z (m)')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(98,101)
plt.tight_layout()

# fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
# axes.set_title("Fixed Wing Positions")
# axes.plot(all_x[0,:], all_x[1,:])
# axes.set_xlabel("X (m)")
# axes.set_ylabel("Y (m)")

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
axes[0].set_title("Fixed Wing Positions")
axes[0].plot(all_time_pen, all_x_pen[0,:], label="Penalty")
axes[0].plot(all_time_sci, all_x_sci[0,:], label="Scipy")
axes[0].set_ylabel("X (m)")
axes[0].legend()
axes[1].plot(all_time_pen, -all_x_pen[1,:])
axes[1].plot(all_time_sci, -all_x_sci[1,:])
axes[1].set_ylabel("Y (m)")
axes[1].set_ylim(-10,10)
axes[2].plot(all_time_pen, -all_x_pen[2,:])
axes[2].plot(all_time_sci, -all_x_sci[2,:])
axes[2].set_ylabel("Z (m)")
axes[2].set_ylim(90,110)
axes[2].set_xlabel("Time (s)")

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
axes[0].set_title("Fixed Wing Velocities")
axes[0].plot(all_time_pen, all_x_pen[3,:], label="Penalty")
axes[0].plot(all_time_sci, all_x_sci[3,:], label="Scipy")
axes[0].set_ylabel("X (m/s)")
axes[0].legend()
axes[1].plot(all_time_pen, -all_x_pen[4,:])
axes[1].plot(all_time_sci, -all_x_sci[4,:])
axes[1].set_ylabel("Y (m/s)")
axes[2].plot(all_time_pen, -all_x_pen[5,:])
axes[2].plot(all_time_sci, -all_x_sci[5,:])
axes[2].set_ylabel("Z (m/s)")
axes[2].set_xlabel("Time (s)")

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)
axes[0].set_title("Fixed Wing Inputs")
axes[0].plot(all_time_pen, all_u_pen[0,:], label="Penalty")
axes[0].plot(all_time_sci, all_u_sci[0,:], label="Scipy")
axes[0].set_ylabel("Aileron")
axes[0].legend()
axes[1].plot(all_time_pen, all_u_pen[1,:])
axes[1].plot(all_time_sci, all_u_sci[1,:])
axes[1].set_ylabel("Elevator")
axes[2].plot(all_time_pen, all_u_pen[2,:])
axes[2].plot(all_time_sci, all_u_sci[2,:])
axes[2].set_ylabel("Throttle")
axes[3].plot(all_time_pen, all_u_pen[3,:])
axes[3].plot(all_time_sci, all_u_sci[3,:])
axes[3].set_ylabel("Rudder")
axes[3].set_xlabel("Time (s)")

plt.show()
