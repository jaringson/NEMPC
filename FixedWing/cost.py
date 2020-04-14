import numpy as np
from copy import deepcopy
from IPython.core.debugger import set_trace

from FixedWing import FixedWing

from tools import Euler2Quaternion
from tools import boxminus

class CostFunctor:
    def __init__(self, return_states=False, use_penalty=False):
        # self.dyn = Dynamics(dt=0.02)
        self.fw = FixedWing()
        self.dt = 0.01
        self.Q = 0.1*np.diag([0,0,100, 1,0,0, 50,50,50, 0,0,0])
        # self.Qf = np.array([10,10,100,4.5,4.5,5,0.8,0.8,1])
        # self.x0 = np.array([0,0,-5.,0,0,0,0,0,0])
        # self.x_des = self.x0.copy()


        phi0 = 0.0  # roll angle
        theta0 =  0.  # pitch angle
        psi0 = 0.0  # yaw angle

        e = Euler2Quaternion(phi0, theta0, psi0, 1)
        e0 = e.item(0)
        e1 = e.item(1)
        e2 = e.item(2)
        e3 = e.item(3)

        self.xgoal = np.array([[0],  # (0)
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

        # self.initialized = False
        self.return_states = return_states
        w_max = np.pi / 2
        self.u_max = np.array([w_max, w_max, 1, w_max])
        self.u_min = np.array([-w_max,-w_max, 0, -w_max])
        self.mu = 10.

    def __call__(self, z):
        if z.shape[0] == z.size:
            Tm = z.size
            pop_size = 1
        else:
            pop_size, Tm = z.shape # default to use with nempc
        horizon = Tm // 4
        # xk = np.tile(self.x0[:,None], pop_size)
        cost = np.zeros(pop_size)
        u_traj = z.reshape(pop_size,horizon,4)
        if self.return_states:
            x_traj = []

        # self.fw._state = deepcopy(self.fw._start)

        xk = np.tile(self.fw._start, pop_size)

        for k in range(horizon):

            xk = self.fw.forward_simulate_dt(xk, u_traj[:,k].T, self.dt)

            # set_trace()

            cost = np.zeros(pop_size)


            one =  np.sum(self.Q[0:6,0:6].dot(np.square(xk[0:6]-self.xgoal[0:6])), axis=0)
            two = np.sum(self.Q[6:9,6:9] @ np.square(boxminus(deepcopy(xk[6:10]),deepcopy(self.xgoal[6:10]))), axis=0)
            three = np.sum(self.Q[9:,9:].dot(np.square(xk[10:]-self.xgoal[10:])), axis=0)

            # for i in range(u.shape[0]):
            cost = one + two + three

            # xk = self.dyn.rk4(xk, u_traj[:,k].T)
            # x_err = xk - self.x_des[:,None]
            # if k < horizon-1:
            #     cost += np.sum(x_err**2 * self.Q[:,None], axis=0)
            # else:
            #     cost += np.sum(x_err**2 * self.Qf[:,None], axis=0)



        if self.return_states:
            x_traj = np.array(x_traj).flatten()
            return x_traj, cost
        
        return cost
