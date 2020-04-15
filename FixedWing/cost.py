import numpy as np
from copy import deepcopy
from IPython.core.debugger import set_trace

from FixedWing import FixedWing

from tools import Euler2Quaternion
from tools import boxminus
from tools import Quaternion2Euler
from tools import normalize

def follow_orbit(state):

        # p = np.array([state[0], state[1], state[2]]).T
        g = 9.81
        Vg = state[3]

        e2 = np.array([0,1,0])
        e3 = np.array([0,0,1])

        # psi = state.psi

        phi,theta,psi = Quaternion2Euler(normalize(state[6:10]))
        chi = psi

        k_orbit = 0.75

        c = np.array([0, 0, -100])
        rho = 20
        dir = -1
        d = np.sqrt((state[1]-c.item(1))**2+(state[0]-c.item(0))**2)
        varphi = np.arctan2(state[1]-c.item(1), state[0]-c.item(0))
        while any(varphi - chi < -np.pi):
            # pdb.set_trace()
            varphi = varphi + 2*np.pi * (varphi - chi < -np.pi).astype(int)
        while any(varphi - chi > np.pi):
            # pdb.set_trace()
            varphi = varphi - 2*np.pi * (varphi - chi > np.pi).astype(int)

        chi_c = varphi + dir*(np.pi/2+np.arctan(k_orbit*(d-rho)/rho))

        # self.autopilot_commands.course_command = chi_c
        # self.autopilot_commands.altitude_command = c.item(2)
        phi_feedforward = dir*np.arctan(Vg**2./(g*rho*np.cos(chi_c-psi)))


        R_90 = np.array([[0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]])
        pos_noZ = np.array([state[0][0],state[1][0],0])
        closest_to_cir = 2.5 * normalize(pos_noZ)
        closest_to_cir[2] = c[2]
        # line_dir = R_90 @ normalize(pos_noZ)
        # line_err = state[0:3].T - closest_to_cir

        phi_feedforward = 0
        wp = np.array([[100],[0],[-100]])
        wp_prev = np.array([[0],[0],[-100]])
        line_dir = (wp - wp_prev) / np.linalg.norm(wp-wp_prev)
        # set_trace()
        line_err = (np.eye(3)- line_dir * line_dir.T) @ (state[0:3] - wp_prev)
        line_err = line_err.T

        chi_l = np.arctan2(line_dir[1], line_dir[0])

        gamma_l = np.arctan(-line_dir[2]/ np.sum(np.abs(line_dir[0:2])**2,axis=0)**(1./2))
        gamma = theta

        R_I2l = np.identity(3)
        R_I2l[0,0] = np.cos(chi_l)
        R_I2l[0,1] = np.sin(chi_l)
        R_I2l[1,0] = -np.sin(chi_l)
        R_I2l[1,1] = np.cos(chi_l)

        k_orbit = 0.1
        chi_ref = chi_l - 0.2618 * 2.0 / np.pi * np.arctan(k_orbit * e2.dot(R_I2l @ line_err.T))
        chi_c = chi_ref


        k_path = 0.5
        gamma_ref = gamma_l + 0.2618 * 2.0 / np.pi * np.arctan(k_path * e3.dot(R_I2l @ line_err.T))
        # gamma_ref += 0.5

        # return normalize(Euler2Quaternion(0.0*np.ones(state.shape[1]),
        #     0.0*np.ones(state.shape[1]),
        #     chi_c*np.ones(state.shape[1]),
        #     state.shape[1]))
        # pdb.set_trace()
        return normalize(Euler2Quaternion(phi_feedforward*np.ones(state.shape[1]),
            gamma_ref[0]*np.ones(state.shape[1]),
            chi_c*np.ones(state.shape[1]),
            state.shape[1]))




class CostFunctor:
    def __init__(self, return_states=False, use_penalty=False):
        # self.dyn = Dynamics(dt=0.02)
        self.fw = FixedWing()
        self.x0 = self.fw._start
        self.dt = 0.01
        self.Q = np.diag([0,0,100, 0,0,0, 50,50,50, 0,0,0])
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

        self.x_des = np.array([[0],  # (0)
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

        # self.initialized = False
        self.return_states = return_states
        w_max = np.pi / 1.5
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
        xk = np.tile(self.x0, pop_size)
        cost = np.zeros(pop_size)
        u_traj = deepcopy(z.reshape(pop_size,horizon,4))

        x_traj = np.zeros((self.fw._start.shape[0],horizon))

        # self.fw._state = deepcopy(self.fw._start)

        self.x_des = np.tile(self.x_des, pop_size)

        for k in range(horizon):

            # if self.return_states:


            xk = self.fw.forward_simulate_dt(xk, u_traj[:,k].T, self.dt)

            # print(follow_orbit(xk).shape)
            # if self.x_des[6:10].size == 4:
            #     set_trace()

            self.x_des[6:10] = follow_orbit(xk)

            one =  np.sum(self.Q[0:6,0:6].T.dot(np.square(xk[0:6]-self.x_des[0:6])), axis=0)
            two = np.sum(self.Q[6:9,6:9] @ np.square(boxminus(deepcopy(xk[6:10]),deepcopy(self.x_des[6:10]))), axis=0)
            three = np.sum(self.Q[9:,9:].dot(np.square(xk[10:]-self.x_des[10:])), axis=0)

            # for i in range(u.shape[0]):
            cost += one + two + three
            # set_trace()
            # print(cost)

            if self.return_states:
                x_traj[:,k] = xk.flatten()

            # xk = self.dyn.rk4(xk, u_traj[:,k].T)
            # x_err = xk - self.x_des[:,None]
            # if k < horizon-1:
            #     cost += np.sum(x_err**2 * self.Q[:,None], axis=0)
            # else:
            #     cost += np.sum(x_err**2 * self.Qf[:,None], axis=0)

        self.x_des = self.x_des[:,0].reshape((13,1))


        if self.return_states:
            return x_traj, cost

        return cost
