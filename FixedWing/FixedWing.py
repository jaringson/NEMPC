# from Model import Model
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from copy import deepcopy
import aerosonde_parameters as MAV
# import vaporlite_parameters as MAV

from tools import normalize
from tools import Quaternion2Euler
import pdb
import warnings

from IPython.core.debugger import set_trace

class FixedWing():

    def __init__(self):
        self.numStates = 13
        self.numInputs = 4

        self._start = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)

        self._state = deepcopy(self._start)

        self.state_max = np.array([[5],  # (0)
                               [5],   # (1)
                               [5],   # (2)

                               [6],    # (3)
                               [1],    # (4)
                               [1],    # (5)

                               [1],    # (6)
                               [1],    # (7)
                               [1],    # (8)
                               [1],    # (9)

                               [np.pi/1],    # (10)
                               [np.pi/1],    # (11)
                               [np.pi/1]])   # (12)

        self.uMax = 1.0
        self.uTMax = 1.0
        # self.uMax = 0.9
        # self.uTMax = 0.9
        self.uTMin = 0
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0

        self.plotObjects = []

        # plane in ned
        self.plane = np.array([[0,0,0],
                  [0.5,0,0],
                  [0.1,0,0],
                  [0,0.5,-0.1], #left wing
                  [0.1,0,0],
                  [0,-0.5,-0.1], #right wing
                  [0.1,0,0],
                  [-0.5,0,0],
                  [-0.5,0,-0.25],
                  [-0.5,0.1,-0.25],
                  [-0.5,-0.1,-0.25]]).T

    def draw_plane_nwu(self, plane_in):
        R = np.array([[1,0,0],
                      [0,-1,0],
                      [0,0,-1]])
        p = R.dot(plane_in)
        return p[0,:], p[1,:], p[2,:]


    def forward_simulate_dt(self,x,u,dt=.01):
        x = deepcopy(x)
        self._state = x
        # u = deepcopy(u.clip(-self.uMax,self.uMax))
        u = deepcopy(u)
        x = x.reshape([self.numStates,-1])
        # xdot = np.zeros(x.shape)
        forces_moments = self._forces_moments(u)
        # xdot = self._derivatives(x, forces_moments).reshape((-1,13)).T
        # xdot[6:10] = normalize(xdot[6:10])
        # xdot[1,:] = x[0,:]
        # x = x + xdot*dt

        # self._state[self._state<-1e100]=0
        # self._state[self._state>1e100]=0
        k1 = self._derivatives(self._state, forces_moments).reshape((-1,13)).T
        k1[k1<-1e100]=0
        k1[k1>1e100]=0
        k2 = self._derivatives(self._state + dt/2.*k1, forces_moments).reshape((-1,13)).T
        k2[k2<-1e100]=0
        k2[k2>1e100]=0
        k3 = self._derivatives(self._state + dt/2.*k2, forces_moments).reshape((-1,13)).T
        k3[k3<-1e100]=0
        k3[k3>1e100]=0
        k4 = self._derivatives(self._state + dt*k3, forces_moments).reshape((-1,13)).T
        k4[k4<-1e100]=0
        k4[k4>1e100]=0
        self._state += dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        self._state[self._state<-1e100]=0
        self._state[self._state>1e100]=0

        self._state[6:10] = normalize(self._state[6:10])
        x = deepcopy(self._state)



        return x




    def visualize(self,x,ax,color='red'):
        # CoM = [-0.5*np.sin(x[1]),0.5*np.cos(x[1])]
        # theta = x[1]
        #
        # x = [CoM[0] + self.l/2.0*np.sin(theta),CoM[0] - self.l/2.0*np.sin(theta)]
        # y = [CoM[1] - self.l/2.0*np.cos(theta),CoM[1] + self.l/2.0*np.cos(theta)]
        #
        # massX = CoM[0] - self.l/2.0*np.sin(theta)
        # massY = CoM[1] + self.l/2.0*np.cos(theta)

        for plot in self.plotObjects:
            plot[0].remove()
        self.plotObjects = []

        # # self.plotObjects.append(ax.scatter(x[0], x[1], -x[2], 'bo', c='blue'))
        # self.plotObjects.append(ax.plot(*self.draw_plane_nwu(self.plane), linewidth=2, color='red'))
        phi, theta, psi = Quaternion2Euler(x[6:10])

        Rphi = np.array([[1,0,0],
                  [0,np.cos(-phi),np.sin(-phi)],
                  [0,-np.sin(-phi),np.cos(-phi)]])
        Rtheta = np.array([[np.cos(-theta),0,-np.sin(-theta)],
                      [0,1,0],
                      [np.sin(-theta),0,np.cos(-theta)]])
        Rpsi = np.array([[np.cos(-psi),np.sin(-psi),0],
                      [-np.sin(-psi),np.cos(-psi),0],
                      [0,0,1]])

        T = np.array([x[0],x[1],x[2]])

        R = Rphi.dot(Rtheta).dot(Rpsi)
        # pdb.set_trace()

        # plt.clf()
        xs, ys, zs = self.draw_plane_nwu(R.dot(1.5*self.plane)+T)
        self.plotObjects.append(ax.plot(xs, ys, zs, linewidth=2, color=color))

        # plt.draw()
        # plt.plot(x[0],x[1], 'bo')
        # # ax.scatter(x[0], x[1], x[2], 'bo')
        # # plt.scatter(massX,massY,50,'r')
        # plt.axis([-20,20,-20,20])
        ax.set_xlim3d([-6, 6])
        ax.set_ylim3d([-6, 6])
        ax.set_zlim3d([-10, 20])
        plt.ion()
        plt.show()
        plt.pause(.0000001)


    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state[0]
        pe = state[1]
        pd = state[2]
        u = state[3]
        v = state[4]
        w = state[5]
        # state[6:10] = normalize(state[6:10])
        e0 = state[6]
        e1 = state[7]
        e2 = state[8]
        e3 = state[9]
        p = state[10]
        q = state[11]
        r = state[12]
        #   extract forces/moments
        fx = forces_moments[:,0]
        fy = forces_moments[:,1]
        fz = forces_moments[:,2]
        l = forces_moments[:,3]
        m = forces_moments[:,4]
        n = forces_moments[:,5]


        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
            # position kinematics
                pn_dot = (e1**2+e0**2-e2**2-e3**2)*u + 2*(e1*e2-e3*e0)*v + 2*(e1*e3+e2*e0)*w
                pe_dot = 2*(e1*e2+e3*e0)*u + (e2**2+e0**2-e1**2-e3**2)*v + 2*(e2*e3-e1*e0)*w
                pd_dot = 2*(e1*e3-e2*e0)*u + 2*(e2*e3+e1*e0)*v + (e3**2+e0**2-e1**2-e2**2)*w
            except Warning as e:
                pdb.set_trace()
                print(e)

        # position dynamics
        mass = MAV.mass
        u_dot = (r*v-q*w)+fx/mass
        v_dot = (p*w-r*u)+fy/mass
        w_dot = (q*u-p*v)+fz/mass

        # rotational kinematics
        e0_dot = 0.5*(-p*e1-q*e2-r*e3)
        e1_dot = 0.5*(p*e0+r*e2-q*e3)
        e2_dot = 0.5*(q*e0-r*e1+p*e3)
        e3_dot = 0.5*(r*e0+q*e1-p*e2)

        # rotatonal dynamics
        p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + MAV.gamma3*l + MAV.gamma4*n
        q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2-r**2) + m/MAV.Jy
        r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + MAV.gamma4*l + MAV.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T

        return x_dot

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_t, delta_r)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # assert delta.shape == (4,1)
        da = delta[0]
        de = delta[1]
        dt = delta[2]
        dr = delta[3]

        e0 = self._state[6]
        e1 = self._state[7]
        e2 = self._state[8]
        e3 = self._state[9]
        p = self._state[10]
        q = self._state[11]
        r = self._state[12]

        self._Va = np.sqrt(self._state[3]**2 + self._state[4]**2 + self._state[5]**2)
        self._alpha = np.arctan(self._state[5]/self._state[3])
        self._beta = np.arcsin(self._state[4]/self._Va)



        Fg = MAV.mass*MAV.gravity*np.array([2*(e1*e3-e2*e0),
                                            2*(e2*e3 + e1*e0),
                                            e3**2+e0**2-e1**2-e2**2,
                                            ])

        M_e = 25
        sig = lambda a: (1+np.exp(-M_e*(a-MAV.alpha0))+np.exp(M_e*(a+MAV.alpha0)))/((1+np.exp(-M_e*(a-MAV.alpha0)))*(1+np.exp(M_e*(a+MAV.alpha0))))
        cla = lambda a: (1-sig(a))*(MAV.C_L_0+MAV.C_L_alpha*a)+sig(a)*(2*np.sign(a)*np.sin(a)**2*np.cos(a))
        cda = lambda a: MAV.C_D_p + (MAV.C_L_0+MAV.C_L_alpha*a)**2/(np.pi*MAV.e*MAV.AR)

        cxa = lambda a: -(cda(a)) * np.cos(a) + (cla(a)) * np.sin(a)

        cxq = lambda a: -MAV.C_D_q * np.cos(a) + MAV.C_L_q * np.sin(a)

        cxde = lambda a: -MAV.C_D_delta_e * np.cos(a) + MAV.C_L_delta_e * np.sin(a)

        cza = lambda a: -(cda(a)) * np.sin(a) - (cla(a)) * np.cos(a)

        czq = lambda a: -MAV.C_D_q * np.sin(a) - MAV.C_L_q * np.cos(a)

        czde = lambda a: -MAV.C_D_delta_e * np.sin(a) - MAV.C_L_delta_e * np.cos(a)

        c = MAV.c/(2*self._Va)
        b = MAV.b/(2*self._Va)


        Fa = 0.5*MAV.rho*self._Va**2*MAV.S_wing* (np.array([\
            [1,0,0],[0,1,0],[0,0,1]]) @ np.array([cxa(self._alpha)+cxq(self._alpha)*c*q+cxde(self._alpha)*de,
            MAV.C_Y_0+MAV.C_Y_beta*self._beta+MAV.C_Y_p*b*p+MAV.C_Y_r*b*r+MAV.C_Y_delta_a*da+MAV.C_Y_delta_r*dr,
            cza(self._alpha)+czq(self._alpha)*c*q+czde(self._alpha)*de,
            ]))


        F = Fg + Fa
        #
        # print("Fa:",Fa)

        Fp = 0.5*MAV.rho*MAV.S_prop*MAV.C_prop*((MAV.k_motor*dt)**2-self._Va**2)


        fx = F[0] + Fp
        fy = F[1]
        fz = F[2]

        #  Moment time!!!
        Ma = 0.5*MAV.rho*self._Va**2*MAV.S_wing*np.array([\
            MAV.b*(MAV.C_ell_0+MAV.C_ell_beta*self._beta+MAV.C_ell_p*b*p+MAV.C_ell_r*b*r+MAV.C_ell_delta_a*da+MAV.C_ell_delta_r*dr),
            MAV.c*(MAV.C_m_0+(MAV.C_m_alpha*self._alpha)+(MAV.C_m_q*c*q)+(MAV.C_m_delta_e*de)),
            MAV.b*(MAV.C_n_0+(MAV.C_n_beta*self._beta)+(MAV.C_n_p*b*p)+(MAV.C_n_r*b*r)+(MAV.C_n_delta_a*da)+(MAV.C_n_delta_r*dr))
            ])
        # print("\nMa:", Ma)

        size = 1
        if delta.ndim == 2:
            size = delta.shape[1];

        Mp = np.array([])
        if size == 1:
            Mp = np.array([[-MAV.kTp*(MAV.kOmega*dt)**2],
                           [0.0],
                           [0.0]
                           ])
        else:
            Mp = np.array([-MAV.kTp*(MAV.kOmega*dt)**2,
                           np.zeros(size),
                           np.zeros(size)
                           ])

        M = Mp + Ma

        Mx = M[0]
        My = M[1]
        Mz = M[2]

        # self._forces[0] = fx
        # self._forces[1] = fy
        # self._forces[2] = fz
        return np.array([fx, fy, fz, Mx, My, Mz]).T
