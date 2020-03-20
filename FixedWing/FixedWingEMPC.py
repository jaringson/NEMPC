import numpy as np
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from NonlinearEMPC_OLD import NonlinearEMPC
from LearnedFixedWing import *
from FixedWing import *
import pdb
from tools import Quaternion2Euler
from tools import Euler2Quaternion
from tools import normalize

from tools import boxminus


def follow_orbit(state):

        # p = np.array([state[0], state[1], state[2]]).T
        g = 9.8
        Vg = state[3]


        # psi = state.psi

        phi,theta,psi = Quaternion2Euler(normalize(state[6:10]))
        chi = psi

        k_orbit = 0.75

        c = np.array([0, 0, -10])
        rho = 1.5
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
        line_dir = R_90 @ normalize(pos_noZ)
        line_err = state[0:3].T - closest_to_cir

        chi_l = np.arctan2(line_dir[1], line_dir[0])

        gamma_l = np.arctan(-line_dir[2]/ np.sum(np.abs(line_dir[0:2])**2,axis=0)**(1./2))
        gamma = theta

        R_I2l = np.identity(3)
        R_I2l[0,0] = np.cos(chi_l)
        R_I2l[0,1] = np.sin(chi_l)
        R_I2l[1,0] = -np.sin(chi_l)
        R_I2l[1,1] = np.cos(chi_l)

        e3 = np.array([0,0,1])

        k_path = 0.5
        gamma_ref = gamma_l + 0.2618 * 2.0 / np.pi * np.arctan(k_path * e3.dot(R_I2l @ line_err.T))

        # return normalize(Euler2Quaternion(0.0*np.ones(state.shape[1]),
        #     0.0*np.ones(state.shape[1]),
        #     chi_c*np.ones(state.shape[1]),
        #     state.shape[1]))
        # pdb.set_trace()
        return normalize(Euler2Quaternion(phi_feedforward*np.ones(state.shape[1]),
            gamma_ref[0]*np.ones(state.shape[1]),
            chi_c*np.ones(state.shape[1]),
            state.shape[1]))

if __name__=='__main__':

    learnedSys = LearnedFixedWing()
    sys = FixedWing()
    sys2 = FixedWing()
    numStates = sys.numStates
    numInputs = sys.numInputs

    Q = 0.01*np.diag([0,0,0.0, 0,0,0, 15.5,15.5,15.5, 0,0,0])
    Qf = 0.05*np.diag([0,0,0.0, 0,0,0, 15.5,15.5,15.5, 0,0,0])
    R = .0*np.diag([1.0,1.0,1.0,1.0])

    def myCostFcn(x,u,xgoal,ugoal,final_timestep=False):
        xgoal[6:10] = follow_orbit(x)
        cost = np.zeros(u.shape)
        Qx = np.zeros((x.shape[0]-1,x.shape[1]))
        Ru = np.zeros(u.shape)
        if final_timestep:
            Qx[0:6] = np.abs(Qf[0:6,0:6].dot(x[0:6]-xgoal[0:6]))
            Qx[6:9] = np.abs(Qf[6:9,6:9].dot(boxminus(x[6:10],xgoal[6:10])))
            Qx[9:] = np.abs(Qf[9:,9:].dot(x[10:]-xgoal[10:]))
            # for i in range(u.shape[0]):
            #     cost[i,:] = Qx[i+u.shape[0]]
        else:
            # Qx = np.abs(Q.dot(x-xgoal))
            # Ru = np.abs(R.dot(u-ugoal))
            Qx[0:6] = np.abs(Q[0:6,0:6].dot(x[0:6]-xgoal[0:6]))
            Qx[6:9] = np.abs(Q[6:9,6:9].dot(boxminus(x[6:10],xgoal[6:10])))
            Qx[9:] = np.abs(Q[9:,9:].dot(x[10:]-xgoal[10:]))
            # for i in range(u.shape[0]):
            #     cost[i,:] = Ru[i] + Qx[i+u.shape[0]]

        # pdb.set_trace()
        cost[1,:] = np.sum(Qx,axis=0) #Qx[2] #+ np.sum(Qx[6:10],axis=0)
        cost[2,:] = np.sum(Qx,axis=0) #Qx[2] #+ np.sum(Qx[6:10],axis=0)
        cost[3,:] = np.sum(Qx,axis=0) #np.sum(Qx,axis=0) #Qx[2] + np.sum(Qx[6:10],axis=0)
        # print(np.average(Qx[2]))
        return cost

    umin_ = [-sys.uMax + 0.25]*sys.numInputs
    umin_[2] = 0 + 0.1

    umax_ = [sys.uMax - 0.25]*sys.numInputs
    umax_[2] = 1 - 0.1

    print(umin_,umax_)

    numSims = 2000
    controller = NonlinearEMPC(learnedSys.forward_simulate_dt,
                               myCostFcn,
                               numStates,
                               numInputs,
                               horizon = 10,
                               numSims = numSims,
                               numParents = 10,
                               umin=umin_,
                               umax=umax_)

    x = np.zeros([sys.numStates,1])
    x[0] = 2
    x[1] = 0
    x[2] = -9
    x[3] = 3
    x[6:10] = normalize(Euler2Quaternion(0,0,-np.pi/2,1))

    x2 = deepcopy(x)


    #np.array([0,-np.pi]).reshape(sys.numStates,1)
    u = np.zeros([sys.numInputs,1])
    xgoal = np.zeros([sys.numStates,numSims])
    xgoal[0] = 0
    xgoal[1] = 0
    xgoal[2] = -10
    xgoal[3] = 3
    xgoal[6:10] = Euler2Quaternion(0,0,0,1)

    fig = plt.figure(10)
    plt.ion()
    ax = fig.add_subplot(111, projection='3d')

    plt.pause(0.1)

    horizon = 200
    x_hist = np.zeros([sys.numStates,horizon+1])
    x2_hist = np.zeros([sys.numStates,horizon+1])
    u_hist = np.zeros([sys.numInputs,horizon])

    x_hist[:,0] = x.flatten()
    x2_hist[:,0] = x2.flatten()

    for i in range(0,horizon):
        start = time.time()
        x[6:10] = normalize(x[6:10])
        e = follow_orbit(x)
        xgoal[6] = e[0]*np.ones(numSims)
        xgoal[7] = e[1]*np.ones(numSims)
        xgoal[8] = e[2]*np.ones(numSims)
        xgoal[9] = e[3]*np.ones(numSims)
        # if i > 600:
        #     pdb.set_trace()
        u = controller.solve_for_next_u(x,xgoal,ulast=u,ugoal=np.zeros([sys.numInputs,1]))
        # u = 2*np.random.rand(4)-1
        # if u[2] < 0:
        #     u[2] = 0
        u[0] = 0
        # u[1] = -1
        # u[2] = 1
        # u[3] = 0
        print("solve time: ", i, time.time()-start)
        x = sys.forward_simulate_dt(x,u,.01)
        # out = learnedSys.forward_simulate_dt(x2,u)
        # x2 = deepcopy(out)
        # x2[6:10] = normalize(x2[6:10])

        # pdb.set_trace()
        # x2[0:3] = x2[0:3] + deepcopy(out[0:3])
        # x2[3:] = deepcopy(out[3:])

        # plt.figure(10)
        sys.visualize(x,ax)
        # sys.visualize(x2,ax,color='blue')

        u_hist[:,i] = u.flatten()
        x_hist[:,i+1] = x.flatten()
        # x2_hist[:,i+1] = x2.flatten()

    fig = plt.figure(4)
    ax = fig.add_subplot(411)
    plt.plot(x_hist[1,:],x_hist[0,:])

    rho = 1.5
    x_t = np.linspace(-rho, rho, 50)
    y_t = np.sqrt(rho**2 - x_t**2)
    x_t = np.append(x_t,np.linspace(rho, -rho, 50))
    # print(y_t.shape)
    y_t = np.append(y_t,-y_t)

    plt.plot(x_t,y_t)
    # plt.plot(x2_hist[1,:],x2_hist[0,:])
    ax = fig.add_subplot(413)
    plt.plot(x_hist[0,:])
    # plt.plot(x2_hist[0,:])
    ax = fig.add_subplot(412)
    plt.plot(x_hist[1,:])
    # plt.plot(x2_hist[1,:])
    ax = fig.add_subplot(414)
    plt.plot(x_hist[2,:])
    # plt.plot(x2_hist[2,:])

    plt.xlabel('Timestep')
    plt.ylabel('Theta (rad)')
    # plt.title('Nonlinear EMPC Position')
    plt.show()

    plt.figure(5)
    plt.plot(u_hist[0], label='Ael')
    plt.plot(u_hist[1], label='Evl')
    plt.plot(u_hist[2], label='Thr')
    plt.plot(u_hist[3], label='Rud')
    plt.show()
    plt.xlabel('Timestep')
    plt.ylabel('Torque (Nm)')
    plt.title('Nonlinear EMPC Input')
    plt.legend()

    # data_dict = {'x':x_hist,
    #              'u':u_hist}
    # import scipy.io as sio
    # sio.savemat('InvertedPendulumEMPCTrajectory.mat',data_dict)


    plt.pause(1000)
    # plt.waitforbuttonpress()
