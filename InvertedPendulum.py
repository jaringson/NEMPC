import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class InvertedPendulum():

    def __init__(self,length=1.0,mass=0.2,damping=0.1,gravity=9.81):
        self.numStates = 2
        self.numInputs = 1

        # self.uMax = np.inf
        self.uMax = 1.0

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.I = self.m*self.l**2.0

    def forward_simulate_dt(self,x,u,dt=.01,wrapAngle=True):
        x = deepcopy(x)
        # u = deepcopy(u.clip(-self.uMax,self.uMax))
        u = deepcopy(u)
        x = x.reshape([self.numStates,-1])
        xdot = np.zeros(x.shape)
        xdot[0,:] = (-self.b*x[0,:] + self.m*self.g*np.sin(x[1,:]) + u)/self.I
        xdot[1,:] = x[0,:]
        # pdb.set_trace()
        x = x + xdot*dt
        if wrapAngle==True:
            x[1,:] = (x[1,:] + np.pi) % (2*np.pi) - np.pi
        return x

    def calc_discrete_A_B_w(self,x,u,dt=.01):
        x = deepcopy(x)
        u = deepcopy(u)
        x = x.reshape([self.numStates,-1])
        A = np.matrix([[-self.b/self.I, 0],
                       [1.0, 0]])
        B = np.matrix([[1.0/self.I],
                       [0.0]])
        w = np.matrix([self.m*self.g*np.sin(x[1,:])/self.I,
                       [0.0]])

        [Ad,Bd] = self.discretize_A_and_B(A,B,dt)
        wd = w*dt

        return Ad,Bd,wd


    def visualize(self,x):
        CoM = [-0.5*np.sin(x[1]),0.5*np.cos(x[1])]
        theta = x[1]

        x = [CoM[0] + self.l/2.0*np.sin(theta),CoM[0] - self.l/2.0*np.sin(theta)]
        y = [CoM[1] - self.l/2.0*np.cos(theta),CoM[1] + self.l/2.0*np.cos(theta)]

        massX = CoM[0] - self.l/2.0*np.sin(theta)
        massY = CoM[1] + self.l/2.0*np.cos(theta)

        plt.clf()
        plt.plot(x,y)
        plt.scatter(massX,massY,50,'r')
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.ion()
        plt.show()
        plt.pause(.0000001)
