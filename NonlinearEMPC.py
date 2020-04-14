import numpy as np
import time
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

class NonlinearEMPC():

    def __init__(self,
                 model,
                 cost_function,
                 numStates,
                 numInputs,
                 umin,
                 umax,
                 horizon=50,
                 numSims=500,
                 numParents=10,
                 numStrangers=10,
                 numKnotPoints=3,
                 dt=.01):

        self.model = model
        self.cost_function = cost_function
        self.n = numStates
        self.m = numInputs
        self.umin = np.array(umin)
        self.umax = np.array(umax)
        self.uRange = np.array(umax)-np.array(umin)
        self.horizon = horizon
        self.numSims = numSims
        self.dt = dt
        self.numParents = numParents
        self.numStrangers = numStrangers
        self.numKnotPoints = numKnotPoints
        self.segmentLength = float(self.horizon)/(self.numKnotPoints-1)
        self.U = np.zeros([self.numKnotPoints,self.m,self.numSims])
        self.costs = np.zeros([1,self.numSims])
        self.X = np.zeros([self.horizon,self.n,self.numSims])
        self.warmStart = False

    def get_random_u_trajectories(self,numSims):
        trajectories = np.multiply(np.random.rand(self.numKnotPoints,self.m,numSims),self.uRange[:, np.newaxis]) + self.umin[:, np.newaxis]
        return trajectories.clip(self.umin[:, np.newaxis],self.umax[:, np.newaxis])

    def mutate_child(self,child,mutation_noise):
        child = child + np.random.randn(self.numKnotPoints,child.shape[1])*mutation_noise
        return child.clip(self.umin,self.umax)

    def mate_parents(self,parent1,parent2):
        combinedParents = np.stack([parent1,parent2],axis=2)
        child = np.zeros(parent1.shape)
        for i in range(0,self.m):
            for j in range(0,self.numKnotPoints):
                child[j,i] = combinedParents[j,i,np.random.choice([0,1])]
        return child

    def mate_and_mutate_parents(self,parents,x,xgoal,ulast):

        self.U[:,:,0:self.numParents] = parents

        # Introduce random trajectories to discourage stagnation
        self.U[:,:,self.numParents:self.numParents+self.numStrangers] = self.get_random_u_trajectories(self.numStrangers)

        i = self.numParents + self.numStrangers

        while i<self.numSims:
            mate1 = np.random.choice(range(0,self.numParents))
            mate2 = np.random.choice(range(0,self.numParents))
            while mate1==mate2:
                mate2 = np.random.choice(range(0,self.numParents))

            self.U[:,:,i] = self.mate_parents(parents[:,:,mate1],parents[:,:,mate2])
            # self.U[:,:,i] = self.mutate_child(self.U[:,:,i],mutation_noise=self.uRange[0]/1000.0)
            i += 1

        return self.U

    def get_us_from_U(self,U,i,horizon,numKnotPoints):
        knot = int(i/self.segmentLength)
        # ui = U[knot,:,:]*(1.0-float(i)/self.segmentLength) + U[knot+1,:,:]*(float(i)/self.segmentLength)
        ui = U[knot,:,:]
        for i in range(0,self.m):
            ui[i,:] = ui[i,:].clip(self.umin[i],self.umax[i])

        return ui

    def get_costs_from_trajectories(self,x0,xgoal,ugoal,U):
        self.costs = np.zeros([self.numKnotPoints,self.m,self.numSims])
        self.X[0,:,:] = x0

        for i in range(0,self.horizon-1):
            ui = self.get_us_from_U(U,i,self.horizon,self.numKnotPoints)
            set_trace()
            self.X[i+1,:,:] = self.model(self.X[i,:,:],ui,self.dt)
            self.costs += self.cost_function(self.X[i,:,:],ui,xgoal,ugoal)

        FinalCosts = self.cost_function(self.X[self.horizon-1,:,:],ui,xgoal,ugoal,final_timestep=True)
        self.costs += FinalCosts

        return self.costs

    def select_parents(self,U,costs):
        indices = costs.argsort().flatten()
        self.U_parents = U[:,:,indices[0:self.numParents]]
        self.X_parents = self.X[:,:,indices[0:self.numParents]]
        return self.U_parents

    def get_next_u_from_parents(self,U_parents):
        return U_parents[0,:,0]
        # next_u = U_parents[0,:,0]*(1.0-1.0/self.segmentLength)+U_parents[1,:,0]*(1.0/self.segmentLength)
        # return next_u



    def solve_for_next_u(self,x0,xgoal,ulast,ugoal):
        if self.warmStart == False:
            # self.U = np.zeros((self.numParents, self.m, self.numSims))
            self.U = self.get_random_u_trajectories(self.numSims)
            # set_trace()
            # print(self.U)
            self.warmStart = True
        else:
            self.U = self.mate_and_mutate_parents(self.U_parents,x0,xgoal,ulast)

            # This constrains the first point in the u trajectory to be the last commanded u
            # for i in range(0,self.numSims):
            #     self.U[0,:,i] = ulast

        for i in range(100):
            print('Generation: ', i)
            self.costs = self.get_costs_from_trajectories(x0,xgoal,ugoal,self.U)
            self.U_parents = self.select_parents(self.U,self.costs)
            self.U = self.mate_and_mutate_parents(self.U_parents,x0,xgoal,ulast)

        self.costs = self.get_costs_from_trajectories(x0,xgoal,ugoal,self.U)
        self.U_parents = self.select_parents(self.U,self.costs)
        next_u = self.get_next_u_from_parents(self.U_parents)

        print(repr(self.U_parents[:,:,0].flatten()))
        # print(self.U_parents[:,:,0])
        # print(next_u)
        # set_trace()
        return next_u


if __name__=='__main__':

    # from LearnedInvertedPendulum import *
    from InvertedPendulum import *

    import time

    # learnedSys = LearnedInvertedPendulum(use_gpu=False)
    sys = InvertedPendulum(mass=.2)
    numStates = sys.numStates
    numInputs = sys.numInputs
    Q = 1.0*np.diag([0,1.0])
    Qf = 100.0*np.diag([0,1.0])
    R = 0.0*np.diag([1.0])

    # This cost function must take in vectors of shapes:
    # x: [numStates,numSims]
    # u: [numInputs,numSims]
    # and return costs in the shape:
    # cost: [1,numSims]
    def myCostFcn(x,u,xgoal,ugoal,final_timestep=False):
        cost = np.zeros(u.shape)
        if final_timestep:
            Qx = np.abs(Qf.dot(x-xgoal))
            for i in range(u.shape[0]):
                cost[i,:] = Qx[i+u.shape[0]]
        else:
            Qx = np.abs(Q.dot(x-xgoal))
            Ru = np.abs(R.dot(u-ugoal))
            # set_trace()
            for i in range(u.shape[0]):
                cost[i,:] = Ru[i] + Qx[i+u.shape[0]]
        return cost

    controller = NonlinearEMPC(sys.forward_simulate_dt,
                               myCostFcn,
                               numStates,
                               numInputs,
                               umin=[-sys.uMax],
                               umax=[sys.uMax])

    x = np.array([0,-np.pi]).reshape(sys.numStates,1)
    u = np.zeros([1,1])
    xgoal = np.array([0,0.]).reshape(sys.numStates,1)

    fig = plt.figure(10)
    plt.ion()

    for i in range(0,1000):
        start = time.time()
        u = controller.solve_for_next_u(x,xgoal,ulast=u,ugoal=u*0)
        x = sys.forward_simulate_dt(x,u,.01)
        end = time.time()

        plt.figure(10)
        sys.visualize(x)
        # print("x: ",x)
        # print("u: ",u)
        print(end-start, i)
