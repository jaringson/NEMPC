import numpy as np

from IPython.core.debugger import set_trace

def lhssample(n,p):
    x = np.random.uniform(size=[n,p])
    for i in range(p):
        x[:,i] = (np.argsort(x[:,i])+0.5)/n
    return x

class NEMPC:
    def __init__(self, cost_fn, num_states, num_inputs, u_min, u_max, u_eq,
            horizon=10, slew_rate=0.15, population_size=500, num_gens=5,
            num_parents=20, warm_start=False, mode='tournament'):
        self.cost_fn = cost_fn
        self.n, self.m = num_states, num_inputs
        self.u_min, self.u_max = u_min, u_max
        self.u_range = u_max - u_min
        self.u_eq = u_eq
        self.T = horizon
        # only allow inputs to change by +/- slew_rate % of input range from one
        # time step to the next
        self.slew = self.u_range * slew_rate
        self.pop_size = population_size
        self.costs = np.zeros(self.pop_size)
        self.num_gens = num_gens
        self.num_parents = num_parents
        self.warm_start = warm_start
        self.U = np.empty([self.pop_size, self.T*self.m])
        self.initialized = False
        self.idxs = np.arange(self.pop_size)
        self.mode = mode
        # store some data
        self.cost_hist = []

    def solve(self, x_des):
        # create initial population
        # if using warm_start then the previous population is used to start
        if not self.initialized or not self.warm_start:
            self.U[0] = np.tile(self.u_eq, self.T) # add equilibrium to pop
            self.U[1:] = self.createRandomUTrajectories(self.pop_size-1)
            if self.warm_start:
                self.num_gens = 1 
            self.initialized = True

        # run specified number of generations
        self.cost_fn.x_des = x_des
        for g in range(self.num_gens):
            self.runGeneration()

        # recompute fitness and return the optimum
        self.calcFitness()
        idx = np.argmin(self.costs)
        best_trajectory = self.U[idx]
        return best_trajectory

    def createRandomUTrajectories(self, num):
        ###### Completely random initialization
        U_trajectories = np.tile(self.u_min, self.T) + np.tile(self.u_range,
                self.T)*lhssample(num, self.m*self.T)
        
        ###### Slew rate initialization
        # U_trajectories = np.empty([num, self.T*self.m])
        # ## Uniform random initialization
        # # uk = np.random.uniform(self.u_min, self.u_max, [num,self.m])
        # ## Latin Hypercube Sampling (LHS) initialization
        # uk = self.u_min + self.u_range*lhssample(num, self.m)
        # U_trajectories[:,:self.m] = uk
        # for k in range(1,self.T):
        #     uk = np.random.uniform(uk-self.slew, uk+self.slew)
        #     U_trajectories[:,self.m*k:self.m*(k+1)] = self.saturate(uk, [num,1])
        return U_trajectories

    def runGeneration(self):
        self.calcFitness()
        parents = self.selectParents()
        self.crossover(parents)
        self.addMutation()

    def calcFitness(self):
        self.costs = self.cost_fn(self.U)
        best_idx = np.argmin(self.costs)

        ########### Hack for Quadrotor
        # best_pos_error = np.abs(self.cost_fn.x0 - self.cost_fn.x_des)
        # if best_pos_error[0] < 1e-10:
        #     # set_trace()
        #     test = self.U[best_idx].reshape(-1,self.m)
        #     test[:,2] = 0
        #     test = test.flatten()
        #     test_cost = self.cost_fn(test)
        #     if test_cost < self.costs[best_idx]:
        #         self.U[best_idx] = test
        #         self.costs[best_idx] = test_cost
        # if best_pos_error[1] < 1e-10:
        #     # set_trace()
        #     test = self.U[best_idx].reshape(-1,self.m)
        #     test[:,1] = 0
        #     test = test.flatten()
        #     test_cost = self.cost_fn(test)
        #     if test_cost < self.costs[best_idx]:
        #         self.U[best_idx] = test
        #         self.costs[best_idx] = test_cost
        # if best_pos_error[5] < 1e-10:
        #     # set_trace()
        #     test = self.U[best_idx].reshape(-1,self.m)
        #     test[:,3] = 0
        #     test = test.flatten()
        #     test_cost = self.cost_fn(test)
        #     if test_cost < self.costs[best_idx]:
        #         self.U[best_idx] = test
        #         self.costs[best_idx] = test_cost
            
        self.cost_hist.append(self.costs[best_idx])

    def selectParents(self):
        if self.mode == 'keep_best':
            best_trajectories = self.costs.argsort()[:self.num_parents]
            survivors = self.U[best_trajectories]
            # introduce random strangers to mating pool to prevent stagnation
            strangers = self.createRandomUTrajectories(self.num_parents)
            parents = np.block([[survivors],[strangers]])
        elif self.mode == 'tournament':
            # idxs = np.arange(self.pop_size)
            parents = np.empty(self.U.shape)
            for trial in range(2):
                np.random.shuffle(self.idxs)
                for p in range(self.pop_size//2):
                    i,j = self.idxs[2*p:2*(p+1)]
                    if self.costs[i] < self.costs[j]:
                        parent_idx = i
                    else:
                        parent_idx = j
                    parents[trial*self.pop_size//2+p] = self.U[parent_idx]

        return parents

    def crossover(self, mating_pool):
        if self.mode == 'keep_best':
            num = self.num_parents*2
            # keep parents in case no child is better
            self.U[:num] = mating_pool

            # randomly mate from mating pool
            repeat = (self.pop_size - num) // self.num_parents
            for r in range(1,repeat+1):
                batch = np.empty([self.num_parents, self.T*self.m])
                idxs = np.arange(num)
                np.random.shuffle(idxs)
                for child in range(self.num_parents):
                    i,j = idxs[2*child:2*(child+1)]
                    batch[child] = self.mateParents(mating_pool[i], mating_pool[j])
                self.U[self.num_parents*r:self.num_parents*(r+1)] = batch
            
            # in case num doesn't divide evenly into pop_size
            remainder = (self.pop_size - num) % self.num_parents
            for r in range(1, remainder+1):
                i,j = np.random.choice(range(0,num), 2, False)
                self.U[-r] = self.mateParents(mating_pool[i], mating_pool[j])
        elif self.mode == 'tournament':
            best_trajectories = self.U[self.costs.argsort()[:self.num_parents]]
            np.random.shuffle(self.idxs)
            for p in range(self.pop_size//2):
                i,j = self.idxs[2*p:2*(p+1)]
                if self.costs[i] < self.costs[j]:
                    won = i
                    lost = j
                else:
                    won = j
                    lost = i
                # linear crossover
                child1 = (mating_pool[i] + mating_pool[j]) / 2
                child2 = (2*mating_pool[won] - mating_pool[lost])
                child2 = self.saturate(child2, self.T)

                self.U[2*p] = child1
                self.U[2*p+1] = child2

            # overwrite first few children with best parents
            self.U[:self.num_parents] = best_trajectories

    def addMutation(self):
        num_p = self.num_parents
        ####### Apply mutation to each variable individually
        # r,c = self.U.shape
        # mask = np.random.rand(r-num_p, c) < 0.05
        # mutation = np.zeros(mask.shape)
        # mutation[mask] += np.random.randn(*mutation[mask].shape)*0.02
        # self.U[num_p:] += mutation
        # self.U = self.saturate(self.U, [self.pop_size,self.T])

        ####### Apply mutation to whole member of population
        mask = np.random.rand(self.pop_size-num_p) < 0.05
        mutation = np.random.randn(*self.U[num_p:][mask].shape)*0.02
        self.U[num_p:][mask] += mutation
        self.U = self.saturate(self.U, [self.pop_size,self.T])
        # for i in idxs:
        #     mutation = self.U[i] + np.random.randn(self.T*self.m)*0.02
        #     mutation = self.saturate(mutation.reshape(self.T,self.m))
        #     self.U[i] = mutation.flatten()

    def mateParents(self, parent1, parent2):
        child = np.empty(self.T*self.m)
        for gene in range(self.T*self.m):
            idx = np.random.choice([1,2])
            if idx == 1:
                child[gene] = parent1[gene]
            else:
                child[gene] = parent2[gene]
        return child 

    def saturate(self, u, tile_shape):
        lim = np.tile(self.u_max, tile_shape)
        mask = u > lim
        u[mask] = lim[mask]
        lim = np.tile(self.u_min, tile_shape)
        mask = u < lim
        u[mask] = lim[mask]
        return u

