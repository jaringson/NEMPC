import numpy as np

from IPython.core.debugger import set_trace
def lhssample(n,p):
    x = np.random.uniform(size=[n,p])
    for i in range(p):
        x[:,i] = (np.argsort(x[:,i])+0.5)/n
    return x

class NEMPC:
    def __init__(self, cost_fn, num_states, num_inputs, u_min, u_max, 
            horizon=10, slew_rate=0.15, population_size=500, num_gens=3,
            num_parents=20):
        self.cost_fn = cost_fn
        self.n, self.m = num_states, num_inputs
        self.u_min, self.u_max = u_min, u_max
        self.u_range = u_max - u_min
        self.T = horizon
        # only allow inputs to change by +/- slew_rate % of input range from one
        # time step to the next
        self.slew = self.u_range * slew_rate
        self.pop_size = population_size
        self.costs = np.zeros(self.pop_size)
        self.num_gens = num_gens
        self.num_parents = num_parents
        # initialize population
        self.U = self.createRandomUTrajectories(self.pop_size)
        self.U[0] = np.tile([0.5,0,0,0], self.T) # add equilibrium to population
        self.idxs = np.arange(self.pop_size)
        # self.mode = 'keep_best'
        self.mode = 'tournament'
        # store some data
        self.cost_hist = []

    def solve(self, x_des):
        # set_trace()
        self.cost_fn.x_des = x_des
        for g in range(self.num_gens):
            self.runGeneration()
            # print(f'Gen {g}: lowest cost = {np.min(self.costs)}')
        self.calcFitness()
        idx = np.argmin(self.costs)
        best_trajectory = self.U[idx]
        return best_trajectory

    def createRandomUTrajectories(self, num):
        U_trajectories = np.zeros([num, self.T*self.m])
        # uk = np.random.uniform(self.u_min, self.u_max, [num,self.m])
        uk = self.u_min + self.u_range*lhssample(num, self.m)
        U_trajectories[:,:self.m] = uk
        for k in range(1,self.T):
            uk = np.random.uniform(uk-self.slew, uk+self.slew)
            U_trajectories[:,self.m*k:self.m*(k+1)] = self.saturate(uk)
        return U_trajectories

    def runGeneration(self):
        self.calcFitness()
        parents = self.selectParents()
        self.crossover(parents)
        self.addMutation()

    def calcFitness(self):
        self.costs = self.cost_fn(self.U)
        best_idx = np.argmin(self.costs)
        best_pos_error = np.abs(self.cost_fn.x0 - self.cost_fn.x_des)
        if best_pos_error[0] < 1e-10:
            # set_trace()
            test = self.U[best_idx].reshape(-1,self.m)
            test[:,2] = 0
            test = test.flatten()
            test_cost = self.cost_fn(test)
            if test_cost < self.costs[best_idx]:
                self.U[best_idx] = test
                self.costs[best_idx] = test_cost
        if best_pos_error[1] < 1e-10:
            # set_trace()
            test = self.U[best_idx].reshape(-1,self.m)
            test[:,1] = 0
            test = test.flatten()
            test_cost = self.cost_fn(test)
            if test_cost < self.costs[best_idx]:
                self.U[best_idx] = test
                self.costs[best_idx] = test_cost
        if best_pos_error[5] < 1e-10:
            # set_trace()
            test = self.U[best_idx].reshape(-1,self.m)
            test[:,3] = 0
            test = test.flatten()
            test_cost = self.cost_fn(test)
            if test_cost < self.costs[best_idx]:
                self.U[best_idx] = test
                self.costs[best_idx] = test_cost
            
        self.cost_hist.append(self.costs[best_idx])
        # print(self.U[best_idx,:self.m])

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
            # self.U[:self.pop_size//2] = mating_pool
            # idxs = np.arange(self.pop_size)
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
                child2 = self.saturate(child2.reshape(-1,self.m)).flatten()

                self.U[2*p] = child1
                self.U[2*p+1] = child2

            # overwrite first few children with best parents
            self.U[:self.num_parents] = best_trajectories


    def addMutation(self):
        idxs = np.where(np.random.rand(self.pop_size) < 0.05)[0]
        for i in idxs:
            mutation = self.U[i] + np.random.randn(self.T*self.m)*0.02
            mutation = self.saturate(mutation.reshape(self.T,self.m))
            self.U[i] = mutation.flatten()

    def mateParents(self, parent1, parent2):
        child = np.empty(self.T*self.m)
        for gene in range(self.T*self.m):
            idx = np.random.choice([1,2])
            if idx == 1:
                child[gene] = parent1[gene]
            else:
                child[gene] = parent2[gene]
        return child 

    def saturate(self, u):
        l = len(u)
        mask = u > self.u_max
        u[mask] = np.tile(self.u_max, [l,1])[mask]
        mask = u < self.u_min
        u[mask] = np.tile(self.u_min, [l,1])[mask]
        return u

    # def keepTopTrajectories(self):
    #     # keep a few of the best input trajectories
    #     best_trajectories = self.costs.argsort()[:self.num_parents]
    #     survivors = self.U[best_trajectories]
    #     # introduce random strangers to mating pool to prevent stagnation
    #     strangers = self.createRandomUTrajectories(self.num_parents)
    #     parents = np.block([[survivors],[strangers]])
    #     return parents

    # def tournament(self):
    #     idxs = np.arange(self.pop_size)
    #     np.random.shuffle(idxs)
    #     idxs = idxs.reshape(pop_size, 2)

