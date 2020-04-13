import numpy as np
from dynamics import Dynamics

class CostFunctor:
    def __init__(self, use_penalty=False, return_states=False):
        self.dyn = Dynamics(dt=0.02)
        self.Q = np.array([10,10,100,2.5,2.5,2,9.8,9.8,10])
        self.Qf = np.array([10,10,100,4.5,4.5,5,0.8,0.8,1])
        self.x0 = np.array([0,0,-5.,0,0,0,0,0,0])
        self.x_des = self.x0.copy()
        self.use_penalty = use_penalty
        self.initialized = False
        self.return_states = return_states
        w_max = np.pi / 2
        self.u_max = np.array([1, w_max, w_max, w_max])
        self.u_min = np.array([0,-w_max,-w_max,-w_max])
        self.mu = 10.
    
    def __call__(self, z):
        if z.shape[0] == z.size:
            Tm = z.size
            pop_size = 1
        else:
            pop_size, Tm = z.shape # default to use with nempc
        horizon = Tm // 4
        xk = np.tile(self.x0[:,None], pop_size)
        cost = np.zeros(pop_size)
        u_traj = z.reshape(pop_size,horizon,4)
        if self.return_states:
            x_traj = []
        for k in range(horizon):
            xk = self.dyn.rk4(xk, u_traj[:,k].T)
            x_err = xk - self.x_des[:,None]
            if k < horizon-1:
                cost += np.sum(x_err**2 * self.Q[:,None], axis=0)
            else:
                cost += np.sum(x_err**2 * self.Qf[:,None], axis=0)

        # penalty is only used to add cost for bound constraint violations
        # when testing with gradient-based optimizer
        penalty = 0
        if self.use_penalty:
            cost = cost.item()
            if not self.initialized:
                self.z_max = np.tile(self.u_max, horizon)
                self.z_min = np.tile(self.u_min, horizon)
                self.initialized = True
            upper_bound = z - self.z_max
            mask = upper_bound > 0
            penalty += np.sum(upper_bound[mask]**2)

            lower_bound = self.z_min - z
            mask = lower_bound > 0
            penalty += np.sum(lower_bound[mask]**2)

            penalty *= self.mu/2 

        if self.return_states:
            x_traj = np.array(x_traj).flatten()
            return x_traj, cost + penalty

        return cost + penalty

