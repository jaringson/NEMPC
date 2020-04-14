import numpy as np

from IPython.core.debugger import set_trace
 
class Dynamics():
    def __init__(self, c_drag=0.1, gravity=9.81, throttle_eq=0.5, dt=0.02):
        self.cd = c_drag
        self.g = gravity
        self.se = throttle_eq
        self.dt = dt

    def rk4(self, x, u):
        k1 = self.f(x, u)
        k2 = self.f(x + k1*(self.dt/2), u)
        k3 = self.f(x + k2*(self.dt/2), u)
        k4 = self.f(x + k3*self.dt, u)
        x_next = x + (k1 + 2*(k2 + k3) + k4) * (self.dt/6)
        return x_next

    def f(self, x, u):
        phi,theta,psi = x[3:6]
        Cp, Sp = np.cos(phi), np.sin(phi)
        Ct, St = np.cos(theta), np.sin(theta)
        Cs, Ss = np.cos(psi), np.sin(psi)
        Tt = np.tan(theta)

        s,wx,wy,wz = u
        vx,vy,vz = x[6:9]

        # R = np.array([[Ct*Cs, Sp*St*Cs-Cp*Ss, Cp*St*Cs+Sp*Ss],
        #               [Ct*Ss, Sp*St*Ss+Cp*Cs, Cp*St*Ss-Sp*Cs],
        #               [-St,   Sp*Ct,          Cp*Ct]])
        
        xdot = np.empty(x.shape)
        xdot[0] = (Ct*Cs*vx) + (Sp*St*Cs-Cp*Ss)*vy + (Cp*St*Cs+Sp*Ss)*vz
        xdot[1] = (Ct*Ss*vx) + (Sp*St*Ss+Cp*Cs)*vy + (Cp*St*Ss-Sp*Cs)*vz
        xdot[2] = (-St*vx)   + (Sp*Ct)*vy          + (Cp*Ct)*vz
        # xdot[:3] = R @ x[6:9]
        xdot[3] = wx + Sp*Tt*wy + Cp*Tt*wz
        xdot[4] = Cp*wy-Sp*wz
        xdot[5] = (Sp*wy+Cp*wz) / Ct
        xdot[6] = (wz*vy-wy*vz)*1 - self.g*St - self.cd*vx
        xdot[7] = (wx*vz-wz*vx)*1 + self.g*Ct*Sp - self.cd*vy
        xdot[8] = (wy*vx-wx*vy)*1 + self.g*Ct*Cp - self.g*s/self.se

        return xdot
