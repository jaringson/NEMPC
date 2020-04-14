import numpy as np
import math
from IPython.core.debugger import set_trace
import warnings

def boxminus(q1,q2):
    # q2 = q2

    q2[1:] = -q2[1:]
    q1w = q1[0]
    q1x = q1[1]
    q1y = q1[2]
    q1z = q1[3]
    q2w = q2[0]
    q2x = q2[1]
    q2y = q2[2]
    q2z = q2[3]
    dq = np.zeros_like(q1)
    dq[0] = q2w*q1w - q2x*q1x - q2y*q1y - q2z*q1z
    dq[1] = q2w*q1x + q2x*q1w + q2y*q1z - q2z*q1y
    dq[2] = q2w*q1y - q2x*q1z + q2y*q1w + q2z*q1x
    dq[3] = q2w*q1z + q2x*q1y - q2y*q1x + q2z*q1w

    # print("dq",dq)
    dq[0][dq[0] < 0] = -1.0*dq[0][dq[0] < 0]

    dqw = dq[0]
    dqv = dq[1:]
    normV = np.sum(np.abs(dqv)**2,axis=0)**(1./2)
    out = np.zeros((3,q1.shape[1]))
    out2 = 2.0*np.arctan2(normV,dqw)*dqv/normV

    out[:,normV > 1e-8] = out2[:,normV > 1e-8]
    # set_trace()


    return out

def wrapAngle(angle, amt=np.pi):

    out = deepcopy(angle)
    while any(out < -amt):
        out = out + 2*np.pi * (out < -amt).astype(int)
    while any(out > amt):
        out = out - 2*np.pi * (out > amt).astype(int)
    return out

def normalize(v,axis=0):
    norm = np.sum(np.abs(v)**2,axis=axis)**(1./2)
    # if all(norm == 0:
    #    return v
    return v / norm

def sat(value, max=1, min=-1):
    if value < min:
        return min
    elif value > max:
        return max
    return value


def Quaternion2Rotation(quat):
    pdb.set_trace()
    e0 = quat[0]
    e1 = quat[1]
    e2 = quat[2]
    e3 = quat[3]

    R = np.array([[e1**2+e0**2-e2**2-e3**2,2*(e1*e2-e3*e0),2*(e1*e3+e2*e0)],
                  [2*(e1*e2+e3*e0),e2**2+e0**2-e1**2-e3**2,2*(e2*e3-e1*e0)],
                  [2*(e1*e3-e2*e0),2*(e2*e3+e1*e0),e3**2+e0**2-e1**2-e2**2]])

    return R

def Quaternion2Euler(quat):
    quat = normalize(quat)
    e0 = quat[0]
    ex = quat[1]
    ey = quat[2]
    ez = quat[3]

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:

            phi = np.arctan2(2 * (e0*ex + ey*ez), e0**2 + ez**2 - ex**2 - ey**2) # phi
            theta = np.arcsin(np.clip((2 * (e0*ey - ex*ez)),-1,1)) # theta
            psi = np.arctan2(2*(e0*ez + ex*ey), e0**2 + ex**2 - ey**2 - ez**2) # psi
        except Warning as e:
            pdb.set_trace()
            print(e)


    return phi, theta, psi

def Euler2Quaternion(phi, theta, psi, size):
    c_phi2 = np.cos(phi/2.0)
    s_phi2 = np.sin(phi/2.0)
    c_theta2 = np.cos(theta/2.0)
    s_theta2 = np.sin(theta/2.0)
    c_psi2 = np.cos(psi/2.0)
    s_psi2 = np.sin(psi/2.0)


    quat = np.empty((4,size))
    quat[0] = c_psi2 * c_theta2 * c_phi2 + s_psi2 * s_theta2 * s_phi2  # e0
    quat[1] = c_psi2 * c_theta2 * s_phi2 - s_psi2 * s_theta2 * c_phi2  # ex
    quat[2] = c_psi2 * s_theta2 * c_phi2 + s_psi2 * c_theta2 * s_phi2  # ey
    quat[3] = s_psi2 * c_theta2 * c_phi2 - c_psi2 * s_theta2 * s_phi2  # ez

    return quat

def Euler2Rotation(phi,theta,psi):
    q = Euler2Quaternion(phi,theta,psi)
    R = Quaternion2Rotation(q)

    return R
