import flavorstuff as fs
import numpy as np
import sympy as sp

def get_F_down_simple(u_L, d_L, u_R, d_R, gs):
    F_ij = np.array([0, gs[0]*u_R, gs[1]*d_L, 0],
                    [-gs[2]*u_R, 0, gs[3]*u_L, 0],
                    [gs[4]*d_L, gs[5]*u_L, 0, gs[6]*u_R,
                     [0,0, gs[7]*u_R, gs[8]*d_R]])
    return F_ij

def get_F_up_simple(u_L, d_L, u_R, d_R, gs):
    F_ij = np.array([gs[0]*u_R,0, gs[1]*u_L, 0],
                    [ 0, -gs[2]*u_R, gs[3]*d_L, 0],
                    [0,0,gs[4]*u_R, gs[5]*d_R],
                    [[gs[6]*d_L,0, gs[7]*u_L, gs[8]*u_R]])
    return F_ij

def get_omega(theta):
    omega = np.array([[0, np.cos(theta), 0 , np.sin(theta)],
                      [-np.cos(theta), 0, -np.sin(theta), 0],
                      [0, np.sin(theta), 0, -np.cos(theta)],
                      [-np.sin(theta), 0, np.cos(theta), 0]])
    return omega

def sym_term_one(F_down, y, omega, mu_1, mu_5 ,mu_10):
    a = np.trace(F_down @ omega.T) * np.trace(F_down @ omega.T)
    b = np.trace((F_down.T @ omega) @ (F_down @ omega.T))
    c = -np.trace((F_down @ omega) @ (F_down @ omega.T))
    d = -0.5*a
    e = b
    f = c
    return y**2*(mu_1*a + mu_5*(b+c+d) + mu_10*(e+f))

def sym_term_two(F_down, F_up, y, omega, mu_1, mu_5, mu_10):
    a = np.trace(F_down @ omega.T) * np.trace(F_up)
    return

def sym_term_three():
    pass

def sym_lagrangian_simple(gs, y, mus, theta):
    u_R,d_L,u_L,d_R = sp.symbols("u_R d_L u_L d_R", real = True)
    F_down = get_F_down_simple(u_L,d_L,u_R, d_R, gs)
    F_up = get_F_up_simple(u_L,d_L,u_R, d_R, gs)
    omega = get_omega(theta)