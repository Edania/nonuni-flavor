import flavorstuff as fs
import numpy as np
import sympy as sp

# Returns the term F_ij
def get_F_down_simple(u_L, d_L, u_R, d_R, gs):
    F_ij = np.array([[0, gs[0]*u_R, gs[1]*d_L, 0],
                    [-gs[2]*u_R, 0, gs[3]*u_L, 0],
                    [gs[4]*d_L, gs[5]*u_L, 0, gs[6]*u_R],
                     [0,0, gs[7]*u_R, gs[8]*d_R]])
    return F_ij

# Returns the term F^i_j
def get_F_up_simple(u_L, d_L, u_R, d_R, gs):
    F_ij = np.array([[gs[0]*u_R,0, gs[1]*u_L, 0],
                    [ 0, -gs[2]*u_R, gs[3]*d_L, 0],
                    [0,0,gs[4]*u_R, gs[5]*d_R],
                    [gs[6]*d_L,0, gs[7]*u_L, gs[8]*u_R]])
    return F_ij

# Returns the Omega matrix depending on theta
def get_omega(theta):
    if type(theta) == sp.Symbol:
        omega = np.array([[0, sp.cos(theta), 0 , sp.sin(theta)],
                      [-sp.cos(theta), 0, -sp.sin(theta), 0],
                      [0, sp.sin(theta), 0, -sp.cos(theta)],
                      [-sp.sin(theta), 0, sp.cos(theta), 0]])
    else:   
        omega = np.array([[0, np.cos(theta), 0 , np.sin(theta)],
                      [-np.cos(theta), 0, -np.sin(theta), 0],
                      [0, np.sin(theta), 0, -np.cos(theta)],
                      [-np.sin(theta), 0, np.cos(theta), 0]])
    return omega

# Finds the first term of the Lagrangian (y^2 * (...))
def sym_term_one(F_down, y, omega, mu_1, mu_5 ,mu_10):
    a = np.trace(F_down @ omega.T) * np.trace(F_down @ omega.T)
    b = np.trace((F_down.T @ omega) @ (F_down @ omega.T))
    c = np.trace((F_down @ omega) @ (F_down @ omega.T))
    d = 0.5*a
    e = b
    f = c
    return y**2*(mu_1*a + mu_5*(b-c-d) + mu_10*(e+f))

# Finds the second term of the Lagrangian (yy* * (...))
def sym_term_two(F_down, F_up, y, omega, mu_1, mu_5, mu_10):
    a = np.trace(F_down @ omega.T) * np.trace(F_up)
    b = np.trace((F_up @ F_down) @ omega.T)
    c = np.trace((F_down @ F_up) @ omega.T)
    d = 0.5*a
    e = b
    f = c
    return y * y.conjugate() *(mu_1*a + mu_5*(b-c-d) + mu_10*(e+f))

# Finds the third term of the Lagrangian ((y*)^2 * (...))
def sym_term_three(F_up, y, omega, mu_1, mu_5, mu_10):
    a = np.trace(F_up)**2
    b = np.trace(F_up @ F_up)
    c = np.trace((omega @ F_up) @ (F_up @ omega))
    d = 0.5*a
    e = c
    f = b
    return y.conjugate()**2 * (mu_1*a + mu_5*(b-c-d) + mu_10*(e+f))

# Finds the entire Lagrangian
def sym_lagrangian_simple(u_R,d_L,u_L,d_R,gs_down, gs_up, y, mus, theta):
    F_down = get_F_down_simple(u_L,d_L,u_R, d_R, gs_down)
    F_up = get_F_up_simple(u_L,d_L,u_R, d_R, gs_up)
    omega = get_omega(theta)
    term_one = sym_term_one(F_down, y, omega, mus[0], mus[1], mus[2])
    print(f"term_one:{term_one}\n")
    term_two = sym_term_two(F_down, F_up, y, omega, mus[3], mus[4], mus[5])
    print(f"term_two:{term_two}\n")
    term_three = sym_term_one(F_up, y, omega, mus[6], mus[7], mus[8])
    print(f"term_three:{term_three}\n")
    lagrangian = term_one + term_two + term_three
    return lagrangian

if __name__ == "__main__":
    syms = True

    # Assigns symbolic variables to all inputs to the Lagrangian
    if syms:    
        [gu1, gQ1, gQ2, gu2, gu3, gd1, gu4,gQ3,gu5,gd2,gQ4,gu6] = sp.symbols(['gu1', 'gQ1', 'gQ2', 'gu2', 'gu3', 'gd1', 'gu4','gQ3','gu5','gd2','gQ4','gu6'])
        gs_down = [gu1, gQ1, gu1, gQ1, gQ2, gQ2, gu2, gu3, gd1]
        gs_up = [gu4, gQ3, gu4, gQ3, gu5, gd2, gQ4, gQ4, gu6]
        y = sp.symbols('y')
        #y = 1
        mus = sp.symbols(['mu1', 'mu5', 'mu10', 'mu1p', 'mu5p', 'mu10p', 'mu1pp', 'mu5pp', 'mu10pp'])
        #mus = np.ones(9)
        theta = sp.symbols('theta')
        #theta = np.pi/4
    # Assigns numerical values to all inputs (us and ds always symbolical)
    else:
        gs_down = np.ones(9)
        gs_up = np.ones(9)
        y = 1
        mus = np.ones(9)
        theta = np.pi/4


    u_R,d_L,u_L,d_R = sp.symbols("u_R d_L u_L d_R", real = True)
    lagrangian = sym_lagrangian_simple(u_R,d_L,u_L,d_R, gs_down,gs_up, y, mus, theta)
    #lagrangian = sp.separatevars(lagrangian)
    lagrangian_list = [sp.collect(lagrangian, y**2, evaluate = False), sp.collect(lagrangian, y*sp.conjugate(y), evaluate = False)]
    key_list = [y**2, y*sp.conjugate(y)]
    #lagrangian = lagrangian_dict[y]
    print(f"\nLagrangian: {lagrangian}\n")
    
    #lagrangian = lagrangian.subs(u_R*u_L, x)
    u_term = 0
    d_term = 0
    for i, lagrangian_dict in enumerate(lagrangian_list):
        lagrangian = sp.separatevars(lagrangian_dict[key_list[i]])
        print(f"{lagrangian} \n")
        u_terms = sp.collect(lagrangian, u_R*u_L, evaluate=False)
        d_terms = sp.collect(lagrangian, d_R*d_L, evaluate=False)
        u_term += key_list[i]*u_terms[u_R*u_L]
        d_term += key_list[i]*d_terms[d_R*d_L]
        #u_term = y*u_terms[u_R*u_L]
    #d_term = y*d_terms[d_R*d_L]
    #u_term = sp.collect_const(lagrangian, u_R)
    v_f = sp.symbols('v_f')
    gQ, gd, gu = sp.symbols('gQ ,gd, gu')
    u_term_simp = u_term.subs(sp.sin(theta), v_f)
    u_term_simp = u_term_simp.subs(sp.cos(theta), 1)
    u_term_simp = u_term_simp.subs([(gQ1, gQ), (gQ2, gQ), (gQ3, gQ), (gQ4, gQ)])
    u_term_simp = u_term_simp.subs([(gu1, gu), (gu2,gu),(gu3,gu),(gu4,gu),(gu5,gu),(gu6,gu)])
    d_term_simp = d_term.subs(sp.sin(theta), v_f)
    d_term_simp = d_term_simp.subs(sp.cos(theta), 1)
    d_term_simp = d_term_simp.subs([(gQ1, gQ), (gQ2, gQ), (gQ3, gQ), (gQ4, gQ)])
    d_term_simp = d_term_simp.subs([(gd1, gd), (gd2,gd)])
    # Check that all terms are linear in v_f
    u_term_simp = sp.collect(u_term_simp, v_f*gQ*gu)
    d_term_simp = sp.collect(d_term_simp, v_f*gQ*gd)
    #print(u_term.keys())
    print(f"\nu-terms: {u_term}")
    print(f'\nsimp u_terms: {u_term_simp}')
    print(f"\nd-terms: {d_term}")
    print(f'\nsimp d_terms: {d_term_simp}')
   