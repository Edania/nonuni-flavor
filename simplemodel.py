#####################################################################################
# Module for model SU(3) x SU(2) x U^3_Y(1) x U^12_B-L(1) x U^2_I3R(1) x U^1_I3R(1) #
#####################################################################################
import flavorstuff as fs
import numpy as np
import GPy
import sympy as sp
import json
import matplotlib.pyplot as plt

from scipy.optimize import minimize, fsolve, least_squares
conts = fs.constants()

# Initiates constants for this particular model
def init_constants(unit = "GeV"):
    valid_units = {"GeV":1000, "TeV":1}
    try:
        s = valid_units[unit]
    except KeyError as e:
        print(f"Invalid key {unit}")
        exit(1)

    m_U = s*1
    m_D = s*1
    m_E = s*1

    v_chi = s*1
    v_phi = s*1
    v_sigma = s*10

    return m_U, m_D, m_E, v_chi, v_phi, v_sigma

# Function for minimize() that finds the gs that give the Z-boson mass
# There should be an exact analytical solution to this.
def closest_Z_mass(gs, mzs,g, vs):
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*vs[0]/(2), gs[0]*vs[0]/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3,              0,                  0],
                        [0, gs[0]*vs[1]/2, -gs[1]*vs[1],                0,                  0],
                        [0, gs[0]*vs[2]/2,              0,-gs[2]*vs[2]/2,                   0],
                        [0, 0,                          0, gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b) 
    return np.sqrt(Delta2[1:]) - mzs
    #return (np.sqrt(Delta2[1]) - mzs[0])**2 + np.sum(np.sqrt(Delta2[2:]) - mzs[1:])**2 + np.sqrt(np.abs(Delta2[0])) #np.abs(np.sqrt(Delta2[1]) - conts.m_Z)

 
    
# Builds the yukawa matrix as specified in the article
def build_yukawa(ys, ms, vs):
    #ys = np.ones(13)
    [v, v_chi, v_phi, v_sigma] = vs.flatten()
    m = np.zeros((3,3))
    m[2,2] = v*ys[0]

    Delta_L = v*np.array([[ys[1], ys[2], 0],[ys[3], ys[4],0], [0,0,0]])
    Delta_R = np.array([[0, ys[5]*v_phi, 0], [0, ys[6]*v_phi, ys[7]*v_chi], [ys[8]*v_sigma,0,0]])
    M = np.array([[ms[0], 0, ys[9]*v_phi], [0, ms[1],ys[10]*v_phi], [ys[11]*v_phi, ys[12]*v_phi, ms[2]]])

    yukawas = np.vstack((np.hstack((m, Delta_L)), np.hstack((Delta_R, M))))

    return yukawas

# function for finding yukawa mass basis via minimize()
def minimize_yukawas(ys, ms,Ms, vs):
   
    yukawas = build_yukawa(ys, [Ms[0],Ms[1],Ms[2]],vs)

    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)
    #diag_yukawa = diag_yukawa[::-1]

    return np.sum((diag_yukawa[:3]- ms)**2)

# builds the Q for this model, given a basis, eg Z^3, Z^3'...
def build_Q(field_type, basis):
    base_charge = [fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Y"), 
                   (fs.find_charge("fermions", field_type, "B") - fs.find_charge("fermions", field_type, "L"))/2,
                   fs.find_charge("fermions", field_type, "I3_R"), fs.find_charge("fermions", field_type, "I3_R")]
    charges = basis*base_charge
 
    # First three regular fermoins, last three new vector-like fermions. 
    Q = charges[0] + np.diag([charges[2] + charges[3], charges[2] + charges[4], charges[1], 
                                charges[2] + charges[3], charges[2] + charges[4], charges[1]])
    return Q

def sm_Q(field_type, basis):
    base_charge = [fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Y")]
    charges = basis*base_charge
    Q = np.diag([1,1,1])*np.sum(charges)
    return Q

def minimize_for_field(gs,ys, field, vs, Ms, ms, mzs, g, g_prim, V, Delta2):
    #gs = thetas[:4]
    #ys = thetas[4:]
  
    yukawas = build_yukawa(ys, Ms, vs)
    
    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)
    
    base = V[:,1]
    base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    
    sm_Q_d_L = sm_Q(f"{field}_L", base_SM)
    sm_Q_d_R = sm_Q(f"{field}_R", base_SM)

    Q_d_L = np.abs(np.diag(build_Q(f"{field}_L", base)))
    Q_d_R = np.abs(np.diag(build_Q(f"{field}_R", base)))

    mass_Q_d_L = np.abs(fs.mass_Q(U_L, Q_d_L))
    mass_Q_d_R = np.abs(fs.mass_Q(np.transpose(Uh_R), Q_d_R))
    
    min_QZ = np.sum((sm_Q_d_L-mass_Q_d_L[:3])**2 + (sm_Q_d_R-mass_Q_d_R[:3])**2)
    
    base = V[:,0]
    base_SM = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])
    
    sm_Q_d_L = sm_Q(f"{field}_L", base_SM)
    sm_Q_d_R = sm_Q(f"{field}_R", base_SM)

    Q_d_L = np.abs(np.diag(build_Q(f"{field}_L", base)))
    Q_d_R = np.abs(np.diag(build_Q(f"{field}_R", base)))

    mass_Q_d_L = np.abs(fs.mass_Q(U_L, Q_d_L))
    mass_Q_d_R = np.abs(fs.mass_Q(np.transpose(Uh_R), Q_d_R))
    
    min_Qgamma = np.sum((sm_Q_d_L-mass_Q_d_L[:3])**2 + (sm_Q_d_R-mass_Q_d_R[:3])**2)
    
    #min_Z = (np.sqrt(Delta2[1]) - mzs[0])**2
    min_Y = np.sum((diag_yukawa[:3]- ms)**2) + np.sum((diag_yukawa[3:]-Ms)**2)
    #print(f"min_Z = {min_Z}, min_Y = {min_Y}, min_Q = {min_Q}")
    return min_Y + min_QZ + min_Qgamma, U_L, diag_yukawa, Uh_R
    #c_sd = (mass_Q_d_L[0,1]**2)/Delta2[1]# + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]

def Q_compare(field, base, base_SM, U_L, Uh_R):
    sm_Q_d_L = sm_Q(f"{field}_L", base_SM)
    sm_Q_d_R = sm_Q(f"{field}_R", base_SM)

    Q_d_L = np.abs(np.diag(build_Q(f"{field}_L", base)))
    Q_d_R = np.abs(np.diag(build_Q(f"{field}_R", base)))

    mass_Q_d_L = np.abs(fs.mass_Q(U_L, Q_d_L))
    mass_Q_d_R = np.abs(fs.mass_Q(np.transpose(Uh_R), Q_d_R))
#    print(mass_Q_d_L.shape)
    compare_Q = np.append(np.abs(mass_Q_d_L[:3])-np.abs(sm_Q_d_L),np.abs(mass_Q_d_R[:3])-np.abs(sm_Q_d_R))
#    print(compare_Q.shape)
    return compare_Q


def solve_for_ys(ys,y3_u, y3_d,vs, g, g_prim, gs, M_Us, m_us, M_Ds, m_ds, mzs):
    ys_u = ys[:12]
    ys_d = ys[12:]
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*v_chi/6, gs[1]*v_chi/3, 0,0],
                        [0,gs[0]*v_chi/2, -gs[1]*v_chi, 0,0],
                        [0, gs[0]*v_phi/2,0,-gs[2]*v_phi/2,0],
                        [0,0,0,gs[2]*v_sigma/2, -gs[3]*v_sigma/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)    
    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    yukawas_u = build_yukawa(np.insert(ys_u, 0, y3_u), M_Us, vs)
    yukawas_d = build_yukawa(np.insert(ys_d, 0, y3_d), M_Ds, vs)

    U_u_L, diag_yukawa_u, Uh_u_R = fs.diag_yukawa(yukawas_u)
    U_d_L, diag_yukawa_d, Uh_d_R = fs.diag_yukawa(yukawas_d)

    m_u_compare = np.concatenate((diag_yukawa_u[:3] - m_us,diag_yukawa_u[3:] - M_Us))
    m_d_compare = np.concatenate((diag_yukawa_d[:3] - m_ds,diag_yukawa_d[3:] - M_Ds))

    base_SM_Z = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    base_SM_gamma = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])

    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R), Q_compare("u", V[:,0], base_SM_gamma, U_u_L, Uh_u_R),
                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R), Q_compare("d", V[:,0], base_SM_gamma, U_d_L, Uh_d_R)))
 #   print(compare_Qs.shape)
    
    V_CKM_calc = np.dot(np.transpose(U_u_L), U_d_L) 
    CKM_compare = (V_CKM_calc[:3,:3]-conts.V_CKM).flatten()
    #print(CKM_compare)
    #Scale according to order?
    compares = np.concatenate((CKM_compare, compare_Qs, m_u_compare, m_d_compare))

    return compares

def LCB(mean, var, beta):
    std_dev = np.sqrt(var)
    return -mean + beta * std_dev

def global_min_GP(model, train_points, beta,vs, g, g_prim, M_Us, m_us, M_Ds, m_ds, mzs, tol = 1e-3, max_iters = 100):
    # Bayesian opt.
    sampled_points = []
    sampled_minim = []
    iter = 0
    diff = 1
    previous_point = np.ones_like(train_points[0,:])
    while diff > tol:
        print(f"On iteration {iter}")
        test_points = np.random.uniform(0,2,np.shape(train_points))
        for point in range(train_points.shape[0]):
            res = minimize(minimize_for_CKM, test_points[point,:],args=(vs, g, g_prim, M_Us,m_us, M_Ds, m_ds, mzs),bounds= ((0,2),(0,2),(0,2),(0,2),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                    (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                        (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)))
            test_points[point,:] = res.x
        #for i in range(train_points):
        #    train_minima[i] = minimize_for_CKM(train_thetas[i],vs, g, g_prim, M_Us, m_us, M_Ds, m_ds, mzs)

        # Predict over the grid
        mean, var = model.predict(test_points)

        # Compute the acquisition function on grid.
        acquisition_values = LCB(mean, var, beta)
        
        # Select point with maximum acquisition value
        new_point_idx = np.argmax(acquisition_values)
        new_point = test_points[new_point_idx, :]
        diff = np.sqrt(np.sum((previous_point-new_point)**2))
        
        sampled_points.append(new_point)
    
        # Compute E at new point
        new_minim = minimize_for_CKM(new_point,vs, g, g_prim, M_Us, m_us, M_Ds, m_ds, mzs)

        sampled_minim.append(new_minim)
    
        # Update X,E and retrain model
        model.set_XY(np.vstack([model.X, new_point]), np.vstack([model.Y, new_minim]))
        model.optimize()

        iter += 1
        if iter == max_iters:
            break
        
        previous_point = new_point

    return iter, np.array(sampled_points), np.array(sampled_minim)

def get_g_models(search_for_gs = False):

    if search_for_gs:
        v_chis = np.arange(1000,10000,10)
        m_Z3s = np.arange(1000,10000,10)
        model_list = []
        tmp_list = []
        for v_chi in v_chis:
            # Conditions:
            for m_Z3 in m_Z3s: 
                v_phi = int(v_chi*np.random.uniform(1.5,5))
                v_sigma = 10*v_chi + np.random.randint(0,100)
                m_Z3prim = int(m_Z3*np.random.uniform(1.5,5))
                m_Z12 = m_Z3*10 + np.random.randint(0,100)
                # Let's do this fsolve for a range of ms and vs, with conditions on the grid :)
                gs = np.random.uniform(0,2,4)
                vs = np.array([conts.v_H, v_chi, v_phi, v_sigma])
                mzs = np.array([conts.m_Z, m_Z3, m_Z3prim, m_Z12])
                #res = gs_eq_system(*vs, *mzs)
                gs, infodict,ier,mesg = fsolve(closest_Z_mass, gs,args=(mzs, g, vs), full_output=True, factor = 0.1)
                diff = infodict["fvec"]
                if all(gs <= 2)  and all(gs > 0.01) and all(np.abs(diff) < 1e-2 ):
                    tmp_list = [mzs, vs, gs]
                    #tmp_dict = {"mzs":list(mzs), "vs":list(vs), "gs":list(gs)}
                    model_list.append(tmp_list)
                    print(f"Successful model")
                    print(f"gs: {gs}")
                    print(f"vs : {vs}")
                    print(f"mzs: {mzs}")
                    print(f"Diff m_Zs = {diff}")
        if model_list:
            np.savez("g_models.npz", *model_list)
    
    g_models = np.load("g_models.npz")
    model_list = [g_models[k] for k in g_models]
    print(f"The number of g_models is: {len(model_list)}")

    return model_list


if __name__ == "__main__":

    m_U, m_D, m_E, v_chi, v_phi, v_sigma = init_constants()
    g = 0.652
    g_prim = 0.357
    
    # Get models for gs
    search_for_gs = False
    plotting = False

    model_list = get_g_models()


    if plotting:
        v_chi_list = [model[1,1] for model in model_list]
        m_z3_list = [model[0,1] for model in model_list]

        fig = plt.figure()
        plt.scatter(v_chi_list, m_z3_list)
        plt.title("Correlation between v_chi and m_z3 for valid gs")
        plt.xlabel("v_chi")
        plt.ylabel("m_z3")
        plt.show()

    y3_u = conts.m_t/conts.v_H
    y3_d = conts.m_b/conts.v_H
    mzs = model_list[0][0,:]
    vs = model_list[0][1,:]
    gs = model_list[0][2,:]
    
    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    M_Ds = np.array([m_D, 1.5*m_D, 10*m_D])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
    M_Us = np.array([m_U, 1.5*m_U, 10*m_U])
    ys = np.ones(24)*0.05

    res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,vs,g,g_prim,gs,M_Us,m_us,M_Ds,m_ds,mzs), loss = "soft_l1")
    ys = res.x
    print(res.message)
    print(res.cost)
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3, 0,0],
                        [0,gs[0]*vs[1]/2, -gs[1]*vs[1], 0,0],
                        [0, gs[0]*vs[2]/2,0,-gs[2]*vs[2]/2,0],
                        [0,0,0,gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)

    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    y_ds = ys[12:]
    y_us = ys[:12]
    yukawas = build_yukawa(np.insert(y3_d, 0, y_ds), M_Ds, vs)

    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)
    print(f"mds diffs: {diag_yukawa[:3]-m_ds}, MDs diffs: {diag_yukawa[3:]-M_Ds}")
 #   print([diag_yukawa[0]-conts.m_d, diag_yukawa[1]-conts.m_s, diag_yukawa[2]-conts.m_b])
    yukawas = build_yukawa(np.insert(y3_u, 0, y_us), M_Us, vs)
    U_u_L, diag_yukawa, Uh_u_R = fs.diag_yukawa(yukawas)

    V_CKM_calc = np.dot(np.transpose(U_u_L), U_L) 
    CKM_compare = (V_CKM_calc[:3,:3]-conts.V_CKM).flatten()
    print(f"CKM compare: {CKM_compare}")
    base = V[:,1]
    base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    
    sm_Q_d_L = sm_Q("d_L", base_SM)

    Q_d_L = build_Q("d_L", base)
    Q_d_R = build_Q("d_R", base)

    #TODO: Compare diagonal charges with SM
    mass_Q_d_L = fs.mass_Q(U_L, Q_d_L)
    mass_Q_d_R = fs.mass_Q(np.transpose(Uh_R), Q_d_R)

    #TODO: CKM matrix!

    #print(mass_Q_d_L)
    #print(sm_Q_d_L)
    print(sm_Q_d_L)
    print(mass_Q_d_L)
    print(f"Charge difference: {np.diag(np.abs(mass_Q_d_L[:3])) - np.diag(np.abs(sm_Q_d_L))}")
    #print(mass_Q_d_R)

    #TODO: Calculate further constants, use in minimization?
    c_sd = (mass_Q_d_L[0,1]**2)/Delta2[1]# + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]
    print(f"The constant for a kaon-antikaon with neutral current is: {c_sd}")

    # Get models for ys (depends on gs)
#        print(gs)
#        print(vs)

'''
    for i in range(5):
        print(f"m = {np.sqrt(np.abs(Delta2[i]))}")
        print(f"V_{i} = {V[:,i]}\n")

    print(f"real m_Z = {conts.m_Z}. Found m_Z = {np.sqrt(Delta2[1])}")
    print(f"Diff m_Zs = {mzs-np.sqrt(Delta2[1:])}")


   
    print(f"ys:{ys}")
    
'''   
