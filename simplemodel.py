#####################################################################################
# Module for model SU(3) x SU(2) x U^3_Y(1) x U^12_B-L(1) x U^2_I3R(1) x U^1_I3R(1) #
#####################################################################################
import flavorstuff as fs
import numpy as np
import GPy
import sympy as sp
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.optimize import minimize, fsolve, least_squares, curve_fit
from scipy.stats import linregress
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

def get_lambda_limits():
    real_Lambda_effs = np.array([980, 18000, 1200, 6200, 510, 1900, 110, 370])
    return real_Lambda_effs

def build_M_b(g,gs,vs):
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*vs[0]/(2), gs[0]*vs[0]/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3,              0,                  0],
                        [0, gs[0]*vs[1]/2, -gs[1]*vs[1],                0,                  0],
                        [0, gs[0]*vs[2]/2,              0,-gs[2]*vs[2]/2,                   0],
                        [0, 0,                          0, gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])

    return M_b 

# Function for minimize() that finds the gs that give the Z-boson mass
# There should be an exact analytical solution to this.
def closest_Z_mass(gs, mzs,g, vs):
    M_b = build_M_b(g,gs,vs)
    Delta2, V = fs.gauge_boson_basis(M_b) 
    return np.sqrt(Delta2[1:]) - mzs
    #return (np.sqrt(Delta2[1]) - mzs[0])**2 + np.sum(np.sqrt(Delta2[2:]) - mzs[1:])**2 + np.sqrt(np.abs(Delta2[0])) #np.abs(np.sqrt(Delta2[1]) - conts.m_Z)

    
# Builds the yukawa matrix as specified in the article
def build_yukawa(ys, ms, vs):
    # Normalization
    vs = vs/np.sqrt(2)
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
    base_charge = [fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Q")]
    charges = basis*base_charge
    Q = np.diag([1,1,1])*np.sum(charges)
    return Q
## OBSOLETE ##
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
    sm_Q_d_L = np.diag(sm_Q(f"{field}_L", base_SM))
    sm_Q_d_R = np.diag(sm_Q(f"{field}_R", base_SM))

    Q_d_L = build_Q(f"{field}_L", base)
    Q_d_R = build_Q(f"{field}_R", base)

    mass_Q_d_L = np.abs(fs.mass_Q(U_L, Q_d_L))
    mass_Q_d_R = np.abs(fs.mass_Q(np.transpose(Uh_R), Q_d_R))
    compare_Q = np.append(np.diag(mass_Q_d_L)[:3]-np.abs(sm_Q_d_L),np.diag(mass_Q_d_R)[:3]-np.abs(sm_Q_d_R))
    compare_Q = np.append(compare_Q, mass_Q_d_L[0,1])
    compare_Q = np.append(compare_Q, mass_Q_d_L[0,2])
    compare_Q = np.append(compare_Q, mass_Q_d_L[1,2])
    return compare_Q


def solve_for_ys(ys,y3_u, y3_d,v_us,v_ds, g, g_prim, V, M_Us, m_us, M_Ds, m_ds, mzs):
    
    # if any(np.abs(ys) < 0.01):
    #     return 1000
        #return [1000]*39
    
    ys_u = ys[:12]
    ys_d = ys[12:]
    

    yukawas_u = build_yukawa(np.insert(ys_u, 0, y3_u), M_Us, v_us)
    yukawas_d = build_yukawa(np.insert(ys_d, 0, y3_d), M_Ds, v_ds)

    U_u_L, diag_yukawa_u, Uh_u_R = fs.diag_yukawa(yukawas_u)
    U_d_L, diag_yukawa_d, Uh_d_R = fs.diag_yukawa(yukawas_d)

    #m_u_compare = np.concatenate((diag_yukawa_u[:3] - m_us,diag_yukawa_u[3:] - M_Us))
    #m_d_compare = np.concatenate((diag_yukawa_d[:3] - m_ds,diag_yukawa_d[3:] - M_Ds))
    m_u_compare = diag_yukawa_u[:3] - m_us
    m_d_compare = diag_yukawa_d[:3] - m_ds
    
    base_SM_Z = np.array([1, -conts.sw2])
    base_SM_gamma = np.array([0, conts.e_em])
    #base_SM_Z = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    #base_SM_gamma = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])

    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R), Q_compare("u", V[:,0], base_SM_gamma, U_u_L, Uh_u_R),
                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R), Q_compare("d", V[:,0], base_SM_gamma, U_d_L, Uh_d_R)))
 #   print(compare_Qs.shape)
    
    V_CKM_calc = np.dot(np.transpose(U_u_L), U_d_L) 
    CKM_compare = (np.abs(V_CKM_calc[:3,:3])-np.abs(conts.V_CKM)).flatten()
    #print(V_CKM_calc[:3,:3])
    #print(conts.V_CKM)
    #print(CKM_compare)
    #Scale according to order?
   # print(compare_Qs)
    # if any(np.abs(compare_Qs) > 0.1):
    #     print(compare_Qs)
    #     return 1000
    
    #compares = np.concatenate((CKM_compare, compare_Qs, m_u_compare, m_d_compare))
    #compares = np.concatenate((CKM_compare, compare_Qs))
    compares = np.concatenate((CKM_compare, m_u_compare, m_d_compare))
    return compares

def wrapper_solve_for_ys(ys,y3_u, y3_d,v_us,v_ds, g, g_prim, V, M_Us, m_us, M_Ds, m_ds, mzs):
    compares = solve_for_ys(ys,y3_u, y3_d,v_us,v_ds, g, g_prim, V, M_Us, m_us, M_Ds, m_ds, mzs)
    return np.sum(compares**2)

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

def get_g_models(filename, g, search_for_gs = False):
    if search_for_gs:
        v_chis = np.arange(1000,10000,10)
        m_Z3s = np.arange(1000,10000,10)
        model_list = []
        tmp_list = []
        for v_chi in tqdm(v_chis):
            #print(f"On v_chi {v_chi}")
            # Conditions:
            for m_Z3 in m_Z3s: 
                #print(f"On m_z3 {m_Z3}")
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
            np.savez(filename, *model_list)
    
    g_models = np.load(filename)
    model_list = [g_models[k] for k in g_models]
    print(f"The number of g_models is: {len(model_list)}")

    return model_list

def calc_vs(tan_beta, vs):
    v_us = vs
    v_ds = vs

    v_ds[0]= np.sqrt(vs[0]**2/(1+tan_beta**2))
    v_us[0] = tan_beta*v_ds[0]
    return v_us, v_ds

def get_y_models(filename, search_for_ys = False, g_model_list = None, cost_tol = 0.5, m_repeats = 20, max_iters = 100):
    if search_for_ys:
        model_list = []
        tmp_list = []
        m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
        m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
        for g_idx, g_model in enumerate(g_model_list):
            print(f"On g model: {g_idx}")
            gs = g_model[2,:]
            mzs = g_model[0,:]
            vs = g_model[1,:]
            #print(f"On g_model {g_idx}")
            M_b = build_M_b(g,gs,vs)

            Delta2, V = fs.gauge_boson_basis(M_b)    
            # Cuts off non-significant contributions, for a cleaner model
            V[np.abs(V) < 0.01] = 0 
            for m in range(m_repeats):

                tan_beta = np.random.randint(10,100)
                M_23 = np.random.randint(1000,10000)
                M_12 = 10*M_23 + np.random.randint(1,100)
                
                M_Ds = np.array([M_23, M_23, M_12])
                M_Us = np.array([M_23, M_23, M_12])

                v_us, v_ds = calc_vs(tan_beta, vs)
                y3_u = np.sqrt(2)*conts.m_t/v_us[0]
                y3_d = np.sqrt(2)*conts.m_b/v_ds[0]

                ys = np.random.uniform(0.01,2,24)
                neg_indices = np.random.random(len(ys)) > 0.8
                np.negative(ys, where=neg_indices, out=ys)
                #print(ys)
                #ys = np.ones(24)*0.1

#                res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,g,g_prim,V,M_Us,m_us,M_Ds,m_ds,mzs), 
#                                    loss = "soft_l1", method= "dogbox", bounds = (-2,2), max_nfev=max_iters)
                res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,g,g_prim,V,M_Us,m_us,M_Ds,m_ds,mzs), 
                            loss = "linear", method= "trf", jac = "2-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2,2)) #
                ys = res.x
                # print(res.cost)
                if res.cost < cost_tol and all(np.abs(ys) > 0.01):
                    print("Successful model")
                    y_ds = np.insert(ys[12:],0,y3_d) 
                    y_us = np.insert(ys[:12],0,y3_u)
                    yukawas_u = build_yukawa(y_us, M_Us, v_us)
                    yukawas_d = build_yukawa(y_ds, M_Ds, v_ds)

                    U_u_L, diag_yukawa_u, Uh_u_R = fs.diag_yukawa(yukawas_u)
                    U_d_L, diag_yukawa_d, Uh_d_R = fs.diag_yukawa(yukawas_d)

                    real_M_Us = diag_yukawa_u[3:]
                    real_M_Ds = diag_yukawa_d[3:]
                    base = V[:,1]
                    base_SM = np.array([1, -conts.sw2])
                    # base_SM_gamma = np.array([0, conts.e_em])

                    
                    #base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
                    #base_SM = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])
                    sm_Q_d_L = np.diag(sm_Q("d_L", base_SM))
                    sm_Q_d_R = np.diag(sm_Q("d_R", base_SM))

                    Q_d_L = build_Q("d_L", base)
                    Q_d_R = build_Q("d_R", base)

                    mass_Q_d_L = np.diag(fs.mass_Q(U_d_L, Q_d_L))
                    mass_Q_d_R = np.diag(fs.mass_Q(np.transpose(Uh_d_R), Q_d_R))
                    print(f"Charge difference L: {np.abs(mass_Q_d_L[:3]) - np.abs(sm_Q_d_L)}")
                    print(f"Charge difference R: {np.abs(mass_Q_d_R[:3]) - np.abs(sm_Q_d_R)}")
                    print(fs.mass_Q(U_d_L, Q_d_L))
                    tmp_list = [y_us, y_ds, M_Us, tan_beta, g_idx]
                    print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {g_idx}")
                    model_list.extend(tmp_list)
        if model_list:
            np.savez(filename, *model_list)
    
    y_models = np.load(filename)
    ex_model_list = [y_models[k] for k in y_models]
    model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(model_list)}")
    return model_list

def refine_y_models(filename, y_model_list, g_model_list, cost_tol = 0.5, max_iters = 100):
    model_list = []
    tmp_list = []
    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])

    for y_idx, y_model in enumerate(y_model_list):
        print(f"On y model {y_idx}")
        [y_us, y_ds, M_Us, tan_beta, g_idx] = y_model
        g_model = g_model_list[g_idx]
        gs = g_model[2,:]
        mzs = g_model[0,:]
        vs = g_model[1,:]
        #print(f"On g_model {g_idx}")
        M_b = build_M_b(g,gs,vs)
        
        Delta2, V = fs.gauge_boson_basis(M_b)    
        # Cuts off non-significant contributions, for a cleaner model
        V[np.abs(V) < 0.01] = 0 

        M_Ds = M_Us

        v_us, v_ds = calc_vs(tan_beta, vs)

        ys = np.concatenate((y_us[1:], y_ds[1:]))

        y3_u = y_us[0]
        y3_d = y_ds[0]
        #bounds = ((-2,2) for i in range(len(ys)))
        res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,g,g_prim,V,M_Us,m_us,M_Ds,m_ds,mzs), 
                            loss = "soft_l1", method= "trf", jac = "3-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2,2)) #
        #res = minimize(wrapper_solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,g,g_prim,V,M_Us,m_us,M_Ds,m_ds,mzs), 
        #                     method= "L-BFGS-B", bounds = bounds, options = {"maxiter" : max_iters})
        ys = res.x
        #print(ys)
        print(res.cost)
        print(np.min(np.abs(ys)))
        if res.cost < cost_tol and all(np.abs(ys) > 0.01):
            print("Successful model")
            y_ds = np.insert(ys[12:],0,y3_d) 
            y_us = np.insert(ys[:12],0,y3_u)
            yukawas_u = build_yukawa(y_us, M_Us, v_us)
            yukawas_d = build_yukawa(y_ds, M_Ds, v_ds)

            U_u_L, diag_yukawa_u, Uh_u_R = fs.diag_yukawa(yukawas_u)
            U_d_L, diag_yukawa_d, Uh_d_R = fs.diag_yukawa(yukawas_d)
            print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
            print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
        
            real_M_Us = diag_yukawa_u[3:]
            real_M_Ds = diag_yukawa_d[3:]
            base = V[:,1]
            base_SM = np.array([1, -conts.sw2])
            # base_SM_gamma = np.array([0, conts.e_em])
            # base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
            #base_SM = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])
            sm_Q_d_L = np.diag(sm_Q("d_L", base_SM))
            sm_Q_d_R = np.diag(sm_Q("d_R", base_SM))

            Q_d_L = build_Q("d_L", base)
            Q_d_R = build_Q("d_R", base)

            #TODO: Compare diagonal charges with SM
            mass_Q_d_L = np.diag(fs.mass_Q(U_d_L, Q_d_L))
            mass_Q_d_R = np.diag(fs.mass_Q(np.transpose(Uh_d_R), Q_d_R))
            print(fs.mass_Q(U_d_L, Q_d_L))
            print(f"Charge difference L: {np.abs(mass_Q_d_L[:3]) - np.abs(sm_Q_d_L)}")
            print(f"Charge difference R: {np.abs(mass_Q_d_R[:3]) - np.abs(sm_Q_d_R)}")

            #model_list["0"]["ys"]

            tmp_list = [y_us, y_ds, M_Us, tan_beta, g_idx]
            print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {g_idx}")
            model_list.extend(tmp_list)
    if model_list:
        np.savez(filename, *model_list)
    
    y_models = np.load(filename)
    ex_model_list = [y_models[k] for k in y_models]
    model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(model_list)}")
    return model_list

if __name__ == "__main__":

    m_U, m_D, m_E, v_chi, v_phi, v_sigma = init_constants()
    
    # According to definition
    g = conts.e_em/np.sqrt(conts.sw2)
    g_prim = conts.e_em/np.sqrt(1-conts.sw2)
    #g = 0.652 
    #g_prim = 0.357
    
    # Get models for gs
    search_for_gs = False
    search_for_ys = False
    g_plotting = False
    y_plotting = True
    g_model_list = get_g_models("correct_g_models.npz", g, search_for_gs)

    #g_model_list = g_model_list[1000:]

    y_model_list = get_y_models("correct_y_models_again.npz", search_for_ys, g_model_list, cost_tol=0.5, max_iters=10, m_repeats=30)

    #y_model_list = refine_y_models("correct_refined_y_models.npz", y_model_list, g_model_list, cost_tol=1, max_iters=100)

    #y_model = y_model_list[0]

    # g_model: [mzs, vs, gs]
    # y_model: [y_us, y_ds, M_Us, tan_beta, g_idx]

    # FINALLY: Find the FCNC Wilson Coeffs

    # g_model = g_model_list[2]
    # M_b = build_M_b(g, g_model[2], g_model[1])
    # (U, S, Vd) = np.linalg.svd(-1j*M_b)
    # print(np.real(np.dot(U, Vd)))
    # print(np.real(np.linalg.eigvals(-1j*M_b)))
    # print(np.linalg.svd(M_b)[1])
    # print(np.sort(np.real(np.linalg.eigvals(-1j*M_b)))-np.sort(np.linalg.svd(M_b)[1]))
    # print(np.array(g_model[2])*np.array(g_model[1])/np.sqrt(2))
    # print([(g_model[2][0]+g_model[2][1])*g_model[1][1]/np.sqrt(2), g_model[2][2]*g_model[1][2]/np.sqrt(2), g_model[2][3]*g_model[1][3]/np.sqrt(2)])

    if g_plotting:
        f  = open("saved_g_regrs.txt", "w")
        g1_list = [model[2,0] for model in g_model_list]
        g2_list = [model[2,1] for model in g_model_list]
        #print(np.array(g1_list) - np.array(g2_list))
        #print(g_list)
        #print(np.sqrt(2)*conts.m_Z/conts.v_H)
        v_strings = ["\chi", "\phi", "\sigma"]
        m_strings = ["Z_3", "Z_3'", "Z_{12}"]
        #f.write(f"g2 : {res.slope}\n")
        for i in range(1,4):
            v_list = [model[1,i] for model in g_model_list]
            m_list = [model[0,i] for model in g_model_list]
            #res = linregress(v_list, m_list)
            
            # I guess we'll continue the optimization

            v_s = np.array(v_list)
            m_zs = np.array(m_list)

            res = curve_fit(lambda x, m: m*x, v_s, m_zs)
            v_chi_lin = np.linspace(np.min(v_s), np.max(v_s), 10000)
            stderr = np.sqrt(np.diag(res[1]))
            slope = res[0][0]

            fig = plt.figure(figsize=(3.5,3))
            plt.scatter(v_s/1000, m_zs/1000, s = 2, label = "Data pts", zorder = 1)
            #plt.errorbar(v_chi_uniq, m_z3_means, yerr=m_z3_vars, fmt = ".")
            plt.plot(v_chi_lin/1000, v_chi_lin*slope/1000, "r", label = f"Regr, g = {slope:.3f}", zorder = 2)
            plt.fill_between(v_chi_lin/1000, v_chi_lin*(slope - stderr)/1000, v_chi_lin*(slope + stderr)/1000, alpha = 0.5, color = "r",zorder=3)
            #plt.plot(v_chi_lin, v_chi_lin*(np.mean(g2_list)), "g")
            plt.title(r"Correlation between $v_" + v_strings[i-1] + r"$ and $m_{" + m_strings[i-1] + r"}$")
            plt.xlabel(r"$v_" + v_strings[i-1] + r"$ [TeV]")
            plt.ylabel(r"$m_{" + m_strings[i-1] + r"}$ [TeV]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"figs/v_m_{i}.png")
            f.write(f"g{i+1} : {slope}\n")
        
            #m_123_list = [model[2] for model in y_model_list] 
        f.close()
        
        # Save every g regression
        #g1_list = [model[2,2] for model in g_model_list]        
        
    if y_plotting:
        m_u_list = []
        tan_beta_list = []
        charge_diff_list = []
        Lambda_effs_list = [[],[],[],[]]

        # y_model_list = [y_model_list[5]]
        real_Lambda_effs = get_lambda_limits()
        for y_model in y_model_list:
            tan_beta= y_model[3]
            g_idx = y_model[4]
            vs = g_model_list[g_idx][1]
            gs = g_model_list[g_idx][2]
            M_b = build_M_b(g,gs,vs)        
            Delta2, V = fs.gauge_boson_basis(M_b)    
            # Cuts off non-significant contributions, for a cleaner model
            V[np.abs(V) < 0.01] = 0 

            v_us, v_ds = calc_vs(tan_beta, vs)
            y_ds = y_model[1]
            y_us = y_model[0]
            yukawas_u = build_yukawa(y_us, y_model[2], v_us)
            yukawas_d = build_yukawa(y_ds, y_model[2], v_ds)

            U_u_L, diag_yukawa_u, Uh_u_R = fs.diag_yukawa(yukawas_u)
            U_d_L, diag_yukawa_d, Uh_d_R = fs.diag_yukawa(yukawas_d)

            real_M_Us = diag_yukawa_u[3:]
            real_M_Ds = diag_yukawa_d[3:]
            m_u_list.append(real_M_Us[0])
            tan_beta_list.append(int(tan_beta))
            m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
            m_us = np.array([conts.m_u, conts.m_c, conts.m_t])        
            base_SM = np.array([1, -conts.sw2])
            sm_Q_d_L = np.diag(sm_Q("d_L", base_SM))
            sm_Q_d_R = np.diag(sm_Q("d_R", base_SM))

            for k in range(1,5):
                base = V[:,k]
                Q_d_L = build_Q("d_L", base)
                Q_d_R = build_Q("d_R", base)

                Q_u_L = build_Q("u_L", base)
                Q_u_R = build_Q("u_R", base)

                mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
                mass_Q_d_R = fs.mass_Q(np.transpose(Uh_d_R), Q_d_R)

                mass_Q_u_L = fs.mass_Q(U_u_L, Q_u_L)
                mass_Q_u_R = fs.mass_Q(np.transpose(Uh_u_R), Q_u_R)
                #print(mass_Q_d_L)
                # Same order as in table
                cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_d_L[0,1]*mass_Q_d_R[0,1], mass_Q_u_L[0,1]**2, 
                               mass_Q_u_L[0,1]*mass_Q_u_R[0,1], mass_Q_d_L[0,2]**2, mass_Q_d_L[0,2]*mass_Q_d_R[0,2],
                               mass_Q_d_L[1,2]**2, mass_Q_d_L[1,2]*mass_Q_d_R[1,2]])
                #Absolute value ok?
                cs = np.abs(cs)/(Delta2[k]/(1000**2))
                
                Lambda_effs_list[k-1].append(np.sqrt(1/cs))
            g_idx_list = [int(model[4]) for model in y_model_list]
        Z_strings = ["Z", "Z_3", "Z_3'", "Z_{12}"]
        c_strings = ["\Lambda^{sd}_{LL}", "\Lambda^{sd}_{LR}", "\Lambda^{uc}_{LL}", "\Lambda^{uc}_{LR}", "\Lambda^{bd}_{LL}",
                     "\Lambda^{bd}_{LR}", "\Lambda^{bs}_{LL}","\Lambda^{bs}_{LR}"]
        simp_c_strings = ["sd_LL", "sd_LR", "uc_LL", "uc_LR", "bd_LL", "bd_LR", "bs_LL", "bs_LR"]
        fig, axs = plt.subplots(2,4,figsize=(7,5))
        lines = [None]*5
        for n in range(8):
            if n > 3:
               r = 1
               xn = n-4
            else:
                r = 0
                xn = n 
            #fig = plt.figure(figsize=(3.5,3))
            for k in range(1,5):
                Lambda_effs = np.array(Lambda_effs_list[k-1])
                line = axs[r,xn].scatter(np.arange(0,len(y_model_list)),Lambda_effs[:,n], s = 2, label = f"${Z_strings[k-1]}$")
                lines[k-1] = line
            line = axs[r,xn].hlines(real_Lambda_effs[n], 0, len(y_model_list), color = "black", label = "Limit")
            lines[4] = line
            #plt.scatter(tan_beta_list, m_u_list)
            #plt.scatter(np.array(v_chi_list)[g_idx_list], m_u_list, s = 2)
            #axs[r,xn].set_title(f"${c_strings[n]}$" )
            axs[r,xn].set_xlabel("y-model index")
            axs[r,xn].set_ylabel(f"${c_strings[n]}$ [log(TeV)]")
            axs[r,xn].set_yscale("log")

            #plt.ylim(0,10000)
            # if n == 6:
            #     axs[r,xn].legend(loc='lower center', bbox_to_anchor = (0,-1), ncol=5)
            # #axs[n].tight_layout()
        plt.suptitle("Each $\Lambda_{eff}$ for each y-model and neutral masssive boson")
        #plt.legend(loc='upper center', bbox_to_anchor=(0.1, -0.8), ncol=5)
        fig.legend(handles = lines, loc = "lower center", ncols = 5, bbox_to_anchor = (0.5,0))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f"figs/Lambda_effs_all.png")
        #plt.savefig(f"figs/Lambda_effs_{simp_c_strings[n]}.png")

            #TODO: Calculate further constants, use in minimization?
            #c_sd = (mass_Q_d_L[0,1]**2)/Delta2[1]# + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]
            #print(f"The constant for a kaon-antikaon with neutral current is: {c_sd}")


    '''
                yukawas = build_yukawa(np.insert(y_ds, 0, y3_d), M_Ds, vs)

            M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                                [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3, 0,0],
                                [0,gs[0]*vs[1]/2, -gs[1]*vs[1], 0,0],
                                [0, gs[0]*vs[2]/2,0,-gs[2]*vs[2]/2,0],
                                [0,0,0,gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])
            
            Delta2, V = fs.gauge_boson_basis(M_b)

            # Cuts off non-significant contributions, for a cleaner model
            V[np.abs(V) < 0.01] = 0 

    U_d_L, diag_yukawa, Uh_d_R = fs.diag_yukawa(yukawas)
    print(f"mds diffs: {diag_yukawa[:3]-m_ds}, MDs diffs: {diag_yukawa[3:]-M_Ds}")
 #   print([diag_yukawa[0]-conts.m_d, diag_yukawa[1]-conts.m_s, diag_yukawa[2]-conts.m_b])
    yukawas = build_yukawa(np.insert(y_us, 0, y3_u), M_Us, vs)
    U_u_L, diag_yukawa, Uh_u_R = fs.diag_yukawa(yukawas)

    V_CKM_calc = np.dot(np.transpose(U_u_L), U_d_L) 
    print(V_CKM_calc[:3,:3])
    print(conts.V_CKM)
    CKM_compare = (np.abs(V_CKM_calc[:3,:3])-np.abs(conts.V_CKM)).flatten()
    print(f"CKM compare: {CKM_compare}")
    base = V[:,1]
    base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    
    sm_Q_d_L = np.diag(sm_Q("d_L", base_SM))

    Q_d_L = build_Q("d_L", base)
    Q_d_R = build_Q("d_R", base)

    #TODO: Compare diagonal charges with SM
    mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
    mass_Q_d_R = fs.mass_Q(np.transpose(Uh_d_R), Q_d_R)

    #TODO: CKM matrix!

    #print(mass_Q_d_L)
    #print(sm_Q_d_L)
    print(sm_Q_d_L)
    print(mass_Q_d_L)
    print(f"Charge difference: {np.abs(np.diag(mass_Q_d_L)[:3]) - np.abs(sm_Q_d_L)}")
    #print(mass_Q_d_R)

    #TODO: Calculate further constants, use in minimization?
    c_sd = (mass_Q_d_L[0,1]**2)/Delta2[1]# + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]
    print(f"The constant for a kaon-antikaon with neutral current is: {c_sd}")

    # Get models for ys (depends on gs)
#        print(gs)
#        print(vs)


    for i in range(5):
        print(f"m = {np.sqrt(np.abs(Delta2[i]))}")
        print(f"V_{i} = {V[:,i]}\n")

    print(f"real m_Z = {conts.m_Z}. Found m_Z = {np.sqrt(Delta2[1])}")
    print(f"Diff m_Zs = {mzs-np.sqrt(Delta2[1:])}")
    '''

   
