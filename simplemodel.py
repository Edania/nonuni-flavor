#####################################################################################
# Module for model SU(3) x SU(2) x U^3_Y(1) x U^12_B-L(1) x U^2_I3R(1) x U^1_I3R(1) #
#####################################################################################
import flavorstuff as fs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from scipy.optimize import minimize, fsolve, least_squares, curve_fit
from scipy.stats import linregress
conts = fs.constants()

def get_lambda_limits():
    #real_Lambda_effs = np.array([980, 18000, 1200, 6200, 510, 1900, 110, 370]) # TeV
    real_Lambda_effs = np.array([980, 1200, 510, 110]) # TeV
    return real_Lambda_effs

def build_M_b(g,gs,vs):
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*vs[0]/(2), gs[0]*vs[0]/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3,              0,                  0],
                        [0, gs[0]*vs[1]/2, -gs[1]*vs[1],                0,                  0],
                        [0, gs[0]*vs[2]/2,              0,-gs[2]*vs[2]/2,                   0],
                        [0, 0,                          0, gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])

    return M_b 

def get_delta2_V(g, gs, vs):
    [g1, g2, g3, g4] = gs.flatten()
    [vH, vchi, vphi, vsigma] = vs.flatten()
    M2 = ([[g**2*vH**2/8, -g*g1*vH**2/8, 0, 0, 0], 
            [-g*g1*vH**2/8, g1**2*vH**2/8 + 5*g1**2*vchi**2/36 + g1**2*vphi**2/8, -5*g1*g2*vchi**2/18, -g1*g3*vphi**2/8, 0], 
            [0, -5*g1*g2*vchi**2/18, 5*g2**2*vchi**2/9, 0, 0], 
            [0, -g1*g3*vphi**2/8, 0, g3**2*vphi**2/8 + g3**2*vsigma**2/8, -g3*g4*vsigma**2/8], 
            [0, 0, 0, -g3*g4*vsigma**2/8, g4**2*vsigma**2/8]])
    (Delta2, V) = np.linalg.eigh(M2)
    return Delta2, V

# Function for minimize() that finds the gs that give the Z-boson mass
# There should be an exact analytical solution to this.
def closest_Z_mass(gs, mzs,g, vs):
    #M_b = build_M_b(g,gs,vs)
    #Delta2, V = fs.gauge_boson_basis(M_b) 
    Delta2, V = get_delta2_V(g,gs,vs)
    return np.sqrt(2*Delta2[1:]) - mzs


# Builds the yukawa matrix as specified in the article. 
# VEVs are treated unnormalized prior to this
def build_yukawa(ys, ms, vs, alt = False):
    vs = vs/np.sqrt(2)
    [v, v_chi, v_phi, v_sigma] = vs.flatten()
    m = np.zeros((3,3))
    m[2,2] = v*ys[0]
    Delta_L = v*np.array([[ys[1], ys[2], 0],[ys[3], ys[4],0], [0,0,0]])
    Delta_R = np.array([[0, ys[5]*v_phi, 0], [0, ys[6]*v_phi, ys[7]*v_chi], [ys[8]*v_sigma,0,0]])
    if alt:
        Delta_R[0,2] = ys[13]

    M = np.array([[ms[0], 0, ys[9]*v_phi], [0, ms[1],ys[10]*v_phi], [ys[11]*v_phi, ys[12]*v_phi, ms[2]]])

    yukawas = np.vstack((np.hstack((m, Delta_L)), np.hstack((Delta_R, M))))
    return yukawas

def build_base_charge(field_type):
    base_charge = np.array([fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Y"), 
                   (fs.find_charge("fermions", field_type, "B") - fs.find_charge("fermions", field_type, "L"))/2,
                   fs.find_charge("fermions", field_type, "I3_R"), fs.find_charge("fermions", field_type, "I3_R")])
    return base_charge


# function for finding yukawa mass basis via minimize()
# builds the Q for this model, given a basis, eg Z^3, Z^3'...
def build_Q(field, basis):
    
    
    if field[0] == "u":
        heavy = "U"
    else:
        heavy = "D"
    
    base_charge_light = build_base_charge(field)
    charges = basis*base_charge_light
    Q_arr_light = charges[0] + np.array([charges[2] + charges[3], charges[2] + charges[4], charges[1]])

    base_charge_alpha = build_base_charge(heavy+"_alpha")    
    charges_alpha = basis*base_charge_alpha
    base_charge_3 = build_base_charge(heavy+"_3")    
    charges_3 = basis*base_charge_3
    Q_arr_heavy = np.array([charges_alpha[1] + charges_alpha[2], charges_alpha[1] + charges_alpha[2], charges_3[2] + charges_3[3]])
    Q_arr = np.concatenate((Q_arr_light, Q_arr_heavy))
    # TODO: Correct heavy particles 
    Q = np.diag(Q_arr)
    return Q

def sm_Q(field_type, basis):
    base_charge = np.array([fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Q")])
    charges = basis*base_charge
    Q = np.diag([1,1,1])*np.sum(charges)
    return Q

def Q_compare(field, base, base_SM, U_L, Uh_R):
    sm_Q_d_L = np.diag(sm_Q(f"{field}_L", base_SM))
    sm_Q_d_R = np.diag(sm_Q(f"{field}_R", base_SM))

    Q_d_L = build_Q(f"{field}_L", base)
    Q_d_R = build_Q(f"{field}_R", base)

    mass_Q_d_L = np.abs(fs.mass_Q(U_L, Q_d_L))
    mass_Q_d_R = np.abs(fs.mass_Q(np.transpose(Uh_R), Q_d_R))
    compare_Q = np.append(np.diag(mass_Q_d_L)[:3]-np.abs(sm_Q_d_L),np.diag(mass_Q_d_R)[:3]-np.abs(sm_Q_d_R))
    return compare_Q


def solve_for_ys(ys,y3_u, y3_d,v_us,v_ds, V, M_Us, m_us, m_ds):
    
    # if any(np.abs(ys) < 0.01):
    #     return 1000
        #return [1000]*39
    
    ys_u = ys[:12]
    ys_d = ys[12:]


    yukawas_u = build_yukawa(np.insert(ys_u, 0, y3_u), M_Us, v_us, alt = False)
    yukawas_d = build_yukawa(np.insert(ys_d, 0, y3_d), M_Us, v_ds, alt = False)

    Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
    Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)

    #m_u_compare = np.concatenate((diag_yukawa_u[:3] - m_us,diag_yukawa_u[3:] - M_Us))
    #m_d_compare = np.concatenate((diag_yukawa_d[:3] - m_ds,diag_yukawa_d[3:] - M_Ds))
    m_u_compare = diag_yukawa_u[:3] - m_us
    m_d_compare = diag_yukawa_d[:3] - m_ds
    
    base_SM_Z = np.array([1, -conts.sw2])
    base_SM_gamma = np.array([0, conts.e_em])
    #base_SM_Z = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
    #base_SM_gamma = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])

#    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R), Q_compare("u", V[:,0], base_SM_gamma, U_u_L, Uh_u_R),
#                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R), Q_compare("d", V[:,0], base_SM_gamma, U_d_L, Uh_d_R)))
    compare_Qs = np.concatenate(Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R),
                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R))
 
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
    
    compares = np.concatenate((CKM_compare, compare_Qs, m_u_compare, m_d_compare))
    #compares = np.concatenate((CKM_compare, compare_Qs))
    #compares = np.concatenate((CKM_compare, m_u_compare, m_d_compare))
    return compares

def alt_solve_for_ys(ys,g,g_prim,v_us,v_ds, V, M_Us, m_us, m_ds):
    
    ys_u = ys[:14]
    ys_d = ys[14:]

    yukawas_u = build_yukawa(ys_u, M_Us, v_us, alt = True)
    yukawas_d = build_yukawa(ys_d, M_Us, v_ds, alt = True)

    Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
    Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)

    m_u_compare = diag_yukawa_u[:3] - m_us
    m_d_compare = diag_yukawa_d[:3] - m_ds
    
    base_SM_Z = fs.get_base_Z(g,g_prim)
    base_SM_gamma = fs.get_base_gamma()
#    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R), Q_compare("u", V[:,0], base_SM_gamma, U_u_L, Uh_u_R),
#                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R), Q_compare("d", V[:,0], base_SM_gamma, U_d_L, Uh_d_R)))
    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R),
                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R)))
    
    V_CKM_calc = np.dot(np.transpose(U_u_L), U_d_L) 
    CKM_compare = (np.abs(V_CKM_calc[:3,:3])-np.abs(conts.V_CKM)).flatten()
    compares = np.concatenate((CKM_compare, compare_Qs, m_u_compare, m_d_compare))

    return compares

def get_g_models(filename, g, search_for_gs = False, verbose = False):
    if search_for_gs:
        v_chis = np.arange(1000,10000,10)
        #m_Z3s = np.arange(1000,10000,10)
        m_Z3s = np.arange(4000,10000,(10000-4000)/len(v_chis))
        model_list = []
        tmp_list = []
        for v_chi in tqdm(v_chis):
            #print(f"On v_chi {v_chi}")
            # Conditions:
            for m_Z3 in m_Z3s: 
                #print(f"On m_z3 {m_Z3}")
                v_phi = int(v_chi*np.random.uniform(1.5,5))
                #v_sigma = 10*v_chi + np.random.randint(0,100)
                v_sigma = int(v_phi*np.random.uniform(2,8))
                m_Z3prim = int(m_Z3*np.random.uniform(1.5,5))
                m_Z12 = int(m_Z3prim*np.random.uniform(2,8))
                #m_Z12 = m_Z3*10 + np.random.randint(0,100)
                # Let's do this fsolve for a range of ms and vs, with conditions on the grid :)
                gs = np.random.uniform(0.01,5,4)
                vs = np.array([conts.v_H, v_chi, v_phi, v_sigma])
                mzs = np.array([conts.m_Z, m_Z3, m_Z3prim, m_Z12])
                gs, infodict,ier,mesg = fsolve(closest_Z_mass, gs,args=(mzs, g, vs), full_output=True, factor = 0.1)
                diff = infodict["fvec"]
                #print(gs)
                
                if all(gs <= 5)  and all(gs > 0.01) and all(np.abs(diff) < 1e-5):
                    tmp_list = [mzs, vs, gs]
                    model_list.append(tmp_list)
                    if verbose:
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

def calc_vs(tan_beta, v_H):
    v_d= np.sqrt(v_H**2/(1+tan_beta**2))
    v_u = tan_beta*v_d
    return v_u, v_d

def get_y_models(filename, search_for_ys = False, g_model_list = None, cost_tol = 0.5, m_repeats = 20, max_iters = 100, verbose = True, alt = True):
    if search_for_ys:
        model_list = []
        tmp_list = []
        m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
        m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
        successes = 0
        M_23s = np.linspace(1000,20000, m_repeats)
        #M_12s = 10*M_23s + np.random.randint(1,100, len(M_23s))

        for g_idx, g_model in enumerate(tqdm(g_model_list)):
            if g_idx % 100 == 0:
                print(f"\nSuccesses: {successes}")
            vs = g_model[1,:]
            gs = g_model[2,:]
            #M_b = build_M_b(g,gs,vs)

            #Delta2, V = fs.gauge_boson_basis(M_b)    
            Delta2, V = get_delta2_V(g, gs, vs)
            #V[np.abs(V) < 0.01] = 0 
            valid_idxs = np.argwhere(M_23s > vs[1]).flatten()
            for idx in valid_idxs:
                M_23 = M_23s[idx]
                M_12 = 0
                while M_12 < vs[3]:
                    M_12 = np.random.uniform(6,40)*M_23
                #M_12 = 10*M_23 + np.random.randint(1,100)
                tan_beta = np.random.randint(10,100)
                #tan_beta = 10
                #M_23 = np.random.randint(1000,10000)
                
                M_Us = np.array([M_23, M_23, M_12])

                v_u, v_d = calc_vs(tan_beta, conts.v_H)
                v_us = np.array([v_u, vs[1], vs[2], vs[3]])
                v_ds = np.array([v_d, vs[1], vs[2], vs[3]])
                # Is this necessary?
                if alt:
                    y3_u = np.sqrt(2)*conts.m_t/v_us[0]
                    y3_d = np.sqrt(2)*conts.m_b/v_ds[0]
                    ys = np.random.uniform(0.01,2,28)
                else:
                    y3_u = np.sqrt(2)*conts.m_t/v_us[0]
                    y3_d = np.sqrt(2)*conts.m_b/v_ds[0]

                    ys = np.random.uniform(0.01,2,24)
                
                neg_indices = np.random.random(len(ys)) > 0.5
                np.negative(ys, where=neg_indices, out=ys)
                ys[0] = y3_u
                ys[14] = y3_d
                if alt:
                    res = least_squares(alt_solve_for_ys, ys, args=(g,g_prim,v_us,v_ds,V,M_Us,m_us,m_ds), 
                                loss = "linear", method= "trf", jac = "2-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2.5,2.5)) #
                    
                else:
                    res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,V,M_Us,m_us,m_ds), 
                                loss = "linear", method= "trf", jac = "2-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2,2)) #
                ys = res.x
                #print(res.cost)
                if res.cost < cost_tol and all(np.abs(ys) > 0.01):
                    successes += 1
                    if alt:
                        y_ds = ys[14:]
                        y_us = ys[:14]
                        
                    else:
                        y_ds = np.insert(ys[12:],0,y3_d) 
                        y_us = np.insert(ys[:12],0,y3_u)
       
                    tmp_list = [y_us, y_ds, M_Us, tan_beta, g_idx]
                    if verbose:
                        yukawas_u = build_yukawa(y_us, M_Us, v_us)
                        yukawas_d = build_yukawa(y_ds, M_Us, v_ds)

                        Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                        Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)
                        real_M_Us = diag_yukawa_u[3:]
                        real_M_Ds = diag_yukawa_d[3:]
                        base = V[:,1]
                        base_SM = fs.get_base_Z(g,g_prim)
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
                        print(f"Charge difference L: {np.abs(mass_Q_d_L[:3]) - np.abs(sm_Q_d_L)}")
                        print(f"Charge difference R: {np.abs(mass_Q_d_R[:3]) - np.abs(sm_Q_d_R)}")
                        print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
                        print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
                 
                        print("Successful model")
                        print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {g_idx}")
                    model_list.extend(tmp_list)
        if model_list:
            np.savez(filename, *model_list)
    
    y_models = np.load(filename)
    ex_model_list = [y_models[k] for k in y_models]
    model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(model_list)}")
    return model_list

def refine_y_models(filename, y_model_list, g_model_list, cost_tol = 0.5, max_iters = 100, verbose = True, alt = True):
    model_list = []
    tmp_list = []
    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
    successes = 0
    #M_12s = 10*M_23s + np.random.randint(1,100, len(M_23s))
    for y_idx, y_model in enumerate(tqdm(y_model_list)):
        if y_idx % 100 == 0:
            print(f"\nSuccesses: {successes}")
        [y_us, y_ds, M_Us, tan_beta, g_idx] = y_model
        g_model = g_model_list[g_idx]
        gs = g_model[2,:]
        vs = g_model[1,:]
        v_u, v_d = calc_vs(tan_beta, conts.v_H)
        v_us = np.array([v_u, vs[1], vs[2], vs[3]])
        v_ds = np.array([v_d, vs[1], vs[2], vs[3]])
        ys = np.concatenate((y_us, y_ds))

        #M_b = build_M_b(g,gs,vs)

        #Delta2, V = fs.gauge_boson_basis(M_b)    
        Delta2, V = get_delta2_V(g, gs, vs)
        #V[np.abs(V) < 0.01] = 0 
        if alt:
            res = least_squares(alt_solve_for_ys, ys, args=(g,g_prim,v_us,v_ds,V,M_Us,m_us,m_ds), 
                        loss = "linear", method= "trf", jac = "3-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2.5,2.5)) #
            
        else:
            res = least_squares(solve_for_ys, ys, args=(v_us,v_ds,V,M_Us,m_us,m_ds), 
                        loss = "linear", method= "trf", jac = "3-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2,2)) #
        ys = res.x
        if verbose:
            print(res.cost)
        if res.cost < cost_tol and all(np.abs(ys) > 0.01):
            successes += 1
            if alt:
                y_ds = ys[14:]
                y_us = ys[:14]
                
            else:
                pass
                #y_ds = np.insert(ys[12:],0,y3_d) 
                #y_us = np.insert(ys[:12],0,y3_u)

            tmp_list = [y_us, y_ds, M_Us, tan_beta, g_idx]
            if verbose:
                yukawas_u = build_yukawa(y_us, M_Us, v_us)
                yukawas_d = build_yukawa(y_ds, M_Us, v_ds)

                Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)
                real_M_Us = diag_yukawa_u[3:]
                real_M_Ds = diag_yukawa_d[3:]
                base = V[:,1]
                base_SM = fs.get_base_Z(g,g_prim)
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
                print(f"Charge difference L: {np.abs(mass_Q_d_L[:3]) - np.abs(sm_Q_d_L)}")
                print(f"Charge difference R: {np.abs(mass_Q_d_R[:3]) - np.abs(sm_Q_d_R)}")
                print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
                print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
            
                print("Successful model")
                print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {g_idx}")
            model_list.extend(tmp_list)
    if model_list:
        np.savez(filename, *model_list)
    
    y_models = np.load(filename)
    ex_model_list = [y_models[k] for k in y_models]
    model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(model_list)}")
    return model_list

def refine_y_models_old(filename, y_model_list, g_model_list, cost_tol = 0.5, max_iters = 100, verbose = True):
    model_list = []
    tmp_list = []
    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
    successes = 0
    for y_idx, y_model in enumerate(tqdm(y_model_list)):
        if y_idx % 100 == 0:
             print(f"\nSuccesses: {successes}")
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
        res = least_squares(solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,V,M_Us,m_us,m_ds), 
                            loss = "linear", method= "trf", jac = "3-point", tr_solver="exact", max_nfev=max_iters, bounds = (-2,2)) #
        #res = minimize(wrapper_solve_for_ys, ys, args=(y3_u,y3_d,v_us,v_ds,g,g_prim,V,M_Us,m_us,M_Ds,m_ds,mzs), 
        #                     method= "L-BFGS-B", bounds = bounds, options = {"maxiter" : max_iters})
        ys = res.x
        # print(res.cost)
        # print(np.min(np.abs(ys)) )
        if res.cost < cost_tol and all(np.abs(ys) > 0.01):
            successes += 1
            y_ds = np.insert(ys[12:],0,y3_d) 
            y_us = np.insert(ys[:12],0,y3_u)

            tmp_list = [y_us, y_ds, M_Us, tan_beta, g_idx]
            if verbose:
                yukawas_u = build_yukawa(y_us, M_Us, v_us)
                yukawas_d = build_yukawa(y_ds, M_Us, v_ds)

                Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)
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
                print(f"Charge difference L: {np.abs(mass_Q_d_L[:3]) - np.abs(sm_Q_d_L)}")
                print(f"Charge difference R: {np.abs(mass_Q_d_R[:3]) - np.abs(sm_Q_d_R)}")
                print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
                print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
            
                print("Successful model")
                print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {g_idx}")
            model_list.extend(tmp_list)
        if model_list:
            np.savez(filename, *model_list)
    
    y_models = np.load(filename)
    ex_model_list = [y_models[k] for k in y_models]
    model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(model_list)}")
    return model_list

def pick_g_models(in_model_list, n_idxs = 1):
    v_list = np.array([model[1,1] for model in in_model_list])
    uniques, unique_idxs = np.unique(v_list, return_index=True)
    unique_idxs = np.append(unique_idxs, len(v_list))
    random_idxs = [np.random.randint(unique_idxs[i],unique_idxs[i+1], n_idxs) if n_idxs <= unique_idxs[i+1]-unique_idxs[i] 
                   else np.random.randint(unique_idxs[i],unique_idxs[i+1],unique_idxs[i+1]-unique_idxs[i]) 
                   for i in range(len(unique_idxs)-1)
                   ]
    random_idxs = np.concatenate(random_idxs).ravel()
    out_model_list = [in_model_list[idx] for idx in random_idxs]
    print(v_list[random_idxs])
    return out_model_list


def calc_grid(g_model_list, m_repeats):
    M_23s = np.linspace(1000,10000, m_repeats)
    #M_12s = 10*M_23s + np.random.randint(1,100, len(M_23s))
    c_sum = 0
    for g_idx, g_model in enumerate(tqdm(g_model_list)):
        vs = g_model[1,:]
        valid_idxs = np.argwhere(M_23s > vs[1]).flatten()
        c_sum += len(valid_idxs)
    return c_sum

if __name__ == "__main__":
    # According to definition
    g = conts.e_em/np.sqrt(conts.sw2)
    g_prim = conts.e_em/np.sqrt(1-conts.sw2)
    
    # Get models for gs
    search_for_gs = False
    search_for_ys = False
    g_plotting = False
    y_plotting = True
    picking_gs = False
    refining_ys = False
    refine_filename = "refined_valid_wide.npz"
    if picking_gs:
        #g_filename = "correct_g_models_again.npz"
        g_filename = "g_models_wide.npz"
    else:
        #g_filename = "saved_g_models.npz"
        g_filename = "saved_g_models_wide.npz"
    g_model_list = get_g_models(g_filename, g, search_for_gs)
    if picking_gs:
        g_model_list = pick_g_models(g_model_list, n_idxs=5)
        np.savez("saved_g_models_wide.npz", *g_model_list)
    #c_sum = calc_grid(g_model_list, 40)
    #print(c_sum)
    
    looping = False
    
    y_filenames = ["sd_list_wide.npz", "uc_list_wide.npz", "bd_list_wide.npz", "bs_list_wide.npz"]
    #valid_filename = "checked_refined_valid_wide.npz"
    valid_filename = "checked_refined_valid_y_models_wide_oops.npz"
    if looping:
        y_model_lists = []
        for y_filename in y_filenames:
            y_model_lists.append(get_y_models(y_filename, search_for_ys, g_model_list, cost_tol=0.3, max_iters=20, m_repeats=40, verbose=False, alt = True))
    else:
        #y_filename = "y_models_dof_28_3.npz"
        #y_filename = "refined_y_dof_28_2_1.npz"
        #y_filename = "valid_y_models_wide.npz"
        y_filename = "checked_refined_valid_y_models_wide.npz"
        #y_filename = "y_models_wide_again.npz"
        y_model_lists = [get_y_models(y_filename, search_for_ys, g_model_list, cost_tol=0.1, max_iters=20, m_repeats=100, verbose=False, alt = True)]

    #TODO: Check couplings for light families to new Z-bosons to ascertain lower mass limit

    #list_names = ["sd_list.npz", "uc_list.npz", "bd_list.npz", "bs_list.npz"]

    if refining_ys:
        y_model_list = refine_y_models(refine_filename, y_model_lists[0], g_model_list, cost_tol=0.1, max_iters=None, verbose=False)
   
    # g_model: [mzs, vs, gs]
    # y_model: [y_us, y_ds, M_Us, tan_beta, g_idx]

    if g_plotting:
        f  = open("saved_g_regrs.txt", "w")
        g1_list = [model[2,0] for model in g_model_list]
        g2_list = [model[2,1] for model in g_model_list]
        g12_list = [model[2,1]+ model[2,0] for model in g_model_list]
        v_strings = ["\chi", "\phi", "\sigma"]
        m_strings = ["Z_3", "Z_3'", "Z_{12}"]
        for i in range(1,4):
            v_list = [model[1,i]/1000/np.sqrt(2) for model in g_model_list]
            m_list = [model[0,i]/1000 for model in g_model_list]
            res = linregress(v_list, m_list)
            #[slope],stderr,_,_ = np.linalg.lstsq(np.array(v_list).reshape(-1,1), m_list) 
            #print(slope)   
            #stderr=0
            #intercept = 0
            v_s = np.array(v_list)
            m_zs = np.array(m_list)

            v_chi_lin = np.linspace(np.min(v_s), np.max(v_s), 10)
            slope = res.slope
            intercept = res.intercept
            stderr = res.stderr
            fig = plt.figure(figsize=(3.5,4))
            plt.scatter(v_s, m_zs, s = 2, label = "Data pts", zorder = 1)
            plt.plot(v_chi_lin, v_chi_lin*slope + intercept, "r", label = f"gx + m, g = {slope:.3f}, m = {intercept:.3f}", zorder = 2)
            plt.fill_between(v_chi_lin, v_chi_lin*(slope - stderr) + intercept, v_chi_lin*(slope + stderr) + intercept, alpha = 0.5, color = "r",zorder=3)
            plt.title(r"Correlation between $v_" + v_strings[i-1] + r"$ and $m_{" + m_strings[i-1] + r"}$")
            plt.xlabel(r"$v_" + v_strings[i-1] + r"/\sqrt{2}$ [TeV]")
            plt.ylabel(r"$m_{" + m_strings[i-1] + r"}$ [TeV]")
            plt.legend(loc = "upper center", bbox_to_anchor = (0.5,-0.2))
            plt.tight_layout()
            plt.savefig(f"figs/v_m_{i}.png")
            f.write(f"g{i+1} : {slope}\n")
        f.close()
        
        v_chi_list = np.array([model[1,1] for model in g_model_list])/1000/np.sqrt(2)
        v_phi_list = np.array([model[1,2] for model in g_model_list])/1000/np.sqrt(2)

        m_Z3_list = np.array([model[0,1] for model in g_model_list])/1000
        m_Z3prim_list = np.array([model[0,2] for model in g_model_list])/1000

        v_res = linregress(v_chi_list, v_phi_list)
        m_res = linregress(m_Z3_list, m_Z3prim_list)
        fig, axs = plt.subplots(1,2, figsize = (7,4))
        axs[0].scatter(v_chi_list, v_phi_list, s = 2, label = "data")
        axs[0].plot(v_chi_list, v_chi_list*v_res.slope + v_res.intercept, "r", label = f"slope = {v_res.slope:.3f}")
        axs[0].plot(v_chi_list, v_chi_list*1.5, label = r"$\times 1.5$ limit")
        axs[0].plot(v_chi_list, v_chi_list*5, label = r"$\times 5$ limit")
        axs[0].set_title("Correlation between $v_\chi$ and $v_\phi$")
        axs[0].set_xlabel("$v_\chi/\sqrt{2}$ [TeV]")
        axs[0].set_ylabel("$v_\phi/\sqrt{2}$ [TeV]")
        axs[0].legend()
        axs[1].scatter(m_Z3_list, m_Z3prim_list, s = 2, label = "data")
        axs[1].plot(m_Z3_list, m_Z3_list*m_res.slope + m_res.intercept, "r", label = f"slope = {m_res.slope:.3f}")
        axs[1].plot(m_Z3_list, m_Z3_list*1.5, label = r"$\times 1.5$ limit")
        axs[1].plot(m_Z3_list, m_Z3_list*5, label = r"$\times 5$ limit")
        axs[1].set_title("Correlation between $m_{Z_3}$ and $m_{Z'_3}$")
        axs[1].set_xlabel("$m_{Z_3}$ [TeV]")
        axs[1].set_ylabel("$m_{Z'_3}$ [TeV]")
        axs[1].legend()
        plt.tight_layout()
        plt.savefig("figs/v_m_ratios.png")
        # Save every g regression
        #g1_list = [model[2,2] for model in g_model_list]        
        
    if y_plotting:
        for k, y_model_list in enumerate(y_model_lists):
            scatter_index = False
            scatter_tan_beta = False
            scatter_m_v_ratio = True
            valid_model_check =  False
            save_Z_checks = False

            good_index = k
            if looping:
                y_filename = y_filenames[k]

            if scatter_index:
                lambda_filename = "Lambda_effs_all.png"
            elif scatter_tan_beta:
                lambda_filename = "Lambda_effs_tan_beta.png"
            elif scatter_m_v_ratio:
                lambda_filename = y_filename[:-4] + "Lambda_effs_m_v_ratio.png"

            tan_beta_list = []
            charge_diff_list = []
            Lambda_effs_list = [[],[],[],[]]
            valid_model_list = []
            g_idx_list = [int(model[4]) for model in y_model_list]
            cost_list = []
            real_Lambda_effs = get_lambda_limits()
            sd_list = []
            uc_list = []
            bd_list = []
            bs_list = []
            all_diffs = []
            compare_list = []
            v_list = []
            for y_model in y_model_list:
                y_us = np.array(y_model[0])
                y_ds = np.array(y_model[1])
                M_Us = y_model[2]
                tan_beta= y_model[3]
                g_idx = y_model[4]
                vs = g_model_list[g_idx][1,:]
                v_list.append(vs[1])
                gs = g_model_list[g_idx][2,:]
                Delta2, V = get_delta2_V(g, gs, vs)   

                v_u, v_d = calc_vs(tan_beta, conts.v_H)
                v_us = [v_u, vs[1], vs[2], vs[3]]
                v_ds = [v_d, vs[1], vs[2], vs[3]]
                yukawas_u = build_yukawa(y_us, M_Us, v_us, alt = True)
                yukawas_d = build_yukawa(y_ds, M_Us, v_ds, alt = True)
                Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)

                real_M_Us = diag_yukawa_u[3:]
                real_M_Ds = diag_yukawa_d[3:]

                tan_beta_list.append(int(tan_beta))
                m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
                m_us = np.array([conts.m_u, conts.m_c, conts.m_t])        
                
                tot_L_diffs = []
                in_ys = np.concatenate((y_us, y_ds))             
                compares = alt_solve_for_ys(in_ys, g, g_prim, v_us, v_ds, V, y_model[2], m_us, m_ds)
                compare_list.append(compares)
                cost = np.sum(compares**2)
                cost_list.append(cost)

                for k in range(1,5):
                    base = V[:,k]
                    Q_d_L = build_Q("d_L", base)
                    Q_d_R = build_Q("d_R", base)

                    Q_u_L = build_Q("u_L", base)
                    Q_u_R = build_Q("u_R", base)

                    mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
                    #mass_Q_d_RL = fs.mass_Q(U_d_L, Q_d_R, np.transpose(Uh_d_R))
                    #mass_Q_d_LR = fs.mass_Q(np.transpose(Uh_d_R), Q_d_L,U_d_L)

                    mass_Q_u_L = fs.mass_Q(U_u_L, Q_u_L)
                    #mass_Q_u_LR = fs.mass_Q(U_u_L, Q_u_R, np.transpose(Uh_u_R))
                    #mass_Q_u_RL = fs.mass_Q(np.transpose(Uh_u_R), Q_u_L,U_u_L)
                    # Same order as in table
                    # cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_d_LR[0,1]*mass_Q_d_RL[0,1], mass_Q_u_L[0,1]**2, 
                    #                mass_Q_u_LR[0,1]*mass_Q_u_RL[0,1], mass_Q_d_L[0,2]**2, mass_Q_d_LR[0,2]*mass_Q_d_RL[0,2],
                    #                mass_Q_d_L[1,2]**2, mass_Q_d_LR[1,2]*mass_Q_d_RL[1,2]])
                    # #Absolute value ok?
                    cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_u_L[0,1]**2, 
                                    mass_Q_d_L[0,2]**2,mass_Q_d_L[1,2]**2])
                    cs = np.abs(cs)/(Delta2[k]/(1000**2))
                    Lambda_eff = np.sqrt(1/cs)
                    Lambda_effs_list[k-1].append(Lambda_eff)
                    L_diffs = Lambda_eff - real_Lambda_effs
                    tot_L_diffs.extend(L_diffs)
                    if k == 1:
                        if save_Z_checks:
                            if L_diffs[0] >= 0:
                                #print("found sd")
                                sd_list.extend(y_model)
                            if L_diffs[1] >= 0:
                                #print("found uc")
                                uc_list.extend(y_model)
                            if L_diffs[2] >= 0:
                                #print("found bd")
                                bd_list.extend(y_model)
                            if L_diffs[3] >= 0:
                                #print("found bs")
                                bs_list.extend(y_model)


                tot_L_diffs = np.array(tot_L_diffs)
                finite_idx = np.isfinite(tot_L_diffs)
                tot_L_diffs = tot_L_diffs[finite_idx]

                all_diffs.append(tot_L_diffs)

                #if (tot_L_diffs[4:] > 0).all():
                if not looping:
                    if (tot_L_diffs > 0).all():
                    #if (tot_L_diffs[2:] > 0).all() and tot_L_diffs[0] > 0:
                        if valid_model_check:
                            print("\nFound valid model")
                        
                            g_model = g_model_list[y_model[4]]
                            print(f"y_model: {y_model}")
                            print(f"g_model: {g_model}")
                            print(f"M_Us: {real_M_Us}")
                            print(f"M_Ds: {real_M_Ds}")
                            print(f"Cost: {cost}")
                        valid_model_list.extend(y_model)
            if save_Z_checks:
                np.savez("sd_list_wide.npz", *sd_list)
                np.savez("uc_list_wide.npz", *uc_list)
                np.savez("bd_list_wide.npz", *bd_list)
                np.savez("bs_list_wide.npz", *bs_list)
            if valid_model_check:
                np.savez(valid_filename, *valid_model_list)
                print("Saved to " + valid_filename)

            avg_cost = np.average(cost_list)
            print(f"The average cost function was {avg_cost}")
            compare_arr = np.array(compare_list)
            compare_avgs = np.average(compare_arr, axis = 0)
            # 9 CKMS , 24, 3 mus, 3 mds
            print(compare_avgs[9:33])

            tan_beta_arr = np.array(tan_beta_list)
            Z_strings = ["Z", "Z_3", "Z_3'", "Z_{12}"]
            c_strings = [r"\Lambda_{sd}^{\text{eff}}", r"\Lambda_{uc}^{\text{eff}}", r"\Lambda_{bd}^{\text{eff}}",
                        r"\Lambda_{bs}^{\text{eff}}"]
            simp_c_strings = ["sd_LL", "uc_LL", "bd_LL", "bs_LL"]

            fig, axs = plt.subplots(1,4,figsize=(7,2.5))
            lines = [None]*5
            if looping:
                indices = np.arange(0,4)
                indices = np.delete(indices, np.where(indices == good_index))
                plus_list = [i for i in range(len(y_model_list)) if all_diffs[i][indices[0]] > 0]
                cross_list = [i for i in range(len(y_model_list)) if all_diffs[i][indices[1]] > 0]
                star_list = [i for i in range(len(y_model_list)) if all_diffs[i][indices[2]] > 0]
                possibles = np.arange(0,len(y_model_list))
                dot_list = [i for i in possibles if i not in plus_list and i not in cross_list and i not in star_list]
            
            for n in range(len(cs)):
                if n > 3:
                    r = 1
                    xn = n-4
                else:
                    r = 0
                    xn = n 
                #fig = plt.figure(figsize=(3.5,3))
                if not looping:
                    end_range = 5
                else:
                    end_range = 2
                for k in range(1,end_range):
                    Lambda_effs = np.array(Lambda_effs_list[k-1])
                    if scatter_index:
                        x_array = np.arange(0,len(y_model_list))
                        limit_start = 0
                        limit_end = len(y_model_list)
                        xlabel = "y-model index"
                    elif scatter_tan_beta:
                        x_array = tan_beta_arr
                        limit_start = np.min(tan_beta_arr)
                        limit_end = np.max(tan_beta_arr)
                        xlabel = r"tan$(\beta)$"
                    elif scatter_m_v_ratio:
                        #v_arr = np.array([g_model_list[g_idx][1,1] for g_idx in g_idx_list])
                        
                        v_arr , _ = calc_vs(tan_beta_arr, conts.v_H)#tan_beta_arr*np.sqrt(conts.v_H**2/(1+tan_beta_arr**2))
                        #v_arr = np.array(v_list)
                        #v_arr = np.array(g_model_list[g_idx_list][1,1])
                        M23_arr = np.array([model[2][0] for model in y_model_list])
                        g_idx_arr = np.array([model[4] for model in y_model_list])
                        #print(g_idx_arr)
                        #x_array = np.array(m_u_list)/v_arr
                        x_array = v_arr/M23_arr
                       # x_array = g_idx_arr
                        xlabel = "$v_u/M_{[23]}$"
                        
                    if not looping:
                        line = axs[xn].scatter(x_array,Lambda_effs[:,n], s = 5, label = f"${Z_strings[k-1]}$", zorder = 5-k, alpha = 0.5)
                        lines[k-1] = line
                    else:
                        lines[0] = axs[xn].scatter(x_array[plus_list],Lambda_effs[plus_list,n], s = None, label = f"${simp_c_strings[indices[0]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 4, alpha = 0.5, marker = '+')
                        lines[1] = axs[xn].scatter(x_array[cross_list],Lambda_effs[cross_list,n], s = None, label = f"${simp_c_strings[indices[1]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 3, alpha = 0.5, marker = 'x')
                        lines[2] = axs[xn].scatter(x_array[star_list],Lambda_effs[star_list,n], s = None, label = f"${simp_c_strings[indices[2]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 2, alpha = 0.5, marker = '*')
                        lines[3] = axs[xn].scatter(x_array[dot_list],Lambda_effs[dot_list,n], s = 5, label = f"${simp_c_strings[good_index][:2]}$", zorder = 1, alpha = 0.5)
                    
                #limit_start = np.min(x_array)
                #limit_end = np.max(x_array)
                #print(limit_start)
                #print(limit_end)
                limit_start = 0.001
                limit_end = 0.2
                        
                line = axs[xn].hlines(real_Lambda_effs[n], limit_start, limit_end, color = "black", label = "Limit", zorder = 5)
                lines[4] = line
                #plt.scatter(tan_beta_list, m_u_list)
                #plt.scatter(np.array(v_chi_list)[g_idx_list], m_u_list, s = 2)
                #axs[r,xn].set_title(f"${c_strings[n]}$" )
                f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                f.set_scientific(False)
                f.set_powerlimits((-10,10))
                f2 = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                f2.set_scientific(False)
                f2.set_powerlimits((-10,10))
                #f2.set_locs([0.02,0.2])
                
                axs[xn].set_xlabel(xlabel)
                axs[xn].set_ylabel(f"${c_strings[n]}$ [log(TeV)]")
                axs[xn].set_yscale("log")
                #axs[xn].set_xlim([limit_start, limit_end])
                axs[xn].set_xscale("log")
                axs[xn].set_xlim([0.01, 0.2])
                axs[xn].set_xticks(ticks = [0.01, 0.1], labels = ["0.01","0.1"])
                axs[xn].set_xticks(ticks= [0.02, 0.03 ,0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2],labels = ["","","","","","", "","","0.2"],minor=True) # note that with a log axis, you can't have x = 0 so that value isn't plotted.
                
                #plt.ylim(0,10000)
                # if n == 6:
                #     axs[r,xn].legend(loc='lower center', bbox_to_anchor = (0,-1), ncol=5)
                # #axs[n].tight_layout()
            if not looping:
                plt.suptitle(r"Each $\Lambda^{\text{eff}}_{ij}$ for each y-model and neutral massive boson")
            else:
                plt.suptitle(r"$\Lambda^{\text{eff}}_{ij}$ for models with fulfilled " + simp_c_strings[good_index][:2] + " constraint for $Z$-boson")
            #plt.legend(loc='upper center', bbox_to_anchor=(0.1, -0.8), ncol=5)
            #plt.ticklabel_format(style = "plain")
            
            fig.legend(handles = lines, loc = "lower center", ncols = 5, bbox_to_anchor = (0.5,0))
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            plt.savefig(f"figs/" + lambda_filename)
            #plt.savefig(f"figs/Lambda_effs_{simp_c_strings[n]}.png")

        
          