#####################################################################################
# Module for model SU(3) x SU(2) x U^3_Y(1) x U^12_B-L(1) x U^2_I3R(1) x U^1_I3R(1) #
#####################################################################################
import flavorstuff as fs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from scipy.optimize import least_squares
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

def get_delta2_V(g, gs, vs, alt = True):
    [g1, g2, g3, g4] = gs.flatten()
    if alt: 
        [vH, vchi_q, vchi_l, vphi, vsigma] = vs.flatten()
        M2 = np.array([[g**2*vH**2/8, -g*g1*vH**2/8, 0, 0, 0], 
                [-g*g1*vH**2/8, g1**2*vH**2/8 + g1**2*vchi_l**2/8 + g1**2*vchi_q**2/72 + g1**2*vphi**2/8, -g1*g2*vchi_l**2/4 - g1*g2*vchi_q**2/36, -g1*g3*vphi**2/8, 0], 
                [0, -g1*g2*vchi_l**2/4 - g1*g2*vchi_q**2/36,  g2**2*vchi_l**2/2 + g2**2*vchi_q**2/18, 0, 0], 
                [0, -g1*g3*vphi**2/8, 0, g3**2*vphi**2/8 + g3**2*vsigma**2/8, -g3*g4*vsigma**2/8], 
                [0, 0, 0, -g3*g4*vsigma**2/8, g4**2*vsigma**2/8]])
        (Delta2, V) = np.linalg.eigh(M2)
    else:    
        [vH, vchi, vphi, vsigma] = vs.flatten()
        M2 = np.array([[g**2*vH**2/8, -g*g1*vH**2/8, 0, 0, 0], 
                [-g*g1*vH**2/8, g1**2*vH**2/8 + 5*g1**2*vchi**2/36 + g1**2*vphi**2/8, -5*g1*g2*vchi**2/18, -g1*g3*vphi**2/8, 0], 
                [0, -5*g1*g2*vchi**2/18, 5*g2**2*vchi**2/9, 0, 0], 
                [0, -g1*g3*vphi**2/8, 0, g3**2*vphi**2/8 + g3**2*vsigma**2/8, -g3*g4*vsigma**2/8], 
                [0, 0, 0, -g3*g4*vsigma**2/8, g4**2*vsigma**2/8]])
        (Delta2, V) = np.linalg.eigh(M2)
    return Delta2, V

def Q_compare_gauge(field, base, base_SM):
    sm_Q_d_L = np.diag(sm_Q(f"{field}_L", base_SM))
    sm_Q_d_R = np.diag(sm_Q(f"{field}_R", base_SM))

    Q_d_L = np.diag(build_Q(f"{field}_L", base))
    Q_d_R = np.diag(build_Q(f"{field}_R", base))

    #compare_Q = np.append(np.abs(Q_d_L[:3])-np.abs(sm_Q_d_L),np.abs(Q_d_R[:3])-np.abs(sm_Q_d_R))
    compare_Q = np.abs(Q_d_L[:3])-np.abs(sm_Q_d_L)
    return compare_Q

def comparing_e(gs, g, e_em):
    [g1, g2, g3, g4] = gs.flatten()
    norm = np.sqrt(g4**2 * (1/g**2 + 1/g1**2 + 1/(4*g2**2) + 1/g3**2) + 1)
    #eq = np.array([g4/g, g4/g1, g4/(2*g2), g4/g3, 1])
    eq = np.array([g4/g, 1, g4/(2*g2)])

    return (eq/norm - e_em)/e_em

# Builds the yukawa matrix as specified in the article. 
# VEVs are treated unnormalized prior to this
def build_yukawa(ys, ms, vs, alt = False, alt_v = False):
    vs = vs/np.sqrt(2)
    if alt_v:
        [v, v_chi_q, v_chi_l, v_phi, v_sigma] = vs.flatten()
    else:
        [v, v_chi_q, v_phi, v_sigma] = vs.flatten()
    m = np.zeros((3,3))
    m[2,2] = v*ys[0]
    Delta_L = v*np.array([[ys[1], ys[2], 0],[ys[3], ys[4],0], [0,0,0]])
    Delta_R = np.array([[0, ys[5]*v_phi, 0], [0, ys[6]*v_phi, ys[7]*v_chi_q], [ys[8]*v_sigma,0,0]])
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
    compare_Q = np.append((np.diag(mass_Q_d_L)[:3]-np.abs(sm_Q_d_L))/np.abs(sm_Q_d_L),(np.diag(mass_Q_d_R)[:3]-np.abs(sm_Q_d_R))/np.abs(sm_Q_d_R))
    return compare_Q

def y_compare_es(U, gs, all_e = False):
    [g1, g2, g3, g4] = gs.flatten()
    norm = np.sqrt(g4**2 * (1/g**2 + 1/g1**2 + 1/(4*g2**2) + 1/g3**2) + 1)
    # I3L, Y[3], B-L[12], I3R[2], I3R[1]
    return [g4/norm - conts.e_em]
    '''
        eq = np.array([g4/g, g4/g1, g4/(2*g2), g4/g3, 1])
        mats = []
        if all_e:
            mats.append(g*np.array([g4/g, g4/g, g4/g, 0, 0, 0]))
            mats.append(g2*np.array([g4/(2*g2), g4/(2*g2), 0, g4/(2*g2),g4/(2*g2),g4/(2*g2)]))
            mats.append(g4*np.array([1,0,0,0,0,0]))
            mats.append(g3*np.array([0,g4/g3,0,0,0,g4/g3]))
            mats.append(g1*np.array([0,0,g4/g1, g4/g1, g4/g1, 0]))
        else:
            mats.append(np.array([g4/g, g4/g, g4/g, 0, 0, 0]))
            #mats.append(np.array([g4/(2*g2), g4/(2*g2), 0, g4/(2*g2),g4/(2*g2),g4/(2*g2)]))
            #mats.append(np.array([1,0,0,0,0,0]))
        #    mats.append(np.array([0,g4/g3,0,0,0,g4/g3]))
        #    mats.append(np.array([0,0,g4/g1, g4/g1, g4/g1, 0]))
        
        mats /= norm
        compares = []
        for mat in mats:
            zero_idx = np.where(mat[:3] == 0)
            some_idx = np.where(mat[:3] > 0)
            mass_mat = fs.mass_Q(U, np.diag(mat))
            #compares.append(np.diag(mass_mat[zero_idx]))
            compares.append((np.diag(mass_mat[some_idx]) - conts.e_em))
        return np.concatenate(compares)
    '''

def solve_for_all(thetas, g, g_prim, v_u, v_d, v_chi,m_us, m_ds, all_e = False):
    ys_u = thetas[:14]
    ys_d = thetas[14:28]
    gs = thetas[28:32]
    M_Us = thetas[32:35]
    vs_small = thetas[35:]
    
    vs = np.concatenate(([conts.v_H], vs_small), axis = 0)
    v_us = np.concatenate(([v_u], vs_small), axis = 0)
    v_ds = np.concatenate(([v_d], vs_small), axis = 0)
    #if vs[1] > M_Us[0] or vs[2] > M_Us[0] or vs[3] > M_Us[0] or vs[4] > M_Us[2]:
    #    return 1000

    Delta2, V = get_delta2_V(g,gs,vs, alt = False)
    yukawas_u = build_yukawa(ys_u, M_Us, v_us, alt = True, alt_v=False)
    yukawas_d = build_yukawa(ys_d, M_Us, v_ds, alt = True, alt_v = False)

    Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
    Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)

    m_u_compare = (diag_yukawa_u[:3] - m_us)
    m_d_compare = (diag_yukawa_d[:3] - m_ds)
    #print(np.sqrt(2*Delta2))
    if Delta2[1] < 0:
        m_z_compare = [1000]
    else:
        m_z_compare = [(np.sqrt(2*Delta2[1])- conts.m_Z)]#, (np.sqrt(2*Delta2[2])- 4000)]

    compare_Qs = np.concatenate((y_compare_es(U_d_L, gs, all_e), y_compare_es(U_u_L, gs, all_e)))
                                 #y_compare_es(Uh_d_R.T, gs, all_e), y_compare_es(Uh_u_R.T, gs, all_e)))
    base_SM_Z = fs.get_base_Z(g,g_prim)
    base_SM_gamma = fs.get_base_gamma()
    #compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R), Q_compare("u", V[:,0], base_SM_gamma, U_u_L, Uh_u_R),
    #                             Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R), Q_compare("d", V[:,0], base_SM_gamma, U_d_L, Uh_d_R)))
    
#    compare_Qs = np.concatenate((Q_compare("u", V[:,1], base_SM_Z, U_u_L, Uh_u_R),
#                                 Q_compare("d", V[:,1], base_SM_Z, U_d_L, Uh_d_R)))
    V_CKM_arr = np.array(np.abs(conts.V_CKM))
    V_CKM_calc = np.dot(np.transpose(U_u_L), U_d_L) 
    CKM_compare = (np.abs(V_CKM_calc[:3,:3])-V_CKM_arr).flatten()/V_CKM_arr.flatten()
    #CKM_compare[[2,5,6,7]] = CKM_compare[[2,5,6,7]]*V_CKM_arr.flatten()[[2,5,6,7]]
    #CKM_compare[[2,6]] = CKM_compare[[2,6]]*V_CKM_arr.flatten()[[2,6]]
    compares = np.concatenate((compare_Qs,CKM_compare, m_u_compare, m_d_compare, m_z_compare))
    #compares = compare_Qs
    return compares

def get_models(g_filename, y_filename, g, g_prim, search_for_models = False, verbose = False, tol = 0.01, max_iters = 100):
    if search_for_models:
        successes = 0
        great_count = 0
        real_Lambda_effs = get_lambda_limits()
        V_CKM_arr = np.array(np.abs(conts.V_CKM))
        y_model_list = []
        g_model_list = []
        m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
        m_us = np.array([conts.m_u, conts.m_c, conts.m_t])

        v_chis = np.arange(5000,100000,100)
        M_23s = np.arange(5000,100000,1000)
        low_bound = np.ones(28)*-5
        low_bound = np.append(low_bound, np.ones(4)*0.01)
        low_bound = np.append(low_bound, [1000,1000,8000, 1000,1000,8000])
        
        high_bound = np.ones(32)*5
        high_bound = np.append(high_bound, [100000,100000,1000000,100000,500000,4000000])
        for idx, v_chi in enumerate(tqdm(v_chis)):
            if idx % 50 == 0:
                print(f"\nSuccesses: {successes}")
                print(f"Great count: {great_count}")
            valid_idxs = np.argwhere(M_23s > v_chi).flatten()
            for idx in valid_idxs:
                M_23 = M_23s[idx]
                v_phi = int(v_chi*np.random.uniform(1.5,5))
                v_sigma = int(v_phi*np.random.uniform(2,8))
                gs = np.random.uniform(1,5,4)
                #vs = np.array([conts.v_H, v_chi, v_phi, v_sigma])
                tan_beta = np.random.randint(10,100)
                #tan_beta = 10
                #M_23 = np.random.randint(1000,10000)
                
                #M_Us = np.array([M_23, M_23, M_12])
                v_u, v_d = calc_vs(tan_beta, conts.v_H)
                #v_us = np.array([v_u, vs[1], vs[2], vs[3]])
                #v_ds = np.array([v_d, vs[1], vs[2], vs[3]])

                y3_u = np.sqrt(2)*conts.m_t/v_u
                y3_d = np.sqrt(2)*conts.m_b/v_d
                ys = np.random.uniform(0.01,5,28)
                
                neg_indices = np.random.random(len(ys)) > 0.5
                np.negative(ys, where=neg_indices, out=ys)
                ys[0] = y3_u
                ys[14] = y3_d
                

                gs = np.random.uniform(0.01,5, 4)
                vs_small = np.concatenate(([v_chi], [v_phi], [v_sigma]), axis = 0)
                low_bound[[32, 33,34]] = 0.9*M_23, 0.9*M_23, 5*M_23
                low_bound[[35,36,37]] = 0.9*v_chi, 0.9*v_chi, 0.5*v_sigma
                #ys = np.append(ys, np.random.uniform(1000,10000,2))
                #ys = np.append(ys, np.random.uniform(10000,100000,1))
                thetas = np.concatenate((ys, gs , [M_23], [M_23], np.random.uniform(6*M_23,1000000,1), vs_small), axis = None)
                #rint(f"inits {thetas[32:]}")
                #print(f"high bounds: {high_bound[32:]}")
                #print(f"Low bounds: {low_bound[32:]}\n")
                #print([v_chi, v_phi, v_sigma])
                try:
                    res = least_squares(solve_for_all, thetas, args=(g,g_prim,v_u,v_d,[v_chi, v_phi, v_sigma],m_us,m_ds, True), 
                                loss = "linear", method= "trf", jac = "2-point", tr_solver="exact", max_nfev=max_iters, bounds = (low_bound,high_bound),
                                gtol = 1e-12, ftol = 1e-12, xtol = 1e-12) #
                except Exception as e:
                    print("Error in least squares: ", e)
                    continue
                thetas = res.x
                gs = thetas[28:32]
                l_gs = np.insert(gs, 0, g)
                ys = thetas[:28]
                vs_small = thetas[35:]
                vs = np.concatenate(([conts.v_H], vs_small), axis = 0)
                Delta2, V = get_delta2_V(g,gs,vs, alt = False)
                real_mzs = np.sqrt(2*Delta2[1:])
                M_Us = thetas[32:35]
                if verbose:
                    print(2*res.cost)
                if (all(gs <= 5)  and all(gs > 0.01) and all(np.abs(gs) <= 5)  and all(np.abs(ys) > 0.01) and np.abs(2*res.cost) < tol
                    and all(V[:,0] > 0) and all(real_mzs[1:] > 4000) and vs[1] < M_Us[0] and vs[2] < M_Us[0] and vs[3] < M_Us[2]):
                    y_ds = thetas[14:28]
                    y_us = thetas[:14]
                    g_tmp_list = [real_mzs, vs, gs]
                    g_model_list.append(g_tmp_list)
                    v_us = np.concatenate(([v_u], vs_small), axis = 0)
                    v_ds = np.concatenate(([v_d], vs_small), axis = 0)

                    y_tmp_list = [y_us, y_ds, M_Us, tan_beta, successes]
                    successes += 1
                    
                    if verbose:
                        print("Successful model")

                        yukawas_u = build_yukawa(y_us, M_Us, v_us, alt = True)
                        yukawas_d = build_yukawa(y_ds, M_Us, v_ds, alt = True)

                        Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                        Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)
                        real_M_Us = diag_yukawa_u[3:]
                        real_M_Ds = diag_yukawa_d[3:]

                        V_CKM_calc = U_u_L.T @ U_d_L
                        print(f"CKM compare: {(np.abs(V_CKM_calc[:3,:3])-V_CKM_arr)}")
                        #e_compares = solve_for_all(thetas,g,g_prim,v_u,v_d,[v_chi, v_phi, v_sigma],m_us,m_ds, all_e = True)[:30] 
                        #print(f"e-Compares: {e_compares} (res: {np.sum(e_compares**2)})")
                        #base_SM = fs.get_base_Z(g,g_prim)
                        # base_SM_gamma = np.array([0, conts.e_em])
                        # base_SM = np.array([(g**2)/np.sqrt(g**2 + g_prim**2), (g_prim**2)/np.sqrt(g**2 + g_prim**2)])
                        #base_SM = np.array([(g*g_prim)/np.sqrt(g**2 + g_prim**2), (g*g_prim)/np.sqrt(g**2 + g_prim**2)])
                        #sm_Q_d_L = np.diag(sm_Q("d_L", base_SM))
                        #sm_Q_d_R = np.diag(sm_Q("d_R", base_SM))
                        L_diffs = []
                        for k in range(1,4):
                            base = l_gs*V[:,k]
                            
                            Q_d_L = build_Q("d_L", base)
                            Q_u_L = build_Q("u_L", base)
                            
                            mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
                            mass_Q_u_L = fs.mass_Q(U_u_L, Q_u_L)
                            #Q_d_R = build_Q("d_R", base)
                            if k ==1:
                                sm_Q_d_L = sm_Q("d_L",fs.get_base_Z(g,g_prim)) 
                                sm_Q_d_R = sm_Q("d_R",fs.get_base_Z(g,g_prim)) 
                                Q_d_R = build_Q("d_R", base)
                                mass_Q_d_R = fs.mass_Q(Uh_d_R.T, Q_d_R)
                            
                                print(f"Charge difference L: {np.abs(np.diag(mass_Q_d_L[:3])) - np.abs(np.diag(sm_Q_d_L))}")
                                print(f"Charge difference R: {np.abs(np.diag(mass_Q_d_R[:3])) - np.abs(np.diag(sm_Q_d_R))}")
                            
                            #TODO: Compare diagonal charges with SM
                            cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_u_L[0,1]**2, 
                                        mass_Q_d_L[0,2]**2,mass_Q_d_L[1,2]**2])
                            cs = np.abs(cs)/(Delta2[k]/(1000**2))
                            Lambda_eff = np.sqrt(1/cs)
                            L_diffs.extend(Lambda_eff - real_Lambda_effs)
                        print(L_diffs)
                        if (np.array(L_diffs) > 0).all():
                            great_count += 1
                            print("*******************")
                            print(f"Found great model!")
                            print("*******************")
                        print(f"Cost: {2*res.cost}")
                        print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
                        print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
                        print(f"gs: {gs}")
                        print(f"vs : {vs}")
                        print(f"Real mzs: {np.sqrt(2*Delta2)}")
                        print(f"Diff m_Zs = {np.sqrt(2*Delta2[1])- conts.m_Z}")  
                        print(f"M_Us: {M_Us}")
                        print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {successes}")
                    y_model_list.extend(y_tmp_list)
        if y_model_list:
            np.savez(g_filename, *g_model_list)
            np.savez(y_filename, *y_model_list)
    
    y_models = np.load(y_filename)
    ex_model_list = [y_models[k] for k in y_models]
    y_model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(y_model_list)}")
    g_models = np.load(g_filename)
    g_model_list = [g_models[k] for k in g_models]
    print(f"The number of g_models is: {len(g_model_list)}")

    return y_model_list, g_model_list
          
def refine_models(in_y_model_list,in_g_model_list,g_filename, y_filename, g, g_prim, verbose = False, tol = 0.01, max_iters = 100):
    successes = 0
    great_count = 0
    real_Lambda_effs = get_lambda_limits()
    V_CKM_arr = np.array(np.abs(conts.V_CKM))
    y_model_list = []
    g_model_list = []
    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])

    low_bound = np.ones(28)*-5
    low_bound = np.append(low_bound, np.ones(4)*0.01)
    low_bound = np.append(low_bound, [1000,1000,8000, 1000,1000,8000])
    
    high_bound = np.ones(32)*5
    high_bound = np.append(high_bound, [100000,100000,1000000,100000,500000,4000000])
    for idx, y_model in enumerate(tqdm(in_y_model_list)):
        if idx % 50 == 0:
            print(f"\nSuccesses: {successes}")
            print(f"Great count: {great_count}")
        
        g_model = in_g_model_list[y_model[4]]
        
        M_23 = y_model[2][0]
        M_23_prim = y_model[2][1]
        M_12 = y_model[2][2]
        [_, v_chi, v_phi, v_sigma] = g_model[1]
        gs = g_model[2]
        tan_beta = y_model[3]

        v_u, v_d = calc_vs(tan_beta, conts.v_H)
        
        ys = np.concatenate((y_model[0], y_model[1]))
        vs_small = np.concatenate(([v_chi], [v_phi], [v_sigma]), axis = 0)
        low_bound[[32, 33,34]] = 0.9*M_23, 0.9*M_23_prim, 0.9*M_12
        low_bound[[35,36,37]] = 0.9*v_chi, 0.9*v_phi, 0.9*v_sigma

        thetas = np.concatenate((ys, gs , [M_23], [M_23_prim], [M_12], vs_small), axis = None)
        #print(f"inits {thetas[32:]}")
        #print(f"high bounds: {high_bound[32:]-thetas[32:]}")
        #print(f"Low bounds: {thetas[32:]-low_bound[32:]}\n")
     
        try:
            res = least_squares(solve_for_all, thetas, args=(g,g_prim,v_u,v_d,[v_chi, v_phi, v_sigma],m_us,m_ds, True), 
                        loss = "linear", method= "trf", jac = "2-point", tr_solver="exact", max_nfev=max_iters, bounds = (low_bound,high_bound),
                        gtol = 1e-12, ftol = 1e-12, xtol = 1e-12) #
        except Exception as e:
            print("Error in least squares: ", e)
            continue
        thetas = res.x
        gs = thetas[28:32]
        l_gs = np.insert(gs, 0, g)
        ys = thetas[:28]
        vs_small = thetas[35:]
        vs = np.concatenate(([conts.v_H], vs_small), axis = 0)
        Delta2, V = get_delta2_V(g,gs,vs, alt = False)
        real_mzs = np.sqrt(2*Delta2[1:])
        M_Us = thetas[32:35]
        if verbose:
            print(2*res.cost)
        if (all(gs <= 5)  and all(gs > 0.01) and all(np.abs(gs) <= 5)  and all(np.abs(ys) > 0.01) and np.abs(2*res.cost) < tol
            and all(V[:,0] > 0) and all(real_mzs[1:] > 4000) and vs[1] < M_Us[0] and vs[2] < M_Us[0] and vs[3] < M_Us[2]):
            y_ds = thetas[14:28]
            y_us = thetas[:14]
            g_tmp_list = [real_mzs, vs, gs]
            g_model_list.append(g_tmp_list)
            v_us = np.concatenate(([v_u], vs_small), axis = 0)
            v_ds = np.concatenate(([v_d], vs_small), axis = 0)

            y_tmp_list = [y_us, y_ds, M_Us, tan_beta, successes]
            successes += 1
            
            if verbose:
                print("Successful model")

                yukawas_u = build_yukawa(y_us, M_Us, v_us, alt = True)
                yukawas_d = build_yukawa(y_ds, M_Us, v_ds, alt = True)

                Uh_u_R, diag_yukawa_u, U_u_L = fs.diag_yukawa(yukawas_u)
                Uh_d_R, diag_yukawa_d, U_d_L = fs.diag_yukawa(yukawas_d)
                real_M_Us = diag_yukawa_u[3:]
                real_M_Ds = diag_yukawa_d[3:]

                V_CKM_calc = U_u_L.T @ U_d_L
                print(f"CKM compare: {(np.abs(V_CKM_calc[:3,:3])-V_CKM_arr)}")

                L_diffs = []
                for k in range(1,4):
                    base = l_gs*V[:,k]
                    
                    Q_d_L = build_Q("d_L", base)
                    Q_u_L = build_Q("u_L", base)
                    
                    mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
                    mass_Q_u_L = fs.mass_Q(U_u_L, Q_u_L)
                    #Q_d_R = build_Q("d_R", base)
                    if k ==1:
                        sm_Q_d_L = sm_Q("d_L",fs.get_base_Z(g,g_prim)) 
                        sm_Q_d_R = sm_Q("d_R",fs.get_base_Z(g,g_prim)) 
                        Q_d_R = build_Q("d_R", base)
                        mass_Q_d_R = fs.mass_Q(Uh_d_R.T, Q_d_R)
                    
                        print(f"Charge difference L: {np.abs(np.diag(mass_Q_d_L[:3])) - np.abs(np.diag(sm_Q_d_L))}")
                        print(f"Charge difference R: {np.abs(np.diag(mass_Q_d_R[:3])) - np.abs(np.diag(sm_Q_d_R))}")
                    
                    #TODO: Compare diagonal charges with SM
                    cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_u_L[0,1]**2, 
                                mass_Q_d_L[0,2]**2,mass_Q_d_L[1,2]**2])
                    cs = np.abs(cs)/(2*Delta2[k]/(1000**2))
                    Lambda_eff = np.sqrt(1/cs)
                    L_diffs.extend(Lambda_eff - real_Lambda_effs)
                print(L_diffs)
                if (np.array(L_diffs) > 0).all():
                    great_count += 1
                    print("*******************")
                    print(f"Found great model!")
                    print("*******************")
                print(f"Cost: {2*res.cost}")
                print(f"mds diffs: {diag_yukawa_d[:3]-m_ds}")
                print(f"mus diffs: {diag_yukawa_u[:3]-m_us}")
                print(f"gs: {gs}")
                print(f"vs : {vs}")
                print(f"Real mzs: {np.sqrt(2*Delta2)}")
                print(f"Diff m_Zs = {np.sqrt(2*Delta2[1])- conts.m_Z}")  
                print(f"M_Us: {M_Us}")
                print(f"y_us: {y_us}\n y_ds: {y_ds}\n M_Us: {real_M_Us}\n M_Ds: {real_M_Ds}\n tan_beta = {tan_beta}\n g_idx = {successes}")
            y_model_list.extend(y_tmp_list)
    if y_model_list:
        np.savez(g_filename, *g_model_list)
        np.savez(y_filename, *y_model_list)

    y_models = np.load(y_filename)
    ex_model_list = [y_models[k] for k in y_models]
    y_model_list = [[ex_model_list[i], ex_model_list[i+1],ex_model_list[i+2],ex_model_list[i+3],ex_model_list[i+4]] for i in np.arange(0,len(ex_model_list),5)]
    print(f"The number of y_models is: {len(y_model_list)}")
    g_models = np.load(g_filename)
    g_model_list = [g_models[k] for k in g_models]
    print(f"The number of g_models is: {len(g_model_list)}")

    return y_model_list, g_model_list

def calc_vs(tan_beta, v_H):
    v_d= np.sqrt(v_H**2/(1+tan_beta**2))
    v_u = tan_beta*v_d
    return v_u, v_d

def calc_grid():
    v_chis = np.arange(5000,100000,100)
    M_23s = np.arange(5000,100000,1000)
    c_sum = 0
    for v_chi in tqdm(v_chis):
        valid_idxs = np.argwhere(M_23s > v_chi).flatten()
        c_sum += len(valid_idxs)
    return c_sum

def calc_confidence(x_arr,slope, intercept, slope_err, intercept_err):
    slope_space = np.linspace(slope-slope_err, slope+slope_err,100)
    intercept_space = np.linspace(intercept-intercept_err, intercept+intercept_err,100)
    mean_ys = []
    low_ys = []
    high_ys = []
    for x in x_arr:
        y_arr = slope_space*x
        y_arr = (y_arr.reshape(-1,1)+intercept_space.reshape(1,-1)).flatten()
        std = np.std(y_arr)
        mean = np.mean(y_arr)
        mean_ys.append(mean)
        low_ys.append(mean - std)
        high_ys.append(mean +  std)
    return mean_ys, low_ys, high_ys


if __name__ == "__main__":
    # According to definition
    g = conts.e_em/np.sqrt(conts.sw2)
    g_prim = conts.e_em/np.sqrt(1-conts.sw2)
    
    # Get models for gs
    search_for_all = False
    g_plotting = False
    y_plotting = True
    picking_gs = False
    refining_ys = False
    looping = True
    #refine_filename = "p_refined_valid_wide.npz"

    y_refine_filename = "all_y_refine.npz"
    g_refine_filename = "all_g_refine.npz"
    valid_filename = "e_valid.npz"

    y_filename = "all_y_models_2.npz"
    y_filename = "all_y_refine.npz"
    #y_filename = "e_valid.npz"
    g_filename = "all_g_models_2.npz"
    g_filename = "all_g_refine.npz"
    
    y_filenames = ["e_sd_list.npz", "e_uc_list.npz", "e_bd_list.npz", "e_bs_list.npz"]
    y_model_list, g_model_list = get_models(g_filename, y_filename, g, g_prim, search_for_all, verbose = False,
                                                tol = 0.1, max_iters=200)
    if refining_ys:
        y_model_list, g_model_list = refine_models(y_model_list, g_model_list,g_refine_filename, y_refine_filename, g, g_prim, verbose = False,
                                                tol = 0.1, max_iters=None)
    
    if looping:
        y_model_lists = []
        for y_filename in y_filenames:
            y_model_list, g_model_list = get_models(g_filename, y_filename, g, g_prim, search_for_all, verbose = False,
                                                tol = 0.1, max_iters=200)
            y_model_lists.append(y_model_list)
    else:
        y_model_lists = [y_model_list]


    if g_plotting:
        f  = open("saved_g_regrs.txt", "w")
        g1_list = [model[2,2] for model in g_model_list]
        g2_list = [model[2,3] for model in g_model_list]
        g12_list = [model[2,1]+ model[2,0] for model in g_model_list]
        print(np.mean(g12_list))
        print(np.mean(g1_list))
        print(np.mean(g2_list))
        
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
            print(res.intercept_stderr)
            stderr = res.stderr

            mean_y, low_y, high_y = calc_confidence(v_chi_lin, slope, intercept, stderr, res.intercept_stderr)

            fig = plt.figure(figsize=(3.5,4))
            plt.scatter(v_s, m_zs, s = 2, label = "Data pts", zorder = 1)
            plt.plot(v_chi_lin, v_chi_lin*slope + intercept, "r", label = f"gx + m, g = {slope:.3f}, m = {intercept:.3f}", zorder = 2)
            plt.fill_between(v_chi_lin, low_y, high_y, alpha = 0.5, color = "r",zorder=3)
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
            cs = []
            tot_Q_d_comp = []
            tot_Q_u_comp = []
            sm_Q_dl = sm_Q("d_L", fs.get_base_Z(g,g_prim))
            sm_Q_ul = sm_Q("u_L", fs.get_base_Z(g,g_prim))
                             
            for y_model in y_model_list:
                #print(y_model)
                y_us = np.array(y_model[0])
                y_ds = np.array(y_model[1])
                M_Us = y_model[2]
                tan_beta= y_model[3]
                g_idx = y_model[4]
                vs = g_model_list[g_idx][1,:]
                v_list.append(vs[1])
                gs = g_model_list[g_idx][2,:]
                Delta2, V = get_delta2_V(g, gs, vs, alt = False)   

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
                in_ys = np.concatenate((y_us, y_ds, gs, M_Us, vs[1:]))             
                compares = solve_for_all(in_ys, g, g_prim, v_u, v_d, vs, m_us, m_ds, True)
                compare_list.append(compares)
                cost = np.sum(compares**2)
                cost_list.append(cost)
                l_gs = np.insert(gs, 0, g)
                
                for k in range(1,5):
                    base = l_gs*V[:,k]
                    Q_d_L = build_Q("d_L", base)
                    Q_d_R = build_Q("d_R", base)

                    Q_u_L = build_Q("u_L", base)
                    Q_u_R = build_Q("u_R", base)

                    mass_Q_d_L = fs.mass_Q(U_d_L, Q_d_L)
                  
                    mass_Q_u_L = fs.mass_Q(U_u_L, Q_u_L)
                    cs = np.array([mass_Q_d_L[0,1]**2, mass_Q_u_L[0,1]**2, 
                                    mass_Q_d_L[0,2]**2,mass_Q_d_L[1,2]**2])
                    
                    cs = np.abs(cs)/(2*Delta2[k]/(1000**2))
                    #print(cs)
                    Lambda_eff = np.sqrt(1/cs)
                    Lambda_effs_list[k-1].append(Lambda_eff)
                    L_diffs = Lambda_eff - real_Lambda_effs
                    if k == 1:
                        tot_L_diffs.extend(L_diffs)
                    if k == 1:
                        Q_compares_d = np.abs(np.diag(mass_Q_d_L)[:3])-np.diag(np.abs(sm_Q_dl))
                        #print(f"Q-compares d: {Q_compares_d}")
                        Q_compares_u = np.abs(np.diag(mass_Q_u_L)[:3])-np.diag(np.abs(sm_Q_ul))
                        #print(f"Q-compares u: {Q_compares_u}")
                        
                        tot_Q_d_comp.append(np.sum(Q_compares_d**2))
                        tot_Q_u_comp.append(np.sum(Q_compares_u**2))

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

                if not looping:
                    if (tot_L_diffs > 0).all():
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
                np.savez("e_sd_list.npz", *sd_list)
                np.savez("e_uc_list.npz", *uc_list)
                np.savez("e_bd_list.npz", *bd_list)
                np.savez("e_bs_list.npz", *bs_list)
            if valid_model_check:
                np.savez(valid_filename, *valid_model_list)
                print("Saved to " + valid_filename)

            avg_cost = np.average(cost_list)
            print(f"The average cost function was {avg_cost}")
            compare_arr = np.array(compare_list)
            compare_avgs = np.average(compare_arr, axis = 1)

            tot_Q_d_comp = np.average(tot_Q_d_comp)
            tot_Q_u_comp = np.average(tot_Q_u_comp)
            print(f" avg Cost Qd : {tot_Q_d_comp}")
            print(f" avg Cost Qu : {tot_Q_u_comp}")
           
            # 9 CKMS , 24, 3 mus, 3 mds
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
                print(len(plus_list))
            
            for n in range(len(cs)):
                if n > 3:
                    r = 1
                    xn = n-4
                else:
                    r = 0
                    xn = n 
                if not looping:
                    end_range = 5
                else:
                    end_range = 2
                for k in range(1,end_range):
                    Lambda_effs = np.array(Lambda_effs_list[k-1])
                    v_arr , _ = calc_vs(tan_beta_arr, conts.v_H)
                    M23_arr = np.array([model[2][0] for model in y_model_list])
                    g_idx_arr = np.array([model[4] for model in y_model_list])
                    x_array = v_arr/M23_arr
                    xlabel = "$v_u/M_{[23]}$"
                    
                    if not looping:
                        line = axs[xn].scatter(x_array,Lambda_effs[:,n], s = 5, label = f"${Z_strings[k-1]}$", zorder = 5-k, alpha = 0.5)
                        lines[k-1] = line
                    else:
                        lines[0] = axs[xn].scatter(x_array[plus_list],Lambda_effs[plus_list,n], s = None, label = f"${simp_c_strings[indices[0]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 4, alpha = 0.5, marker = '+')
                        lines[1] = axs[xn].scatter(x_array[cross_list],Lambda_effs[cross_list,n], s = None, label = f"${simp_c_strings[indices[1]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 3, alpha = 0.5, marker = 'x')
                        lines[2] = axs[xn].scatter(x_array[star_list],Lambda_effs[star_list,n], s = None, label = f"${simp_c_strings[indices[2]][:2]}, {simp_c_strings[good_index][:2]}$", zorder = 2, alpha = 0.5, marker = '*')
                        lines[3] = axs[xn].scatter(x_array[dot_list],Lambda_effs[dot_list,n], s = 5, label = f"${simp_c_strings[good_index][:2]}$", zorder = 1, alpha = 0.5)
                    
                limit_start = 240/100000
                limit_end = 0.02
                        
                line = axs[xn].hlines(real_Lambda_effs[n], limit_start, limit_end, color = "black", label = "Limit", zorder = 5)
                lines[4] = line
                f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                f.set_scientific(False)
                f.set_powerlimits((-10,10))
                f2 = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                f2.set_scientific(False)
                f2.set_powerlimits((-10,10))
                
                axs[xn].set_xlabel(xlabel)
                axs[xn].set_ylabel(f"${c_strings[n]}$ [log(TeV)]")
                axs[xn].set_yscale("log")
                axs[xn].set_xlim([limit_start, limit_end])
                axs[xn].set_xscale("log")
                #axs[xn].set_xlim([0.01, 0.2])
                axs[xn].set_xticks(ticks = [0.01], labels = [r"$10^{-2}$"])
                axs[xn].set_xticks(ticks= [0.002, 0.0024, 0.003 ,0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.02],labels = ["","$2.4\cdot10^{-3}$","","","","","", "","",""],minor=True) # note that with a log axis, you can't have x = 0 so that value isn't plotted.
            if not looping:
                plt.suptitle(r"Each $\Lambda^{\text{eff}}_{ij}$ for each y-model and neutral massive boson")
            else:
                plt.suptitle(r"$\Lambda^{\text{eff}}_{ij}$ for models with fulfilled " + simp_c_strings[good_index][:2] + " constraint for $Z$-boson")
            
            if len(y_model_list) != 0:
                fig.legend(handles = lines, loc = "lower center", ncols = 5, bbox_to_anchor = (0.5,0))
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            plt.savefig(f"figs/" + lambda_filename)
          
        
          