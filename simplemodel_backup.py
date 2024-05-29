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

def gs_eq_system(vH,vchi,vphi,vsigma,mz,mz3,mz3_prim,mz12):
    g,g1,g2,g3,g4,vH,vchi_q,vchi_l,vphi,vsigma = sp.symbols("g g1 g2 g3 g4 vH vchi_q vchi_l vphi vsigma", real=True)
    #g,g1,g2,g3,g4 = sp.symbols("g g1 g2 g3 g4", real=True)
    M_b = -sp.I*1/sp.sqrt(2)*sp.Matrix([[-g*vH/(2), g1*vH/(2), 0, 0 , 0],
                        [0, -g1*vchi_q/6, g2*vchi_q/3,              0,                  0],
                        [0, g1*vchi_l/2, -g2*vchi_l,                0,                  0],
                        [0, g1*vphi/2,              0,-g3*vphi/2,                   0],
                        [0, 0,                          0, g3*vsigma/2, -g4*vsigma/2]])
    print(M_b.H*M_b)
#     eigen_dict = M_b.singular_values()
#  #   print(eigen_dict)
#     eigen_vals = list(eigen_dict.keys())
#  #   print(eigen_vals)
#     res = sp.solve([eigen_vals[0], eigen_vals[1]-mz, eigen_vals[2]-mz3, eigen_vals[3]-mz3_prim, eigen_vals[4]-mz12], [g1,g2,g3,g4])
#     return res
 
    
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

def minimize_for_CKM(thetas,vs, g, g_prim, M_Us, m_us, M_Ds, m_ds, mzs):
    gs = thetas[:4]
    ys_u = thetas[4:17]
    ys_d = thetas[17:]
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*v_chi/6, gs[1]*v_chi/3, 0,0],
                        [0,gs[0]*v_chi/2, -gs[1]*v_chi, 0,0],
                        [0, gs[0]*v_phi/2,0,-gs[2]*v_phi/2,0],
                        [0,0,0,gs[2]*v_sigma/2, -gs[3]*v_sigma/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)
    min_Z = np.sum((np.sqrt(Delta2[1:]) - mzs)**2)
    min_gamma = (np.sqrt(np.abs(Delta2[0])))**2
    
    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    min_u, U_uL, diag_yukawa_u, Uh_uR = minimize_for_field(gs, ys_u,"u", vs, M_Us, m_us, mzs, g, g_prim, V, Delta2)
    min_d, Ud_L, diag_yukawa_d, Uh_dR = minimize_for_field(gs, ys_d,"d", vs, M_Ds, m_ds, mzs, g, g_prim, V, Delta2)
    V_CKM_calc = np.dot(np.transpose(U_uL), Ud_L) 

    min_V_CKM =np.sum((conts.V_CKM - np.abs(V_CKM_calc[:3,:3]))**2)
    #print(min_V_CKM)
    return min_V_CKM + min_u + min_d + min_Z + min_gamma

def LCB(mean, var, beta):
    std_dev = np.sqrt(var)
    return -mean + beta * std_dev

def mean_variance_2_G_alpha_beta(mean,variance):
    alpha = mean**2/variance
    beta = variance/mean
    return alpha, beta

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
if __name__ == "__main__":
    m_U, m_D, m_E, v_chi, v_phi, v_sigma = init_constants()
    g = 0.652
    g_prim = 0.357

    m_Z3s = np.linspace(4000,8000,100)
    m_Z3 = 5000
    m_Z3prim = m_Z3*3
    m_Z12 = m_Z3*11

    v_chi = 0.1*m_Z3
    v_phi = 0.1*m_Z3prim
    v_sigma = 0.1*m_Z12

    mzs = np.array([conts.m_Z, m_Z3, m_Z3prim, m_Z12])

    m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
    M_Ds = np.array([m_D, 1.5*m_D, 10*m_D])
    m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
    M_Us = np.array([m_U, 1.5*m_U, 10*m_U])

    vs = np.array([conts.v_H, v_chi, v_phi, v_sigma])
    gs_eq_system(vs[0],vs[1],vs[2],vs[3],mzs[0],mzs[1],mzs[2],mzs[3])


'''

    local_minimization = False
    global_minimization = False
    search_for_gs = False
    gs = np.array([0.1,0.02,1.1,0.35])
    ys = np.ones(13)*0.05
    thetas = np.hstack((gs,ys))
    thetas = np.hstack((thetas,ys))
        
    if local_minimization:
        # MZ > 4 TeV
        # Basis = (W3, B1, B2, B3, B4)^T. Rows: H, chi_f, chi_l, phi, sigma
        #res = minimize(closest_Z_mass, gs, args = (mzs,g, vs), method = "L-BFGS-B", bounds= ((0,2),(0,2),(0,2),(0,2)))

        #gs = res.x
        #print(gs)
        
        #res = minimize(minimize_all, thetas,args=("d",vs,Ms,ms, mzs, g, g_prim),bounds= ((0,2),(0,2),(0,2),(0,2),(0,1000),(0,1000),(0,1000),(0,1000),
        #                                                                               (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)))
        res = minimize(minimize_for_CKM, thetas,args=(vs, g, g_prim, M_Us,m_us, M_Ds, m_ds, mzs),bounds= ((0,2),(0,2),(0,2),(0,2),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                    (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                        (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)))
        gs = res.x[:4]
        ys = res.x[17:]

    elif global_minimization:

        train_points = 100
        train_thetas = np.random.uniform(0,2,(train_points, len(thetas)))
        train_minima = np.zeros((train_points, 1))
        for i in range(train_points):
            res = minimize(minimize_for_CKM, train_thetas[i,:],args=(vs, g, g_prim, M_Us,m_us, M_Ds, m_ds, mzs),bounds= ((0,2),(0,2),(0,2),(0,2),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                    (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),
                                                                                        (0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000),(0,1000)))
            train_thetas[i,:] = res.x
            train_minima[i] = minimize_for_CKM(train_thetas[i],vs, g, g_prim, M_Us, m_us, M_Ds, m_ds, mzs)


        #kernel = GPy.kern.RBF(input_dim=len(thetas))
        #k1['lengthscale'].set_prior(GPy.priors.Gamma(a=l_a, b=l_b))
        #k1['variance'].constrain_bounded(0.2*true_var, 1.8*true_var)
        #For some reason, the prior doesn't work?
        #k1['variance'].set_prior(GPy.priors.InverseGamma(a=v_a, b=1/v_b))
        #k1['lengthscale'].constrain_bounded(0.1, 5)
        #k2 = GPy.kern.Bias(input_dim=len(thetas))
        #kernel = k1 + k2
        #model = GPy.models.GPRegression(train_thetas, train_minima, kernel)
        #beta = 2
        #model.optimize()
        #iter, sampled_points, sampled_minimina = global_min_GP(model, train_thetas, beta,vs, g, g_prim, M_Us, 
        #                                                       m_us, M_Ds, m_ds, mzs, tol = 1e-3, max_iters = 10)
        #iter, theta_points, minimina = global_min_GP(model,X, optimal_x, optimal_y, beta, 300)
    
        #thetas = sampled_points[-1,:]
        thetas = train_thetas[np.argmin([train_minima]),:]
        gs = thetas[:4]
        ys = thetas[17:]

    elif search_for_gs:
        
        v_chis = np.arange(1000,10000,100)
        v_phis = np.arange(1000,10000,100)
        v_sigmas = np.arange(10000,100000,1000)
        m_Z3s = np.arange(1000,10000,100)
        m_Z3_prims = np.arange(1000,10000,100)
        m_Z12s = np.arange(10000,100000,1000)
        X_vchi, X_vphi, X_vsigma, X_z3, X_z3prim, X_z12 = np.meshgrid(v_chis, v_phis, v_sigmas, m_Z3s, m_Z3_prims, m_Z12s)
        # Conditions:
        c1 = 10*X_vchi <= X_vsigma
        c2 = 10*X_vphi <= X_vsigma
        c3 = X_z3/X_vchi < 2
        c4 = X_z3prim/X_vphi < 2
        c5 = X_z12/X_vsigma < 2
        c6 = 10*X_z3 <= X_z12
        c7 = 10*X_z3prim <= X_z12
        print(c1)
        #cond = c1+c2+c3+c4+c5+c6+c7
        
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
                #res = least_squares(closest_Z_mass, gs, args=(mzs, g, vs), x_scale = 'jac',method = 'dogbox',bounds=((0,0,0,0),(10,10,10,10)))
                #print(res)
                #print(infodict)
                #print(mesg)
                #gs = res.x
                #gs = res.x
                #print(f"gs: {gs}")
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
            #f = open("g_models.json", 'w')
            #json.dump(model_list, f)        
            #f.close()
    
    else:
        g_models = np.load("g_models.npz")
        model_list = [g_models[k] for k in g_models]
        print(f"The number of g_models is: {len(model_list)}")
#        gs = model_list[0][2,:]
#        vs = model_list[0][1,:]
#        mzs = model_list[0][0,:]
#        print(gs)
#        print(vs)

    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3, 0,0],
                        [0,gs[0]*vs[1]/2, -gs[1]*vs[1], 0,0],
                        [0, gs[0]*vs[2]/2,0,-gs[2]*vs[2]/2,0],
                        [0,0,0,gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)

    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    for i in range(5):
        print(f"m = {np.sqrt(np.abs(Delta2[i]))}")
        print(f"V_{i} = {V[:,i]}\n")

    print(f"real m_Z = {conts.m_Z}. Found m_Z = {np.sqrt(Delta2[1])}")
    print(f"Diff m_Zs = {mzs-np.sqrt(Delta2[1:])}")
   
    #res = minimize(minimize_yukawas, ys, args = (ms,Ms, vs), method = "BFGS")
    
    #ys = res.x
    #print(ys)

    print(f"ys:{ys}")
    yukawas = build_yukawa(ys, M_Ds, vs)
    
    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)
    print(diag_yukawa)

    print([diag_yukawa[0]-conts.m_d, diag_yukawa[1]-conts.m_s, diag_yukawa[2]-conts.m_b])
    
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
    print(np.abs(np.diag(mass_Q_d_L[:3])) - np.abs(np.diag(sm_Q_d_L)))
    #print(mass_Q_d_R)

    #TODO: Calculate further constants, use in minimization?
    c_sd = (mass_Q_d_L[0,1]**2)/Delta2[1]# + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]
    print(f"The constant for a kaon-antikaon with neutral current is: {c_sd}")
'''
