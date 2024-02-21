#####################################################################################
# Module for model SU(3) x SU(2) x U^3_Y(1) x U^12_B-L(1) x U^2_I3R(1) x U^1_I3R(1) #
#####################################################################################
import flavorstuff as fs
import numpy as np
import GPy

from scipy.optimize import minimize
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
def closest_Z_mass(gs, mzs,g, vs):
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*vs[0]/(2), gs[0]*vs[0]/(2), 0, 0 , 0],
                        [0, -gs[0]*vs[1]/6, gs[1]*vs[1]/3,              0,                  0],
                        [0, gs[0]*vs[1]/2, -gs[1]*vs[1],                0,                  0],
                        [0, gs[0]*vs[2]/2,              0,-gs[2]*vs[2]/2,                   0],
                        [0, 0,                          0, gs[2]*vs[3]/2, -gs[3]*vs[3]/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b) 
    return (np.sqrt(Delta2[1]) - mzs[0])**2 #+ np.sum(np.sqrt(Delta2[2:]) - mzs[1:])**2 + Delta2[0] #np.abs(np.sqrt(Delta2[1]) - conts.m_Z)

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
    
    min_gamma = (np.sqrt(np.abs(Delta2[0])))**2
    #min_Z = (np.sqrt(Delta2[1]) - mzs[0])**2
    min_Y = np.sum((diag_yukawa[:3]- ms)**2) + np.sum((diag_yukawa[3:]-Ms)**2)
    #print(f"min_Z = {min_Z}, min_Y = {min_Y}, min_Q = {min_Q}")
    return min_Y + min_QZ + min_Qgamma + min_gamma, U_L, diag_yukawa, Uh_R
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
    
    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    min_u, U_uL, diag_yukawa_u, Uh_uR = minimize_for_field(gs, ys_u,"u", vs, M_Us, m_us, mzs, g, g_prim, V, Delta2)
    min_d, Ud_L, diag_yukawa_d, Uh_dR = minimize_for_field(gs, ys_d,"d", vs, M_Ds, m_ds, mzs, g, g_prim, V, Delta2)
    V_CKM_calc = np.dot(np.transpose(U_uL), Ud_L) 

    min_V_CKM =np.sum((conts.V_CKM - np.abs(V_CKM_calc[:3,:3]))**2)
    #print(min_V_CKM)
    return min_V_CKM + min_u + min_d + min_Z

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

    local_minimization = False
    gs = np.array([0.1,0.02,1.1,0.35])
    ys = np.ones(13)*0.05
    thetas = np.hstack((gs,ys))
    thetas = np.hstack((thetas,ys))
        
    if local_minimization:
        # MZ > 4 TeV
        m_Z3 = 4000
        m_Z3prim = 2*m_Z3
        m_Z12 = m_Z3*10
        
        v_chi = 0.1*m_Z3
        v_phi = 0.1*m_Z3prim
        v_sigma = 0.1*m_Z12

        mzs = np.array([conts.m_Z, m_Z3, m_Z3prim, m_Z12])
        
        m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
        M_Ds = np.array([m_D, 1.5*m_D, 10*m_D])
        m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
        M_Us = np.array([m_U, 1.5*m_U, 10*m_U])
        
        vs = np.array([conts.v_H/4, v_chi, v_phi, v_sigma])

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

    else:
        m_Z3 = 4000
        m_Z3prim = 2*m_Z3
        m_Z12 = m_Z3*10

        v_chi = 0.1*m_Z3
        v_phi = 0.1*m_Z3prim
        v_sigma = 0.1*m_Z12
        mzs = np.array([conts.m_Z, m_Z3, m_Z3prim, m_Z12])
        
        m_ds = np.array([conts.m_d, conts.m_s, conts.m_b])
        M_Ds = np.array([m_D, 1.5*m_D, 10*m_D])
        m_us = np.array([conts.m_u, conts.m_c, conts.m_t])
        M_Us = np.array([m_U, 1.5*m_U, 10*m_U])
        
        vs = np.array([conts.v_H/4, v_chi, v_phi, v_sigma])

        #column = np.linspace(0,2,3)#.reshape(-1,1)
        #in_grid = np.hstack([column]*len(thetas))
        #in_grid = (column,)*len(thetas)
        #print(*in_grid)
        #print(len(in_grid))
        #grid_vectors = np.meshgrid(*in_grid)
        #print(len(grid_vectors))
        #print(grid_vectors)
        #grid_vectors = [vector.ravel() for vector in grid_vectors]
        #print(len(grid_vectors))
        #grid = np.hstack(grid_vectors)

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

    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*v_chi/6, gs[1]*v_chi/3, 0,0],
                        [0,gs[0]*v_chi/2, -gs[1]*v_chi, 0,0],
                        [0, gs[0]*v_phi/2,0,-gs[2]*v_phi/2,0],
                        [0,0,0,gs[2]*v_sigma/2, -gs[3]*v_sigma/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)

    # Cuts off non-significant contributions, for a cleaner model
    V[np.abs(V) < 0.01] = 0 

    for i in range(5):
        print(f"m = {np.sqrt(Delta2[i])}")
        print(f"V_{i} = {V[:,i]}\n")

    print(f"real m_Z = {conts.m_Z}. Found m_Z = {np.sqrt(Delta2[1])}")

   
    #res = minimize(minimize_yukawas, ys, args = (ms,Ms, vs), method = "BFGS")
    
    #ys = res.x
    #print(ys)
    print(f"ys:{ys}")
    yukawas = build_yukawa(ys, np.array([m_D, m_D*1.5, m_D*2]), vs)
    
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
   
