import flavorstuff as fs
import numpy as np

from scipy.optimize import minimize

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
    
def closest_Z_mass(gs, g, g_prim, v_chi, v_phi,v_sigma):
    conts = fs.constants()
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), gs[0]*conts.v_H/(2), 0, 0 , 0],
                        [0, -gs[0]*v_chi/6, gs[1]*v_chi/3,              0,                  0],
                        [0, gs[0]*v_chi/2, -gs[1]*v_chi,                0,                  0],
                        [0, gs[0]*v_phi/2,              0,-gs[2]*v_phi/2,                   0],
                        [0, 0,                          0, gs[2]*v_sigma/2, -gs[3]*v_sigma/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b) 
    return np.abs(np.sqrt(Delta2[1]) - conts.m_Z)

def build_yukawa(ys, y_3, v, v_chi, v_phi, v_sigma, m_U):
    #ys = np.ones(13)

    m = np.zeros((3,3))
    m[2,2] = v*y_3

    Delta_L = v*np.array([[ys[0], ys[1], 0],[ys[2], ys[3],0], [0,0,0]])
    Delta_R = np.array([[0, ys[4]*v_phi, 0], [0, ys[5]*v_phi, ys[6]*v_chi], [ys[7]*v_sigma,0,0]])
    M = np.array([[m_U, 0, ys[8]*v_phi], [0, m_U,ys[9]*v_phi], [ys[10]*v_phi, ys[11]*v_phi, m_U]])

    yukawas = np.vstack((np.hstack((m, Delta_L)), np.hstack((Delta_R, M))))

    return yukawas

def minimize_yukawas_d(ys, v_chi, v_phi, v_sigma, m_D):
    conts = fs.constants()
   
    yukawas = build_yukawa(ys, conts.m_b/conts.v_H, conts.v_H, v_chi, v_phi, v_sigma, m_D)

    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)

    return np.abs(diag_yukawa[3]-conts.m_b + diag_yukawa[4]-conts.m_s + diag_yukawa[5]-conts.m_d)

def build_Q(field_type, base):
    base_charge = [fs.find_charge("fermions", field_type, "I3"), fs.find_charge("fermions", field_type, "Y"), 
                   (fs.find_charge("fermions", field_type, "B") - fs.find_charge("fermions", field_type, "L"))/2,
                   fs.find_charge("fermions", field_type, "I3_R"), fs.find_charge("fermions", field_type, "I3_R")]
    charges = base*base_charge
 
    # First three regular fermoins, last three new vector-like fermions. 
    Q = charges[0] + np.diag([charges[2] + charges[3], charges[2] + charges[4], charges[1], 
                                charges[2] + charges[3], charges[2] + charges[4], charges[1]])
    return Q

if __name__ == "__main__":
    m_U, m_D, m_E, v_chi, v_phi, v_sigma = init_constants()
    g = 0.652
    g_prim = 0.357
    g1 = 0.5
    g2 = 0.5
    g3 = 0.5
    g4 = 0.5

    conts = fs.constants()
    # Basis = (W3, B1, B2, B3, B4)^T. Rows: H, chi_f, chi_l, phi, sigma
    minim = True
    res = minimize(closest_Z_mass, [g1,g2,g3,g4], args = (g,g_prim, v_chi, v_phi, v_sigma), method = "BFGS")#, bounds= ((0.1,2), (0.1,2), (0.1,2), (0.1,2)))
    print(res.x)

    g1 = res.x[0]
    g2 = res.x[1]
    g3 = res.x[2]
    g4 = res.x[3]
  
    M_b = -1j*1/np.sqrt(2)*np.array([[-g*conts.v_H/(2), g1*conts.v_H/(2), 0, 0 , 0],
                        [0, -g1*v_chi/6, g2*v_chi/3, 0,0],
                        [0,g1*v_chi/2, -g2*v_chi, 0,0],
                        [0, g1*v_phi/2,0,-g3*v_phi/2,0],
                        [0,0,0,g3*v_sigma/2, -g4*v_sigma/2]])
    
    Delta2, V = fs.gauge_boson_basis(M_b)
    V[np.abs(V) < 0.01] = 0 
    #print(f"Delta_2 = {Delta2}")
    #print(f"V = {V}")
    for i in range(5):
        print(f"m = {np.sqrt(Delta2[i])}")
        print(f"V_{i} = {V[:,i]}\n")

    print(f"real m_Z = {conts.m_Z}. Found m_Z = {np.sqrt(Delta2[1])}")

    ys = np.ones(12)*0.5
    
    res = minimize(minimize_yukawas_d, ys, args = (v_chi, v_phi, v_sigma, m_D), method = "BFGS")#, bounds= ((0.1,2), (0.1,2), (0.1,2), (0.1,2)))
    ys = res.x
    yukawas = build_yukawa(ys, conts.m_b/conts.v_H, conts.v_H, v_chi, v_phi, v_sigma, m_D)
    
    U_L, diag_yukawa, Uh_R = fs.diag_yukawa(yukawas)

    #print(diag_yukawa)
    #print([diag_yukawa[3]-conts.m_b, diag_yukawa[4]-conts.m_s, diag_yukawa[5]-conts.m_d])
    
    # Try Z^3 for down quarks, check FCNC
    base = V[:,2]
    
    Q_d_L = build_Q("d_L", base)
    Q_d_R = build_Q("d_R", base)

    mass_Q_d_L = fs.mass_Q(U_L, Q_d_L)
    mass_Q_d_R = fs.mass_Q(np.transpose(Uh_R), Q_d_R)

    #print(mass_Q_d_L)
    #print(mass_Q_d_R)

    c_sd = mass_Q_d_L[0,1]**2 + mass_Q_d_R[0,1]**2 + 2*mass_Q_d_L[0,1]*mass_Q_d_R[0,1]
    print(f"The constant for a kaon-antikaon with neutral current is: {c_sd}")
   
