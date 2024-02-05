import numpy as np
import json
f = open("charges.json")
charges = json.load(f)
f.close()

class constants:
    def __init__(self, unit = "GeV") -> None:
        valid_units = {"GeV":1, "TeV":0.001}
        try:
            s = valid_units[unit]
        except KeyError as e:
            print(f"Invalid key {unit}")
            exit(1)
        
        self.v_H = s*246
        self.m_u = s*2.3*10**(-3)
        self.m_d = s*4.8*10**(-3)
        self.m_c = s*1.275
        self.m_s = s*0.095
        self.m_t = s*173.21
        self.m_b = s*4.18

        self.m_Z = s*91.1876
        self.m_W = s*80.377

def gauge_boson_basis(M):
    MT = M.conj().T
    M2 = np.dot(MT,M)
    (Delta2, V) = np.linalg.eigh(np.real(M2))
    return Delta2, V

def diag_yukawa(Y):
    (U,Delta,Vh) = np.linalg.svd(Y)
    return U,Delta,Vh


def mass_Q(V, Q):
    return np.matmul(np.matmul(V.conj().T,Q),V)

def find_charge(type_, field, charge):
    charge_str = charges[type_][field][charge]
    if "/" in charge_str:
        [a,b] = charge_str.split("/")
        ans = int(a)/int(b)
    else:
        ans = int(charge_str)
    return ans

    