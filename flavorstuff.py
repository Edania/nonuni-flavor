########################################################
# Module for general flavor non-universal calculations #
########################################################

import numpy as np
import json
import emcee as mc
import corner
import matplotlib.pyplot as plt

from scipy.stats import invgamma


f = open("charges.json")
charges = json.load(f)
f.close()

#Stores "fundamental" constants, eg all SM masses and Higgs vev
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

        S = s*1000
        self.Lambda_sd_2 = S*980
        self.Lambda_sd_LR = S*18000
        self.Lambda_cu_2 = S*1200
        self.Lambda_cu_LR = S*6200
        self.Lambda_bd_2 = S*510
        self.Lambda_bd_LR = S*1900
        self.Lambda_bs_2 = S*110
        self.Lambda_bs_LR = S*370

        self.V_CKM = [[0.97373, 0.2243, 0.00382],
                 [0.221, 0.975, 0.0408],
                 [0.0086, 0.0415, 1.014]]

# Method for finding the gauge boson basis given an M
def gauge_boson_basis(M):
    MT = M.conj().T
    M2 = np.dot(MT,M)
    (Delta2, V) = np.linalg.eigh(np.real(M2))
    return Delta2, V

#This function is completely unneccessary. Live and let learn
def find_yukawas(init_ys, init_sigma, a, ms, vs, build_yukawa, ms_build, nsamples = 1000, file = None):
    ms = np.array(ms)
    ms = -np.sort(-ms)
    vs = np.array(vs)

    def log_sigma_prior(thetas):
        sigma2 = thetas[-1]
        if sigma2 <= 0:
            return -np.inf
        if sigma2 > 2:
            return -np.inf

        return invgamma.logpdf(sigma2, a=a, scale=init_sigma*(a+1))

    def log_ys_prior(thetas):
        ys = thetas[:-1]
        if any(ys < -2) or any(ys > 2):
            prob = -np.inf
        else:
           prob = 0
        return prob

    def log_likelihood(thetas):
        ys = thetas[:-1]
        sigma2 = thetas[-1]
        Y = build_yukawa(ys, ms_build, vs)
        U, Y_mass, Vh = diag_yukawa(Y)
        Y_diag = np.abs(Y_mass)
        prob = -0.5*np.sum((Y_diag[:] - ms[:])**2)/sigma2 - len(ms[:])/2 * (sigma2)
        return prob

    def log_posterior(thetas):
        llh = log_likelihood(thetas)
        lsp = log_sigma_prior(thetas)
        lyp = log_ys_prior(thetas)
        
        if np.isneginf(llh) or np.isneginf(lsp) or np.isneginf(lyp):
            return -np.inf

        return llh + lsp + lyp

    ndim = len(init_ys) + 1
    nwalkers = ndim*2
    init_thetas = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        init_thetas[i,:-1] = init_ys + 0.05*(np.random.rand(len(init_ys))-0.5)
        init_thetas[i,-1] = invgamma.rvs(a=a, scale=init_sigma*(a+1))
    
    if file:
        backend = mc.backends.HDFBackend(file)
        sampler = mc.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend)
    else:
        sampler = mc.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(init_thetas, nsamples, progress = True)

    largest_autocorr = 1
    samples = sampler.get_chain(discard = int(nsamples/4),  thin = largest_autocorr, flat = True)
    fig, axs = plt.subplots(1,2, figsize = (10,7))
    for i in range(13):
        axs[0].plot(samples[:,i])
    axs[1].plot(samples[:,-1])
    plt.savefig("test_mc.png")
    return samples

# Method for Diagonalizing Yukawa matrices
def diag_yukawa(Y):
    (U,Delta_tmp,Vh) = np.linalg.svd(Y)
    U = U[:,::-1]
    Delta = Delta_tmp[::-1]
    Vh = Vh[::-1,:]
    return U,Delta,Vh

# Method for finding the mass basis for Q given left- or right-rotating unitary matrix
def mass_Q(V, Q):
    return np.matmul(np.matmul(V.conj().T,Q),V)

# Finds specified charge from charges.json
def find_charge(type_, field, charge):
    charge_str = charges[type_][field][charge]
    if "/" in charge_str:
        [a,b] = charge_str.split("/")
        ans = int(a)/int(b)
    else:
        ans = int(charge_str)
    return ans

def Z_coupling(g,g_prim):
    pass

    