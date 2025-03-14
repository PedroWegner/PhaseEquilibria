from scipy.optimize import fsolve, newton
import numpy as np
a_ind = []
b_ind = []
b_ij = []
a_ij = []
k_ij = []
a_mix = 0.0
b_mix = 0.0
a_par = []
b_par = []
q_par = []
q = 0.0
beta = 0.0
# !!! Os parametros abaixo depende do modelo, a implementacao natural eh Peng-Robison !!!
_omega = 0.07780
_psi = 0.45724
_eps = 1 - 2**0.5
_sig = 1 + 2**0.5
Z = 0.0
phi = None
ln_phi = None
R = 8.314 # J moçl1 K-1

def calcular_a_b(T, ncomp, Tc, Pc, omega):
    global a_ind, b_ind
    a_ind = []
    b_ind = []
    for i in range(ncomp):
        # !!! alpha depende do modelo adotado, foi implementado com Peng-Robinson!!!
        alpha = (1 + (0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i] ** 2) * (1 - (T / Tc[i])**0.5)) ** 2
        a = (_psi * (R * Tc[i]) ** 2 * alpha) / Pc[i]
        b = _omega * R * Tc[i] / Pc[i]
        a_ind.append(a)
        b_ind.append(b)

def calcular_parametros_cruzados(ncomp):
    global a_ij, b_ij
    a_ij = []
    b_ij = []
    for i in range(ncomp):
        a_ij.append([])
        b_ij.append([])
        for j in range(ncomp):
            a_ij[i].append((a_ind[i] * a_ind[j])**0.5) # * (1 - k_ij[i][j])
            if i != j:
                b_ij[i].append(0.0)
            else:
                b_ij[i].append(b_ind[i])

def calcular_mistura(x: list, ncomp):
    global a_mix, b_mix
    a_mix = 0.0
    b_mix = 0.0
    for i in range(ncomp):
        for j in range(ncomp):
            a_mix += x[i] * x[j] * a_ij[i][j]
            b_mix +=  x[i] * b_ij[i][j]

def calcular_param_parcial(x: list, ncomp):
    global a_par, b_par, q_par
    a_par = []
    b_par = []
    q_par = []
    for i in range(ncomp):
        a_aux = - a_mix
        for j in range(ncomp):
            a_aux += 2 * x[j] * a_ij[i][j]
        b_aux = b_ind[i]
        a_par.append(a_aux)
        b_par.append(b_aux)
        q_aux = q * (1 + a_par[i] / a_mix - b_par[i] / b_mix)
        q_par.append(q_aux)

def f_z_vapor(_z):
    return _z - (1 + beta - beta * q * (_z - beta) / ((_z + _eps * beta) * (_z + _sig * beta)))

def df_z_vapor(_z):
    aux = ((_z + _eps*beta) * (_z + _sig*beta))
    return 1.0 + q*beta*(aux - (_z - beta)*(2*_z + beta * (_eps + _sig))) / aux**2

def f_z_liquid(_z):
    return _z - (beta + (_z + _eps * beta) * (_z + _sig * beta) * ((1 + beta - _z) / (q * beta)))

def df_z_liquid(_z):
    return 1 - ((2*_z + beta * (_eps + _sig)) * (1 + beta - _z) - (2*_z + beta * (_eps + _sig))) / (q * beta)

def solve_z(T: float, P: float, ncomp: int, x: list, Tc: list, Pc: list, omega: list, vapor: bool):
    global beta, q, Z
    calcular_a_b(T=T, ncomp=ncomp, Tc=Tc, Pc=Pc, omega=omega)
    calcular_parametros_cruzados(ncomp=ncomp)
    calcular_mistura(x=x, ncomp=ncomp)

    beta = b_mix * P / (R * T)
    q = a_mix / (b_mix * R * T)

    if vapor:
        Z = (fsolve(f_z_vapor, x0=[1.0]))[0]
        # Z = newton(func=f_z_vapor, x0=1.0, fprime=df_z_vapor, maxiter=150)
    else:
        print('sou liq')
        Z = (fsolve(f_z_liquid, x0=[beta]))[0]
        # Z = newton(func=f_z_liquid, x0=beta, fprime=df_z_liquid, maxiter=150)
    print(Z, beta, q)

def solve_fugacity(T: float, P: float, ncomp: int, x: list, Tc: list, Pc: list, omega: list, vapor: bool):
    global phi, ln_phi
    solve_z(T=T, P=P, ncomp=ncomp, x=x, Tc=Tc, Pc=Pc, omega=omega, vapor=vapor)
    calcular_param_parcial(x=x, ncomp=ncomp)
    I = float(np.log((Z + _sig * beta) / (Z + _eps * beta)) / (_sig - _eps))
    print(I)
    for i in range(ncomp):
        print(b_par[i], (Z - beta))
    ln_phi = [(b_par[i] / b_mix) * (Z - 1) - np.log(Z - beta) - q_par[i] * I for i in range(ncomp)]
    phi = [np.exp(ln_phi[i]) for i in range(ncomp)]
    print(Z, phi, ln_phi, vapor)
    return phi
    pass
