import numpy as np

T = 350 # K
P = 9.4573e5 # Pa

R = 8.314
Mw = 58.123
Tc = 425.1
Pc = 37.96e5
omega = 0.200
m = 2.5699
x = np.array([1.0])
m_aver = np.sum(x * m)


# As eq. 20 - 22

m0 = (1 - m) / m
c0 = - 128000 * (1 + m0)
c1 = 566400 * (1 + m0)
c2 = -249840 - 748800 * m0
c3 = -145562 + 188800 * m0
c4 = 36366 + 182400 * m0
c5 = 45486
c6 = -13718 - 60800 * m0


coeffs = [c6, c5, c4, c3, c2, c1, c0]
print(coeffs)
roots = np.roots(coeffs)
beta_c = None
for root in roots:
    if np.isreal(root) and 0 < root.real < 1:
        beta_c = root.real
        break

# lambdas
lambda_chain = 840 * beta_c / (40 - 19 * beta_c)**2
lambda_mon = - beta_c * (beta_c**2 + 2 * beta_c - 1) / (1 + beta_c)**2

# os Z
termo_aux = (1 + beta_c / (lambda_mon * (1 + beta_c)))**(-1)

Zc_chain = (beta_c / (1 + beta_c)) * (lambda_chain / lambda_mon + 40 / (40 - 19 * beta_c)) * termo_aux
Zc_mon = (1 / (1 - beta_c)) * termo_aux

Zc =  Zc_mon - (m - 1) * Zc_chain
print(Zc)

psi_a = (1 / m**2) * (beta_c * Zc**2 / lambda_mon + (m -1) * beta_c * Zc * lambda_chain / lambda_mon)
psi_b = (1 / m) * beta_c * Zc

m = 0.37464 + 1.54266 * omega + 0.26992 * omega**2
Tr = T / Tc
alpha = (1 + m * (1 - Tr**0.5))**2
a = psi_a * (R * Tc)**2 * alpha / Pc
b = psi_b * R * Tc / Pc
