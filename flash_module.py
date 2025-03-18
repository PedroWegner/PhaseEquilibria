from Peng_robinson_module import *

Tc = [190.6, 425.1]
Pc = [45.99e5, 37.96e5]
omega = [0.012, 0.200]

T_op = 310.93 #K
P_op = 113.5555e5 # Pa
z = [0.5594, 1 - 0.5594]

# *** ESTIMATIVA INICIAIS ~meio porco~
x = 0.01
y = 0.99


liquid = Peng_robinson_state(T=T_op,
                            P=P_op,
                            ncomp=2,
                            x=[x, 1 - x],
                            Tc=Tc,
                            Pc=Pc,
                            omega=omega,
                            liquid=True)

gas = Peng_robinson_state(T=T_op,
                            P=P_op,
                            ncomp=2,
                            x=[y, 1 - y],
                            Tc=Tc,
                            Pc=Pc,
                            omega=omega,
                            liquid=False)


m = [1.000, 2.3316]
sigma = [3.7039, 3.7086]
ep = [150.03, 222.88]
# liquid = PC_saft_state(T=T_op,
#                        P=P_op,
#                        ncomp=2,
#                        x=[x, 1 - x],
#                        m=m,
#                        sigma=sigma,
#                        epsilon=ep,
#                        liquid=True)

# gas = PC_saft_state(T=T_op,
#                        P=P_op,
#                        ncomp=2,
#                        x=[y, 1 - y],
#                        m=m,
#                        sigma=sigma,
#                        epsilon=ep,
#                        liquid=True)


update_state(state=liquid)
update_state(state=gas)
print(liquid.Z, 'and: ', liquid.phi)
print(gas.Z, 'and: ', gas.phi)
K = liquid.phi / gas.phi
# *** ESTIMATIVA DA FRAÇÃO DE VAPOR
beta = 0.5
for i in range(250):
    print(i)
    for j in range(250):
        F = np.sum(z * (K - 1) / (1 + beta * (K - 1)))
        primeF = - np.sum(z * (K - 1)**2 / (1 + beta * (K - 1))**2)
        beta_o = beta
        beta = beta - F / primeF

        if np.abs(beta_o - beta) < 1e-9:
            break

    liquid.x = z / (1 + beta * (K - 1))
    gas.x = K * liquid.x

    update_state(state=liquid)
    update_state(state=gas)
    K = liquid.phi / gas.phi
    err = (np.sum((liquid.phi * liquid.x)/ (gas.phi * gas.x) - 1))**2
    if err < 1e-8:
        break

print(liquid.x)
print(gas.x)
print(err)
print(beta)
x = 1 / (1 - K)
print(x)