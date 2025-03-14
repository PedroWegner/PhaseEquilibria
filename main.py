import EoS
from EoS import *
from PhaseEquilibria import *
# ncomp = 2
# Tc = [190.6, 304.2]
# Pc = [45.99*10**5, 73.83*10**5]
# omega = [0.012, 0.224]
# k_ij = [
#     [0.0, 1.0],
#     [1.0, 0.0]
# ]


ncomp = 2
name = ['CH2', 'C4H10']
Tc = [190.6, 425.1]
Pc = [45.99*10**5, 37.96*10**5]
omega = [0.012, 0.200]
T=310.93 # K
P = 16517526.827650828
x = [0.8926174496644295, 0.10738255033557054]
# solve_fugacity(T=T, P=P, ncomp=ncomp, x=x, Tc=Tc, Pc=Pc, omega=omega, vapor=True)
# solve_fugacity(T=T, P=P, ncomp=ncomp, x=x, Tc=Tc, Pc=Pc, omega=omega, vapor=False)


P = 3*10**5
bubble_p(T=T, P_est=P, ncomp=ncomp, Tc=Tc, Pc=Pc, omega=omega, h=1500, max_iter=250)
graph_generator()

# ABAIXO EH EXEMPLO 3.9 DO LIVRO VAN NESS
# ncomp = 2
# name = ['N2', 'CH4']
# Tc = [126.2, 190.6]
# Pc = [34.00*10**5, 45.99*10**5]
# omega = [0.038, 0.012]
# T=200.00 # K
# P = 30*10**5
# x = [0.4, 0.6]
# solve_fugacity(T=T, P=P, ncomp=ncomp, x=x, Tc=Tc, Pc=Pc, omega=omega, vapor=True)
