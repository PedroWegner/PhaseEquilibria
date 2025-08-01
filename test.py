import numpy as np
from CubicEoS_module import *
from critical_point import *
import matplotlib.pyplot as plt
from time import time

metano = Component(name='Methane', Tc=190.6, Pc=45.99e5, omega=0.012)    
dioxide = Component(name='Carbon Dioxide', Tc=304.2, Pc=73.83e5, omega=0.224)
sulfeto = Component(name='sulfeto de hidrogenio', Tc=373.5, Pc=89.63e5, omega=0.094)
ethane = Component(name='Ethane', Tc=305.3, Pc=48.72e5, omega=0.100)
methanol = Component(name='Methanol', Tc=512.6, Pc=80.97e5, omega=0.564) 
# Tc_space = [190.6-273.15, 304.2-273.15]
# Pc_space = [45.99, 73.83]  
Tc_space = [190.6-273.15, 373.5-273.15]
Pc_space = [45.99, 89.63]  



tester_ = tester(EoS_Engine=ModeloPengRobinson)
T_metano_space = np.linspace(140, 190, 50)
P_metano_space = []
P = 5e5
# for T in T_metano_space:
#     P = tester_.calcula(component=metano, T=T, P=P)
#     print(P)
#     P_metano_space.append(P / 10**5)

# T_dioxide_space = np.linspace(220, 303.5, 50)
# P_dioxide_space = []
# P = 5e5
# for T in T_dioxide_space:
#     P = tester_.calcula(component=dioxide, T=T, P=P)
#     print(P)
#     P_dioxide_space.append(P / 10**5)

k_ij = 0.08
k_ij = np.array([[0, k_ij],[k_ij,0]])
# l_ij = 0.20
# l_ij = np.array([[0, l_ij],[l_ij,0]])

mixture = Mixture([metano, dioxide], k_ij=k_ij, l_ij=0.0)
z = np.array([0.001, 0.999])
critical_point_solver = CriticalPointSolver(EoS_Engine=ModeloPengRobinson)
z1_space= np.linspace(0.0001, 0.9999, 150)
trial_state = State(mixture=mixture, T=280, P=25e5, z=z, is_vapor=True)
T_space = []
P_space = []

T, Vm = critical_point_solver.initial_guess(state=trial_state)
print(T, Vm)
# T, P, Vm = critical_point_solver.calcula(state=trial_state, T=T, Vm=Vm)
t01 = time()
# for z1 in z1_space:
#     z = np.array([z1, 1 - z1])
    
#     trial_state.z = z
#     T, P, Vm = critical_point_solver.solver_optimize(state=trial_state, T=T, Vm=Vm)
#     T_space.append(T - 273.15)
#     P_space.append(P / 10**5)
    # print(z, T, P, Vm)
tf1 = time()



t02 = time()

z_alvo = 0.001
T_alvo = 304.2
Vm_alvo = 9.8e-5

z_chute = z_alvo * 1.05
T_chute = T_alvo * 1.1
Vm_chute = Vm_alvo * 0.98
trial_state.T = T_chute
trial_state.Vm = Vm_chute
trial_state.z = np.array([z_chute, 1 - z_chute])
PTVm = critical_point_solver.calcula_2(state=trial_state)
tf2 = time()

x = []
y =[]
for i in PTVm:
    x.append(i[0])
    y.append(i[1])



T_metano_space -= 273.15
# T_dioxide_space -= 273.15
plt.plot(T_space, P_space, linestyle='--', linewidth=1.25, color='red', label="V-L")
plt.plot(x, y, linestyle='--', linewidth=1.25, color='blue', label="V-L")
# plt.plot(T_metano_space, P_metano_space, color='k', linewidth=1.0)
# plt.plot(T_dioxide_space, P_dioxide_space, color='k', linewidth=1.0)
plt.scatter(Tc_space, Pc_space, marker='x', color='k')
plt.xlabel('T / ºC')
plt.ylabel('P / bar')
print(tf1 - t01)
print(tf2 - t02)
plt.show()
