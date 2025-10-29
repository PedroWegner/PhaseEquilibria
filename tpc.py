import openpyxl as pyxl
import matplotlib.pyplot as plt
import os


def gera_lista(name: str) -> list[list]:
        """
        Funcao criada para facilitar retirar os dados dos excels
        name: nome do arquivo que deve ser xlsx
        binary: boolean para analisar como deve-se remover os dados da planilha, eh true se nenhum argumento for passado

        Return:
        list_: uma lista [[temperatura], [pressao], [x_experimental], [y_experimental]]
        """
        wb = pyxl.load_workbook((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\{name}.xlsx'))
        sheet = wb.active
        list_ = [[], []]
        for l in filter(None, sheet.iter_rows(min_row=2, values_only=True)):
            list_[0].append(l[0])
            list_[1].append(l[1])

        return list_



file_ch4 = 'CH4_saturation_points'
file_co2 = 'CO2_saturation_points'
file_h2s = 'H2S_saturation_points'
# file_TP = 'CH4,CO2_critical_points_k_ij=0.093'
file_TP = 'CO2,H2S_critical_points_k_ij=0.099'
files = [file_co2, file_h2s, file_TP]
dict_data = {}

for file in files:
    if not file in dict_data:
         dict_data[file] = None
    list_ = gera_lista(name=file)
    dict_data[file] = list_

# Tc_space = [190.6-273.15, 304.2-273.15]
# Pc_space = [45.99, 73.83]  

Tc_space = [304.2-273.15, 373.5-273.15]
Pc_space = [73.83, 89.63] 


# Experimentais do [CH4 - CO2]
# Tc_exp = [31.1111111111111 ,13.3333333333333 ,0.555555555555556 ,-16.6666666666667 ,-51.1111111111111 ,-82.2222222222222]
# Pc_exp = [73.9807748048285 ,83.7713340054675 ,86.184500005625 ,84.4608100055125 ,67.9133860044325 ,46.4017348030285]

# Experimentais do [CO2-H2S]
Tc_exp = [93.5 ,84.16 ,74.48 ,64.74 ,56.98 ,43.72 ,35.96 ,33.53]
Pc_exp = [89.9664675 ,89.77395 ,88.5073875 ,85.862805 ,83.20809 ,77.8479975 ,74.8285125 ,74.1597675]


plt.figure(figsize=(5, 6))
# file_name = file_1
# plt.plot(dict_data[file_ch4][0], dict_data[file_ch4][1], color='black', linewidth=1.25)
plt.plot(dict_data[file_co2][0], dict_data[file_co2][1], color='black', linewidth=1.25)
plt.plot(dict_data[file_h2s][0], dict_data[file_h2s][1], color='black', linewidth=1.25)
plt.plot(dict_data[file_TP][0], dict_data[file_TP][1], linestyle='--', linewidth=1.25, color='red', label="V-L")
# plt.scatter(Tc_space, Pc_space, marker='+', color='k')
plt.scatter(Tc_exp, Pc_exp, marker='x', color='k')

# plt.scatter(x_exp_1, P_exp_1, marker='x', color='black')
# plt.scatter(y_exp_1, P_exp_1, marker='x', color='black')


plt.ylabel(ylabel=r'$P\;/\;bar$')
plt.xlabel(xlabel=r'$T\;/\; ºC$')
# plt.xlim(left=0.0, right=max(dict_data[file_name][3])*1.10)
# plt.ylim(bottom=dict_data[file_name][1][0]*0.8)
# plt.legend()
file_name = 'CH4,H2S_critical_points'
plt.savefig((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\img\\{file_name}.png'), dpi=600, bbox_inches='tight')
plt.show()