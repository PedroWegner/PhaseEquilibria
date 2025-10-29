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
        list_ = [[], [], [], []]
        for l in filter(None, sheet.iter_rows(min_row=2, values_only=True)):
            list_[0].append(l[0])
            list_[1].append(l[1])
            list_[2].append(l[2])
            list_[3].append(l[3])

        return list_


# file_1 = 'CH4,CO2_T=199.81_k_ij=0.093'
# file_2 = 'CH4,CO2_T=223.71_k_ij=0.093'
# file_3 = 'CH4,CO2_T=241.48_k_ij=0.093'
# file_4 = 'CH4,CO2_T=259.82_k_ij=0.093'
# file_5 = 'CH4,CO2_T=271.48_k_ij=0.093'
file_1 = 'CH4,H2S_T=277.6_k_ij=0.08'
file_2 = 'CH4,H2S_T=344.3_k_ij=0.08'
files = [file_1, file_2]
dict_data = {}



for file in files:
    if not file in dict_data:
         dict_data[file] = None
    list_ = gera_lista(name=file)
    dict_data[file] = list_
    


# PONTOS EXPERIMENTAIS [CH4, H2S]
# Pontos experimentais, Reamer, Sagen e Lacey, 1951, 277.6 [CH4, H2S]
# P_exp_1 = [13.78952, 17.2369, 20.68428, 24.13166, 27.57904, 31.02642, 34.4738, 41.36856, 48.26332, 55.15808, 62.05284, 
#             68.9476, 75.84236, 82.73712001, 86.18450001, 89.63188001, 96.52664001, 103.4214, 110.31616, 117.21092, 120.6583, 124.10568, 
#             131.00044, 134.3788724]
# y_exp_1 = [0.1371, 0.2783, 0.3896, 0.4604, 0.5126, 0.5551, 0.5879, 0.6394, 0.6755, 0.6989, 0.7141, 0.7242, 0.7299, 0.7321,
#             0.7319, 0.7306, 0.7262, 0.7185, 0.7075, 0.6931, 0.6828, 0.6686, 0.613, 0.55 ]
# x_exp_1 = [0.0057, 0.0132, 0.0212, 0.0284, 0.0354, 0.0424, 0.0493, 0.0636, 0.0783, 0.093, 0.1083, 0.125, 0.1433, 0.1635, 
#             0.1752, 0.1868, 0.2137, 0.245, 0.2798, 0.324, 0.3492, 0.3758, 0.4401, 0.55, ]
# # Pontos experimentais, Reamer, Sagen e Lacey, 1951, 310.9 [CH4, H2S]
# P_exp_2 = [27.57904, 31.02642, 34.4738, 37.92118, 41.36856, 48.26332, 55.15808, 62.05284, 68.9476, 75.84236,
#            82.73712001, 86.18450001, 89.63188001, 96.52664001, 103.4214, 110.31616, 117.21092, 120.6583, 124.10568,
#            127.55306, 131.00044, 131.4830732]
# y_exp_2 = [0.0117, 0.0963, 0.1642, 0.2203, 0.2688, 0.3416, 0.3976, 0.4396, 0.4707, 0.4923, 0.5079, 0.513, 0.5182,
#            0.524, 0.5255, 0.5195, 0.5058, 0.4947, 0.4797, 0.458, 0.419, 0.388]
# x_exp_2 = [0.0007, 0.0067, 0.0128, 0.019, 0.0255, 0.0385, 0.0523, 0.067, 0.0828, 0.0996, 0.1182, 0.1282, 0.139, 0.162,
#            0.1885, 0.2192, 0.2532, 0.2725, 0.294, 0.3185, 0.3578, 0.388]
# # Pontos experimentais, Reamer, Sagen e Lacey, 1951, 344.3 [CH4, H2S]
P_exp_3 = [55.15808, 58.60546, 62.05284, 68.9476, 75.84236, 82.73712001, 86.18450001, 89.63188001, 96.52664001, 
         103.4214, 110.31616, 113.76354, 114.453016]
x_exp_3 = [0.0031, 0.0098, 0.0167, 0.0309, 0.0459, 0.0622, 0.072, 0.0814, 0.1021, 0.1245, 0.1547, 0.183, 0.209]
y_exp_3 = [0.0196, 0.0592, 0.0946, 0.1553, 0.2021, 0.2367, 0.2534, 0.2646, 0.2811, 0.2775, 0.258, 0.2295, 0.209]

# PONTOS EXPERIMENTAIS [CH4, CO2] Donnelly e Katz, 1954
# T=271.48
# P_exp_5 = [50.53778268, 55.91560949, 59.98345284, 68.11913955, 72.53171539, 76.39271925]
# x_exp_5 = [0.0675, 0.084, 0.103, 0.16, 0.165, 0.191]
# y_exp_5 = [0.253, 0.3, 0.329, 0.367, 0.387, 0.39]
# # T=259.82
# P_exp_4 = [31.9222283507998, 34.6800882515168, 36.8863761720905, 40.1958080529509, 50.5377826806398, 60.3281853281853,
#            68.119139547711, 70.8080529509101]
# x_exp_4 = [0.0315, 0.036, 0.051, 0.053, 0.1095, 0.164, 0.224, 0.235]
# y_exp_4 = [0.1885, 0.235, 0.266, 0.306, 0.425, 0.485, 0.505, 0.495]
# # T=241.48
# P_exp_3 = [23.9244346387204, 31.0259238830667, 40.7473800330943, 47.0215113072256, 52.6061776061776, 62.6723662437948,
#             68.3949255377827, 75.7032542746828, 79.0126861555433]
# x_exp_3 = [0.0413, 0.086, 0.137, 0.166, 0.191, 0.286, 0.322, 0.426, 0.501]
# y_exp_3 = [0.404, 0.521, 0.605, 0.629, 0.652, 0.676, 0.686, 0.68, 0.672]
# # T=223.71
# P_exp_2 = [14.8234969663541, 34.33535576392721, 40.4026475455047, 53.9161610590182, 60.1213458356316, 
#            61.9838924, 63.7065637065637, 65.4302261445119]
# x_exp_2 = [0.0435, 0.1465, 0.1945, 0.309, 0.392, 0.465, 0.483, 0.525]
# y_exp_2 = [0.509, 0.751, 0.772, 0.792, 0.797, 0.805, 0.783, 0.79]

# # T=199.82
# P_exp_1 = [44.8841698841699, 49.8483177054606]
# x_exp_1 = [0.77, 0.91]
# y_exp_1 = [0.926, 0.941]


# plt.plot(dict_data[file_1][2], dict_data[file_1][1], color='black', linewidth=1.25)
# plt.plot(dict_data[file_1][3], dict_data[file_1][1], color='black', linewidth=1.25, label=r'$T=199.81\;[K]$')
# plt.scatter(x_exp_1, P_exp_1, marker='x', color='black')
# plt.scatter(y_exp_1, P_exp_1, marker='x', color='black')


# plt.plot(dict_data[file_2][2], dict_data[file_2][1], color='black', linewidth=1.25)
# plt.plot(dict_data[file_2][3], dict_data[file_2][1], color='black', linewidth=1.25, label=r'$T=223.71\;[K]$')
# plt.scatter(x_exp_2, P_exp_2, marker='x', color='black')
# plt.scatter(y_exp_2, P_exp_2, marker='x', color='black')


# plt.plot(dict_data[file_3][2], dict_data[file_3][1], color='darkblue', linewidth=1.25)
# plt.plot(dict_data[file_3][3], dict_data[file_3][1], color='darkblue', linewidth=1.25, label=r'$T=241.48\;[K]$')
# plt.scatter(x_exp_3, P_exp_3, marker='x', color='darkblue')
# plt.scatter(y_exp_3, P_exp_3, marker='x', color='darkblue')


# plt.plot(dict_data[file_4][2], dict_data[file_4][1], color='olive', linewidth=1.25)
# plt.plot(dict_data[file_4][3], dict_data[file_4][1], color='olive', linewidth=1.25, label=r'$T=259.82\;[K]$')
# plt.scatter(x_exp_4, P_exp_4, marker='x', color='olive')
# plt.scatter(y_exp_4, P_exp_4, marker='x', color='olive')



file_name = file_2
plt.plot(dict_data[file_name][2], dict_data[file_name][1], color='black', linewidth=1.25)
plt.plot(dict_data[file_name][3], dict_data[file_name][1], color='black', linewidth=1.25, label=r'$T=271.48\;[K]$')
plt.scatter(x_exp_3, P_exp_3, marker='x', color='black')
plt.scatter(y_exp_3, P_exp_3, marker='x', color='black')


plt.ylabel(ylabel=r'$P\;/\;bar$')
plt.xlabel(xlabel=r'$x_{CH_{4}}\;/\;y_{CH_{4}}$')
plt.xlim(left=0.0, right=max(dict_data[file_name][3])*1.10)
plt.ylim(bottom=dict_data[file_name][1][0]*0.8)
# plt.legend()

plt.savefig((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\img\\{file_name}.png'), dpi=600)
plt.show()