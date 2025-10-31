# # coisasa para o hard chain


# # coisas para o dispersion
# def _compute_detaI1I2_dxk(eta: float, am: np.ndarray, bm: np.ndarray, am_xk: np.ndarray, bm_xk: np.ndarray, zeta3_xk: np.ndarray):
#     j = np.arange(7)[:, None]
#     detaI1_deta_xk = np.sum((j + 1) * (am_xk.T * eta**j + j * eta**(j - 1) * am[:, None] * zeta3_xk.T), axis=0)
#     detaI2_deta_xk = np.sum((j + 1) * (bm_xk.T * eta**j + j * eta**(j - 1) * bm[:, None] * zeta3_xk.T), axis=0)
   
#     return detaI1_deta_xk, detaI2_deta_xk

# def _compute_C2_xk(m: np.ndarray, m_mean: float, eta: float, zeta3_xk: np.ndarray, C1: float, C1_xk: np.ndarray):

#     u = m_mean
#     u_xk = m
    
#     o = - 4 * eta**2 + 20 * eta + 8
#     o_xk = -8 * eta * zeta3_xk + 20 * zeta3_xk
#     p = (1 - eta)**5
#     p_xk = - 5 * (1 - eta)**4 * zeta3_xk
#     v = o / p
#     v_xk = (o_xk * p - o * p_xk) / p**2


#     a = (1 - m_mean)
#     a_xk = - m
    
#     o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
#     o_xk = 6 * eta**2 * zeta3_xk + 24 * eta * zeta3_xk - 48 * zeta3_xk
#     p = (eta**2 - 3 * eta + 2)**3
#     p_xk = 3 * (2 * eta * zeta3_xk - 3 * zeta3_xk) * (eta**2 - 3 * eta + 2)**2

#     b = o / p
#     b_xk = (o_xk * p - o * p_xk) / p**2

#     s = - C1**2
#     s_xk = - 2 * C1 * C1_xk

#     t = u * v + a * b
#     t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

#     C2_xk = s_xk * t + s * t_xk
    
#     return C2_xk

# def _compute_dZdisp_dxk(rho: float, eta: float, detaI1_eta: float, detaI2_eta: float, detaI1_xk: np.ndarray, detaI2_xk: np.ndarray,
#                         m: np.ndarray, m_mean: float, C1: float, C2: float, C1_xk: np.ndarray, C2_xk: np.ndarray,
#                         m2es3: float, m2es3_xk: np.ndarray, zeta3_xk: np.ndarray, I2: float, I2_xk: np.ndarray,
#                         m2e2s3: float, m2e2s3_xk: np.ndarray):
    
#     termo_1 = - 2 * np.pi * rho * (detaI1_xk * m2es3 + detaI1_eta * m2es3_xk)

#     u = m_mean * C1
#     du = (m * C1 + m_mean * C1_xk)
#     v = detaI2_eta * m2e2s3
#     dv = (detaI2_xk * m2e2s3 + detaI2_eta * m2e2s3_xk)
#     termo_2 = - np.pi * rho * (du * v + u * dv)
#     u = m_mean * C2 * eta
#     du = m * (C2 * eta) + m_mean * (C2_xk * eta + C2 * zeta3_xk)
#     v = I2 * m2e2s3
#     dv = I2_xk * m2e2s3 + I2 * m2e2s3_xk
#     termo_3 = - np.pi * rho * (du * v + u * dv)

#     dZdips_xk = termo_1 + termo_2 + termo_3

    

    
#     return dZdips_xk



# def _compute_I1I1_xjxk(eta: float, a: np.ndarray, b: np.ndarray, ai_xk:np.ndarray, ai_xjxk: np.ndarray, 
#                        bi_xk: np.ndarray, bi_xjxk: np.ndarray, zeta3_xk: np.ndarray):
    
#     i_vec = np.arange(7)
#     i_minus1 = i_vec - 1
#     i_minus2 = i_vec - 2
#     eta_pow_i = np.power(eta, i_vec)
#     eta_pow_i_minus1 = np.power(eta, i_minus1)
#     eta_pow_i_minus2 = np.power(eta, i_minus2)
#     zeta3_xjxk = np.outer(zeta3_xk, zeta3_xk)

#     # Construção do I1_xjxk
#     term1 = np.einsum('ijk,i->jk', ai_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', ai_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', ai_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I1_xjxk = term1 + term2 + term3

#     # Construção do I12_xjxk
#     term1 = np.einsum('ijk,i->jk', bi_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', bi_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', bi_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I2_xjxk = term1 + term2 + term3

#     return I1_xjxk, I2_xjxk


# def _compute_m2es3_m2e2s3_xjxk(m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float):
#     mjmk = np.outer(m, m)
#     m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
#     m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

#     return m2es3_xjxk, m2e2s3_xjxk

# def _compute_C1_xjxk(eta: float, m:np.ndarray,  C1: float, C1_xk:np.ndarray, C2_xk: np.ndarray, zeta3_xk: np.ndarray):

#     C2_xj_zeta3_xk = np.outer(C2_xk, zeta3_xk)
#     term1 = C2_xj_zeta3_xk

#     # as funcoes de eta dentro do parenteses
#     u = 8 * eta - 2 * eta**2
#     du = 8 * zeta3_xk - 4 * eta * zeta3_xk
#     v = (1 - eta)**4
#     dv = - 4 * (1 - eta)**3 * zeta3_xk
#     s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
#     ds = 20 * zeta3_xk - 54 * eta * zeta3_xk + 36 * eta**2 * zeta3_xk - 8 * eta**3 * zeta3_xk
#     t = (2 - 3 * eta + eta**2)**2
#     dt = 2 * t**0.5 * (2 * eta * zeta3_xk - 3 * zeta3_xk)
#     aux1_xj = (du * v - u * dv) / v**2
#     aux2_xj = (ds * t - s * dt) / t**2
#     func_aux = u/v - s/t
#     term_aux_xj= 2 * C1 * C1_xk * func_aux + C1**2 * (aux1_xj - aux2_xj)
#     term2 = np.outer(term_aux_xj, m)
#     C1_xjxk = term1 - term2

#     return C1_xjxk

# def _compute_dadisp_xjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, I1_xk:np.ndarray, I2_xk:np.ndarray, 
#                          I1_xjxk:np.ndarray, I2_xjxk:np.ndarray,m2es3:float, m2e2s3:float, m2es3_xk:np.ndarray, m2e2s3_xk:np.ndarray,
#                            m2es3_xjxk:np.ndarray, m2e2s3_xjxk:np.ndarray,
#                          C1:float, C1_xk:np.ndarray, C1_xjxk:np.ndarray):
    
#     m2es3_xj_I1_xk = np.outer(m2es3_xk, I1_xk)
    
#     term1_xj = - 2 * np.pi * rho * (I1_xjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + m2es3_xjxk * I1)

#     aux = I2 * m2e2s3
#     aux_xj = (I2_xk * m2e2s3 + I2 * m2e2s3_xk)
#     aux1_xj = C1_xk * aux + C1 * aux_xj
#     aux1_xj = np.outer(aux1_xj, m)

#     u = m_mean * C1_xk
#     du = np.outer(m, C1_xk) + m_mean * C1_xjxk
#     v = aux
#     dv = aux_xj
#     aux2_xj = du * v + np.outer(dv, u)

#     s = m_mean * C1
#     ds = m * C1 + m_mean * C1_xk
#     t = I2_xk * m2e2s3
#     dt = I2_xjxk * m2e2s3 + np.outer(m2e2s3_xk, I2_xk)
#     aux_3_xj = np.outer(ds, t) + s * dt

#     m = I2 * m2e2s3_xk
#     dm = np.outer(I2_xk, m2e2s3_xk) + I2 * m2e2s3_xjxk
#     aux_4_xj = np.outer(ds, m) + s * dm

#     term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
    
#     dadisp_xjxk = term1_xj + term2_xj
#     return dadisp_xjxk


# def _compute_dlnphik_xj_unc(z:np.ndarray, Z:float, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
#     term1 = ((1 - 1 / Z) * dZ_xk)[:, None]
    
#     term3 = (- np.einsum('ji,i->j', dares_xjxk, z))[:, None]

#     dlnphik_xj_unc = term1 + dares_xjxk + term3

#     return dlnphik_xj_unc
 

# def _compute_dlnphik_xj_cons(z:np.ndarray, dlnphik_xj_unc:np.ndarray):
#     N = len(z)

#     A_ij = np.eye(N) - z[:, None]
#     M_ik = dlnphik_xj_unc
#     dlnphik_xj_cons = np.einsum('ij,ik->jk', A_ij, M_ik)
#     return dlnphik_xj_cons

# def _compute_dlnphik_nj(z:np.ndarray, dlnphik_xj_cons:np.ndarray, n:float=100.0):
#     sum_term = - np.einsum('i,ki->k', z, dlnphik_xj_cons)[:, None]
#     n_dlnphik_nj = dlnphik_xj_cons + sum_term
#     dlnphik_nj = n_dlnphik_nj / n

#     # print(n_dlnphik_nj)
#     # print(dlnphik_nj)

#     ni = z * n



# # ---------------- TUDO DO HARDCHAIN
# def _compute_dzhs_dxk(zeta_xk: np.ndarray, zeta: np.ndarray):
#         zeta_aux = 1 - zeta[3]

#         termo_1 = zeta_xk[3, :] / zeta_aux**2

#         u = zeta[1] * zeta[2]
#         du = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
#         v = zeta[0] * zeta_aux**2
#         dv = zeta_xk[0,:] * zeta_aux**2 - 2 * zeta[0] * zeta_aux * zeta_xk[3,:]
#         termo_2 = 3 * (du * v - dv * u) / v**2

#         u = 3 * zeta[2]**3 - zeta[3] * zeta[2]**3
#         du = 9 * zeta[2]**2 * zeta_xk[2,:] - (zeta_xk[3,:] * zeta[2]**3 + 3 * zeta[3] * zeta[2]**2 * zeta_xk[2,:])
#         v = zeta[0] * zeta_aux**3
#         dv = zeta_xk[0,:] * zeta_aux**3 - 3 * zeta[0] * zeta_aux**2 * zeta_xk[3,:]
#         termo_3 = (du * v - u * dv) / v**2

#         dzhs_dxk = termo_1 + termo_2 + termo_3 

#         return dzhs_dxk
#         # print('dzhs_dxi: ', dzhs_dxi)


# def _compute_dgij_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
#     """
#     eu acho que ja tenho..?
#     Returns: 
#         Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
#     """
#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta[3]

#     termo_1 = zeta_xk[3,:] / zeta_aux**2

#     u = 3 * zeta[2]
#     du = 3 * zeta_xk[2,:]
#     v = zeta_aux**2
#     dv = - 2 * zeta_aux * zeta_xk[3,:]
#     termo_2 = (du * v - u * dv) / v**2

#     u = 2 * zeta[2]**2
#     du = 4 * zeta[2] * zeta_xk[2,:]
#     v = zeta_aux**3
#     dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
#     termo_3 = (du * v - u * dv) / v**2

#     dgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
#     return dgij_dxk

# def _compute_drhodhji_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
#     """
    
#     Returns: 
#         Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
#     """
#     _, _, zeta2, zeta3 = zeta
#     _, _, zeta2_xk, zeta3_xk = zeta_xk

#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta3

#     termo_1 = (zeta3_xk * zeta_aux**2 - (zeta3) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4

#     termo_2 = ((3 * zeta2_xk) * zeta_aux**2 - (3 * zeta2) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4
    
#     termo_2 += ((6 * (zeta2_xk * zeta3 + zeta2 * zeta3_xk)) * zeta_aux**3 - (6 * zeta2 * zeta3) * (- 3 *zeta_aux**2 * zeta3_xk)) / zeta_aux**6

#     termo_3 = ((8 * zeta2 * zeta2_xk) * zeta_aux**3 - (4 * zeta2**2) * ((- 3 *zeta_aux**2 * zeta3_xk))) / zeta_aux**6

#     u = 6 * zeta2**2 * zeta3
#     du = 6 * (2 * zeta2 * zeta3 * zeta2_xk + zeta2**2 * zeta3_xk)
#     v = zeta_aux**4
#     dv = - 4 * zeta_aux**3 * zeta3_xk
#     termo_3 += (du * v - u * dv) / v**2

#     drhodgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
#     return drhodgij_dxk


# def _compute_dZhc_dxk(z: np.ndarray, m_mean: float, m:np.ndarray, Zhs: float, dZhs_dk: np.ndarray, gij: np.ndarray, rho_dgij_drho: np.ndarray,
#                       dgij_dxk: np.ndarray, drhodhji_dxk: np.ndarray):

#     termo_1_vec = m * Zhs + m_mean * dZhs_dk

#     # Termos do somatorio
#     gkk = np.diagonal(gij)
#     rho_dgkk_drho = np.diagonal(rho_dgij_drho)
#     sum_1_factor = (m - 1) * rho_dgkk_drho / gkk
    
#     dgii_dxk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
#     drhodhii_dxk = np.diagonal(drhodhji_dxk, axis1=0, axis2=1).T
#     gii = gkk
#     rho_dgii_drho = rho_dgkk_drho

#     factor_aux = (m - 1) * z * rho_dgii_drho / (-gii**2)
#     sum_2_factor = np.sum(factor_aux[:, None] * dgii_dxk, axis=0)

#     factor_aux = (m - 1) * z / gii
#     sum_3_factor = np.sum(factor_aux[:, None] * drhodhii_dxk, axis=0)
    
#     termo_2_vec = sum_1_factor + sum_2_factor + sum_3_factor

#     dZhc_dxk = termo_1_vec - termo_2_vec

    
#     return dZhc_dxk

# def _compute_dZhs_dT(zeta:np.ndarray, dzeta_dT:np.ndarray):
#     zeta0, zeta1, zeta2, zeta3 = zeta
#     dzeta1_dT, dzeta2_dT, dzeta3_dT = dzeta_dT
#     zeta_aux = 1 - zeta3
#     u = zeta3
#     du = dzeta3_dT
#     v = zeta_aux
#     dv = -dzeta3_dT
#     f1 = u / v
#     df1_dT = (du * v - u * dv) / v**2

#     u = 3 * zeta1 * zeta2
#     du = 3 * (dzeta1_dT * zeta2 + zeta1 * dzeta2_dT)
#     v = zeta0 * zeta_aux**2
#     dv = 0.0 - 2 * zeta0 * zeta_aux * dzeta3_dT
#     f2 = u / v
#     df2_dT = (du * v - u * dv) / v**2

#     u = 3 * zeta2**3 - zeta3 * zeta2**3
#     du = 9 * zeta2**2 * dzeta2_dT - (dzeta3_dT * zeta2**3 + 3 * zeta3 * zeta2**2 * dzeta2_dT)
#     v = zeta0 * zeta_aux**3
#     dv = 0.0 - 3 * zeta0 * zeta_aux**2 * dzeta3_dT
#     f3 = u / v
#     df3_dT = (du * v - u * dv) / v**2

#     dZhs_dT = df1_dT + df2_dT + df3_dT
#     return dZhs_dT

# def _compute_dgij_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
#     _, _, zeta2, zeta3 = zeta
#     _, dzeta2_dT, dzeta3_dT = dzeta_dT

#     zeta_aux = 1 - zeta3
    
#     df1_dT = dzeta3_dT / zeta_aux**2

#     u =  3 * zeta2
#     v = zeta_aux**2
#     f2 = u /v
#     df2_dT = ((3 * dzeta2_dT) * v - u * (- 2 * zeta_aux * dzeta3_dT)) / v**2

#     u = 2 * zeta2**2
#     v = zeta_aux**3
#     f3 = u / v
#     df3_dT = (4 * zeta2 * dzeta2_dT * v - u * (- 3 * zeta_aux**2 * dzeta3_dT)) / v**2

#     dgij_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
#     return dgij_dT

# def _compute_drho_dgij_rho_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
#     _, _, zeta2, zeta3 = zeta
#     _, dzeta2_dT, dzeta3_dT = dzeta_dT

#     zeta_aux = 1 - zeta3

#     f1 = zeta3 / zeta_aux**2
#     df1_dT = (dzeta3_dT * zeta_aux**2 - zeta3 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

#     f21 = 3 * zeta2 / zeta_aux**2
#     df21_dT = (3 * dzeta2_dT * zeta_aux**2 - 3 * zeta2 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

#     f22 = 6 * zeta2 * zeta3 / zeta_aux**3
#     df22_dT = (6 * (dzeta2_dT * zeta3 + zeta2 * dzeta3_dT) * zeta_aux**3 - 6 * zeta2 * zeta3 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6

#     f2 = f21 + f22
#     df2_dT = df21_dT + df22_dT

#     f31 = 4 * zeta2**2 / zeta_aux**3
#     df31_dT = (8 * zeta2 * dzeta2_dT * zeta_aux**3 - 4 * zeta2**2 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6
    
#     f32 = 6 * zeta2**2 * zeta3 / zeta_aux**4
#     df32_dT = (6 * (2 * zeta2 * dzeta2_dT * zeta3 + zeta2**2 * dzeta3_dT) * zeta_aux**4 - 6 * zeta2**2 * zeta3 * (-4 * zeta_aux**3 * dzeta3_dT)) / zeta_aux**8

#     f3 = f31 + f32
#     df3_dT = df31_dT + df32_dT

#     drho_dgij_rho_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
#     return drho_dgij_rho_dT

# def _compute_dZhc_dT(z:np.ndarray, m:np.ndarray, m_mean:float, dZhs_dT:float, gij:np.ndarray, rho_dgij_drho:np.ndarray,
#                         dgij_dT:np.ndarray, drho_dgij_drho_dT:np.ndarray):
#     gii = np.diagonal(gij)
#     dgii_dT = np.diagonal(dgij_dT)
#     rho_dgii_drho = np.diagonal(rho_dgij_drho)
#     drho_dgii_drho_dT = np.diagonal(drho_dgij_drho_dT)

#     term1 = m_mean * dZhs_dT

#     f1 = - gii**-2 * dgii_dT * rho_dgii_drho
#     f2 = gii**-1 * drho_dgii_drho_dT
#     term2 = - np.sum(z * (m - 1) * (f1 + f2))

#     dZhc_dT = term1 + term2
#     return dZhc_dT


# def _compute_dash_xjxk(zeta: np.ndarray, zeta_xk: np.ndarray, ahs: float, dahs_xk: np.ndarray):
#     zeta_aux = 1 - zeta[3]
#     termo_1 = np.outer(zeta_xk[0,:], zeta_xk[0,:]) * ahs / zeta[0]**2 - np.outer(dahs_xk, zeta_xk[0,:]) / zeta[0]

#     # T1
#     u = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
#     du = np.outer(zeta_xk[2,:], zeta_xk[1,:]) + np.outer(zeta_xk[1,:], zeta_xk[2,:])
#     v = zeta_aux
#     dv = - zeta_xk[3,:]
#     T1 = 3 * u / v
#     T1_xj = 3 * (du * v - np.outer(dv, u)) / v**2

#     # T2
#     u = 3 * zeta[1] * zeta[2] * zeta_xk[3,:]
#     v = zeta_aux**2
#     du = 3 * (np.outer(zeta_xk[1,:], zeta_xk[3,:]) * zeta[2] + np.outer(zeta_xk[2,:],zeta_xk[3,:]) * zeta[1])
#     dv = - 2 * zeta_aux * zeta_xk[3,:]
#     T2 = u / v
#     T2_xj = (du * v - np.outer(dv, u)) / v**2

#     # T3
#     u = 3 * zeta[2]**2 * zeta_xk[2,:]
#     du = 6 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[2,:])
#     v = zeta[3] * zeta_aux**2
#     dv = zeta_xk[3,:] * zeta_aux**2 - 2 * zeta[3] * zeta_aux * zeta_xk[3,:]
#     T3 = u / v
#     T3_xj = (du * v - np.outer(dv, u)) / v**2

#     # T4
#     u = zeta[2]**3 * zeta_xk[3,:] * (3 * zeta[3] - 1)
#     du = 3 * zeta[2]**2 * (3 * zeta[3] - 1) * np.outer(zeta_xk[2,:], zeta_xk[3,:]) + 3 * zeta[2]**3 * np.outer(zeta_xk[3,:], zeta_xk[3,:])
#     v = zeta[3]**2 * zeta_aux**3
#     dv = 2 * zeta[3] * zeta_aux**3 * zeta_xk[3,:] - 3 * zeta[3]**2 * zeta_aux**2 * zeta_xk[3,:]
#     T4 = u / v
#     T4_xj = (du * v - np.outer(dv, u)) / v**2

#     # T5
#     u = 3 * zeta[2]**2 * zeta[3] * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta_xk[3,:]
#     v = zeta[3]**3
#     du = 6 * zeta[2] * zeta[3] * np.outer(zeta_xk[2,:], zeta_xk[2,:]) + 3 * zeta[2]**2 * np.outer(zeta_xk[3,:], zeta_xk[2,:]) - 6 * zeta[2]**2 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     dv = 3 * zeta[3]**2 * zeta_xk[3,:]

#     T5_1 = (u / v) - zeta_xk[0,:]
#     T5_1xj = (du * v - np.outer(dv, u)) / v**2
#     T5_2 = np.log(zeta_aux)
#     T5_2xj = - zeta_xk[3,:] / zeta_aux
#     T5 = T5_1 * T5_2
#     T5_xj = T5_1xj * T5_2 + np.outer(T5_2xj, T5_1)

#     # T6
#     T6_1 = zeta[0] - zeta[2]**3 / zeta[3]**2
#     T6_1xj = zeta_xk[0,:] - (3 * zeta[2]**2 * zeta[3]**2 * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta[3] * zeta_xk[3,:]) / zeta[3]**4
#     T6_2 = zeta_xk[3,:] / zeta_aux
#     T6_2xj = np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**2
#     T6 = T6_1 * T6_2
#     T6_xj = np.outer(T6_1xj, T6_2) + T6_1 * T6_2xj

#     T = T1 + T2 + T3 + T4 + T5 + T6
#     T_xj = T1_xj + T2_xj + T3_xj + T4_xj + T5_xj + T6_xj
   
#     termo_2 = - np.outer(zeta_xk[0,:], T) / zeta[0]**2 + T_xj / zeta[0]

#     dahs_xjxk = termo_1 + termo_2
    
#     return dahs_xjxk


# def _compute_dgij_xjxk(d: np.ndarray, zeta: np.ndarray, zeta_xk: np.ndarray):
#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta[3]
#     termo_1 = 2 * np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**3


#     termo_21 = (6 / zeta_aux**3) * np.outer(zeta_xk[3,:], zeta_xk[2,:])

#     u = 6 * zeta[2] * zeta_xk[3,:]
#     du = 6 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     v = zeta_aux**3
#     dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
#     termo_22 = (du * v - np.outer(dv, u)) / v**2

#     termo_2 = termo_21 + termo_22

#     u = 4 * zeta[2] * zeta_xk[2,:]
#     du = 4 * np.outer(zeta_xk[2,:], zeta_xk[2,:])
#     termo_31 = (du * v - np.outer(dv, u)) / v**2
    
#     u = 6 * zeta[2]**2 * zeta_xk[3,:]
#     du = 12 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     v = zeta_aux**4
#     dv = - 4 * zeta_aux**3 *zeta_xk[3,:]
#     termo_32 = (du * v - np.outer(dv, u)) / v**2

#     termo_3 = termo_31 + termo_32
#     dgij_xjxk = termo_1[None, None, :, :] + Dij[:, :, None, None] * termo_2[None, None, :, :] + Dij[:, :, None, None]**2 * termo_3[None, None, :, :]
#     # AQUI TEM QUE VER MESMO SE GERA UM TENSOR (n,n,n,n)
#     return dgij_xjxk

# def _dahc_xjxk(z: np.ndarray, m: np.ndarray, m_mean: float, gij: np.ndarray, dgij_xk: np.ndarray, dgij_xjxk: np.ndarray,
#                dahs_xk: np.ndarray, dahs_xjxk: np.ndarray):
#     gii = np.diagonal(gij)
#     gii_inv = 1 / gii
#     gii_inv_sq = gii_inv**2

#     dgii_xk = np.diagonal(dgij_xk, axis1=0, axis2=1).T
#     # dgii_xjxk = np.einsum('iikj->jk', dgij_xjxk)

#     # Termo 1: mₖ (∂ãʰˢ/∂xⱼ) -> Matriz [j, k] = mₖ * (∂ãʰˢ/∂xⱼ)
#     term1 = np.outer(dahs_xk, m)

#     # Termo 2: mₖ (∂ãʰˢ/∂xₖ) -> Matriz [k, j] = mⱼ * (∂ãʰˢ/∂xₖ)
#     term2 = np.outer(m, dahs_xk)

#     # Termo 3: m̄ (∂²ãʰˢ/∂xⱼ∂xₖ) -> Escalar * Matriz
#     term3 = m_mean * dahs_xjxk

#     # Termo 4a: - (mⱼ-1)(gⱼⱼ)⁻¹ (∂gⱼⱼ/∂xₖ) -> Matriz [j, k]
#     term4a = - (m - 1.0)[:, None] * gii_inv[:, None] * dgii_xk

#     # Termo 4b: + Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻² (∂gᵢᵢ/∂xⱼ) (∂gᵢᵢ/∂xₖ) -> Matriz [j, k]
#     sum_4b = z * (m - 1.0) * gii_inv_sq # Vetor (N,)
#     term4b = np.einsum('i,ij,ik->jk', sum_4b, dgii_xk, dgii_xk)

#     # Termo 4c: - Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻¹ (∂²gᵢᵢ/∂xⱼ∂xₖ) -> Matriz [j, k]
#     dgii_xjxk = np.diagonal(dgij_xjxk, axis1=0, axis2=1).transpose(2, 0, 1)
#     sum_4c = z * (m - 1.0) * gii_inv # Vetor (N,)
#     term4c = -np.einsum('i,ijk->jk', sum_4c, dgii_xjxk)

#     # Termo 5: - (mₖ-1)(gₖₖ)⁻¹ (∂gₖₖ/∂xⱼ) -> Matriz [j, k]
#     # dgii_dxk.T[k, j] = ∂gₖₖ/∂xⱼ
#     term5 = - (m - 1.0)[None, :] * gii_inv[None, :] * dgii_xk.T

#     # 3. Soma Final -> Matriz NxN
#     dahc_xjxk = term1 + term2 + term3 + term4a + term4b + term4c + term5
#     return dahc_xjxk





# coisasa para o hard chain


# # coisas para o dispersion
# def _compute_detaI1I2_dxk(eta: float, am: np.ndarray, bm: np.ndarray, am_xk: np.ndarray, bm_xk: np.ndarray, zeta3_xk: np.ndarray):
#     j = np.arange(7)[:, None]
#     detaI1_deta_xk = np.sum((j + 1) * (am_xk.T * eta**j + j * eta**(j - 1) * am[:, None] * zeta3_xk.T), axis=0)
#     detaI2_deta_xk = np.sum((j + 1) * (bm_xk.T * eta**j + j * eta**(j - 1) * bm[:, None] * zeta3_xk.T), axis=0)
   
#     return detaI1_deta_xk, detaI2_deta_xk

# def _compute_C2_xk(m: np.ndarray, m_mean: float, eta: float, zeta3_xk: np.ndarray, C1: float, C1_xk: np.ndarray):

#     u = m_mean
#     u_xk = m
    
#     o = - 4 * eta**2 + 20 * eta + 8
#     o_xk = -8 * eta * zeta3_xk + 20 * zeta3_xk
#     p = (1 - eta)**5
#     p_xk = - 5 * (1 - eta)**4 * zeta3_xk
#     v = o / p
#     v_xk = (o_xk * p - o * p_xk) / p**2


#     a = (1 - m_mean)
#     a_xk = - m
    
#     o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
#     o_xk = 6 * eta**2 * zeta3_xk + 24 * eta * zeta3_xk - 48 * zeta3_xk
#     p = (eta**2 - 3 * eta + 2)**3
#     p_xk = 3 * (2 * eta * zeta3_xk - 3 * zeta3_xk) * (eta**2 - 3 * eta + 2)**2

#     b = o / p
#     b_xk = (o_xk * p - o * p_xk) / p**2

#     s = - C1**2
#     s_xk = - 2 * C1 * C1_xk

#     t = u * v + a * b
#     t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

#     C2_xk = s_xk * t + s * t_xk
    
#     return C2_xk

# def _compute_dZdisp_dxk(rho: float, eta: float, detaI1_eta: float, detaI2_eta: float, detaI1_xk: np.ndarray, detaI2_xk: np.ndarray,
#                         m: np.ndarray, m_mean: float, C1: float, C2: float, C1_xk: np.ndarray, C2_xk: np.ndarray,
#                         m2es3: float, m2es3_xk: np.ndarray, zeta3_xk: np.ndarray, I2: float, I2_xk: np.ndarray,
#                         m2e2s3: float, m2e2s3_xk: np.ndarray):
    
#     termo_1 = - 2 * np.pi * rho * (detaI1_xk * m2es3 + detaI1_eta * m2es3_xk)

#     u = m_mean * C1
#     du = (m * C1 + m_mean * C1_xk)
#     v = detaI2_eta * m2e2s3
#     dv = (detaI2_xk * m2e2s3 + detaI2_eta * m2e2s3_xk)
#     termo_2 = - np.pi * rho * (du * v + u * dv)
#     u = m_mean * C2 * eta
#     du = m * (C2 * eta) + m_mean * (C2_xk * eta + C2 * zeta3_xk)
#     v = I2 * m2e2s3
#     dv = I2_xk * m2e2s3 + I2 * m2e2s3_xk
#     termo_3 = - np.pi * rho * (du * v + u * dv)

#     dZdips_xk = termo_1 + termo_2 + termo_3

    

    
#     return dZdips_xk



# def _compute_I1I1_xjxk(eta: float, a: np.ndarray, b: np.ndarray, ai_xk:np.ndarray, ai_xjxk: np.ndarray, 
#                        bi_xk: np.ndarray, bi_xjxk: np.ndarray, zeta3_xk: np.ndarray):
    
#     i_vec = np.arange(7)
#     i_minus1 = i_vec - 1
#     i_minus2 = i_vec - 2
#     eta_pow_i = np.power(eta, i_vec)
#     eta_pow_i_minus1 = np.power(eta, i_minus1)
#     eta_pow_i_minus2 = np.power(eta, i_minus2)
#     zeta3_xjxk = np.outer(zeta3_xk, zeta3_xk)

#     # Construção do I1_xjxk
#     term1 = np.einsum('ijk,i->jk', ai_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', ai_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', ai_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I1_xjxk = term1 + term2 + term3

#     # Construção do I12_xjxk
#     term1 = np.einsum('ijk,i->jk', bi_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', bi_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', bi_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I2_xjxk = term1 + term2 + term3

#     return I1_xjxk, I2_xjxk


# def _compute_m2es3_m2e2s3_xjxk(m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float):
#     mjmk = np.outer(m, m)
#     m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
#     m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

#     return m2es3_xjxk, m2e2s3_xjxk

# def _compute_C1_xjxk(eta: float, m:np.ndarray,  C1: float, C1_xk:np.ndarray, C2_xk: np.ndarray, zeta3_xk: np.ndarray):

#     C2_xj_zeta3_xk = np.outer(C2_xk, zeta3_xk)
#     term1 = C2_xj_zeta3_xk

#     # as funcoes de eta dentro do parenteses
#     u = 8 * eta - 2 * eta**2
#     du = 8 * zeta3_xk - 4 * eta * zeta3_xk
#     v = (1 - eta)**4
#     dv = - 4 * (1 - eta)**3 * zeta3_xk
#     s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
#     ds = 20 * zeta3_xk - 54 * eta * zeta3_xk + 36 * eta**2 * zeta3_xk - 8 * eta**3 * zeta3_xk
#     t = (2 - 3 * eta + eta**2)**2
#     dt = 2 * t**0.5 * (2 * eta * zeta3_xk - 3 * zeta3_xk)
#     aux1_xj = (du * v - u * dv) / v**2
#     aux2_xj = (ds * t - s * dt) / t**2
#     func_aux = u/v - s/t
#     term_aux_xj= 2 * C1 * C1_xk * func_aux + C1**2 * (aux1_xj - aux2_xj)
#     term2 = np.outer(term_aux_xj, m)
#     C1_xjxk = term1 - term2

#     return C1_xjxk

# def _compute_dadisp_xjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, I1_xk:np.ndarray, I2_xk:np.ndarray, 
#                          I1_xjxk:np.ndarray, I2_xjxk:np.ndarray,m2es3:float, m2e2s3:float, m2es3_xk:np.ndarray, m2e2s3_xk:np.ndarray,
#                            m2es3_xjxk:np.ndarray, m2e2s3_xjxk:np.ndarray,
#                          C1:float, C1_xk:np.ndarray, C1_xjxk:np.ndarray):
    
#     m2es3_xj_I1_xk = np.outer(m2es3_xk, I1_xk)
    
#     term1_xj = - 2 * np.pi * rho * (I1_xjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + m2es3_xjxk * I1)

#     aux = I2 * m2e2s3
#     aux_xj = (I2_xk * m2e2s3 + I2 * m2e2s3_xk)
#     aux1_xj = C1_xk * aux + C1 * aux_xj
#     aux1_xj = np.outer(aux1_xj, m)

#     u = m_mean * C1_xk
#     du = np.outer(m, C1_xk) + m_mean * C1_xjxk
#     v = aux
#     dv = aux_xj
#     aux2_xj = du * v + np.outer(dv, u)

#     s = m_mean * C1
#     ds = m * C1 + m_mean * C1_xk
#     t = I2_xk * m2e2s3
#     dt = I2_xjxk * m2e2s3 + np.outer(m2e2s3_xk, I2_xk)
#     aux_3_xj = np.outer(ds, t) + s * dt

#     m = I2 * m2e2s3_xk
#     dm = np.outer(I2_xk, m2e2s3_xk) + I2 * m2e2s3_xjxk
#     aux_4_xj = np.outer(ds, m) + s * dm

#     term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
    
#     dadisp_xjxk = term1_xj + term2_xj
#     return dadisp_xjxk


# def _compute_dlnphik_xj_unc(z:np.ndarray, Z:float, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
#     term1 = ((1 - 1 / Z) * dZ_xk)[:, None]
    
#     term3 = (- np.einsum('ji,i->j', dares_xjxk, z))[:, None]

#     dlnphik_xj_unc = term1 + dares_xjxk + term3

#     return dlnphik_xj_unc
 

# def _compute_dlnphik_xj_cons(z:np.ndarray, dlnphik_xj_unc:np.ndarray):
#     N = len(z)

#     A_ij = np.eye(N) - z[:, None]
#     M_ik = dlnphik_xj_unc
#     dlnphik_xj_cons = np.einsum('ij,ik->jk', A_ij, M_ik)
#     return dlnphik_xj_cons

# def _compute_dlnphik_nj(z:np.ndarray, dlnphik_xj_cons:np.ndarray, n:float=100.0):
#     sum_term = - np.einsum('i,ki->k', z, dlnphik_xj_cons)[:, None]
#     n_dlnphik_nj = dlnphik_xj_cons + sum_term
#     dlnphik_nj = n_dlnphik_nj / n

#     # print(n_dlnphik_nj)

# coisasa para o hard chain


# # coisas para o dispersion
# def _compute_detaI1I2_dxk(eta: float, am: np.ndarray, bm: np.ndarray, am_xk: np.ndarray, bm_xk: np.ndarray, zeta3_xk: np.ndarray):
#     j = np.arange(7)[:, None]
#     detaI1_deta_xk = np.sum((j + 1) * (am_xk.T * eta**j + j * eta**(j - 1) * am[:, None] * zeta3_xk.T), axis=0)
#     detaI2_deta_xk = np.sum((j + 1) * (bm_xk.T * eta**j + j * eta**(j - 1) * bm[:, None] * zeta3_xk.T), axis=0)
   
#     return detaI1_deta_xk, detaI2_deta_xk

# def _compute_C2_xk(m: np.ndarray, m_mean: float, eta: float, zeta3_xk: np.ndarray, C1: float, C1_xk: np.ndarray):

#     u = m_mean
#     u_xk = m
    
#     o = - 4 * eta**2 + 20 * eta + 8
#     o_xk = -8 * eta * zeta3_xk + 20 * zeta3_xk
#     p = (1 - eta)**5
#     p_xk = - 5 * (1 - eta)**4 * zeta3_xk
#     v = o / p
#     v_xk = (o_xk * p - o * p_xk) / p**2


#     a = (1 - m_mean)
#     a_xk = - m
    
#     o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
#     o_xk = 6 * eta**2 * zeta3_xk + 24 * eta * zeta3_xk - 48 * zeta3_xk
#     p = (eta**2 - 3 * eta + 2)**3
#     p_xk = 3 * (2 * eta * zeta3_xk - 3 * zeta3_xk) * (eta**2 - 3 * eta + 2)**2

#     b = o / p
#     b_xk = (o_xk * p - o * p_xk) / p**2

#     s = - C1**2
#     s_xk = - 2 * C1 * C1_xk

#     t = u * v + a * b
#     t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

#     C2_xk = s_xk * t + s * t_xk
    
#     return C2_xk

# def _compute_dZdisp_dxk(rho: float, eta: float, detaI1_eta: float, detaI2_eta: float, detaI1_xk: np.ndarray, detaI2_xk: np.ndarray,
#                         m: np.ndarray, m_mean: float, C1: float, C2: float, C1_xk: np.ndarray, C2_xk: np.ndarray,
#                         m2es3: float, m2es3_xk: np.ndarray, zeta3_xk: np.ndarray, I2: float, I2_xk: np.ndarray,
#                         m2e2s3: float, m2e2s3_xk: np.ndarray):
    
#     termo_1 = - 2 * np.pi * rho * (detaI1_xk * m2es3 + detaI1_eta * m2es3_xk)

#     u = m_mean * C1
#     du = (m * C1 + m_mean * C1_xk)
#     v = detaI2_eta * m2e2s3
#     dv = (detaI2_xk * m2e2s3 + detaI2_eta * m2e2s3_xk)
#     termo_2 = - np.pi * rho * (du * v + u * dv)
#     u = m_mean * C2 * eta
#     du = m * (C2 * eta) + m_mean * (C2_xk * eta + C2 * zeta3_xk)
#     v = I2 * m2e2s3
#     dv = I2_xk * m2e2s3 + I2 * m2e2s3_xk
#     termo_3 = - np.pi * rho * (du * v + u * dv)

#     dZdips_xk = termo_1 + termo_2 + termo_3

    

    
#     return dZdips_xk



# def _compute_I1I1_xjxk(eta: float, a: np.ndarray, b: np.ndarray, ai_xk:np.ndarray, ai_xjxk: np.ndarray, 
#                        bi_xk: np.ndarray, bi_xjxk: np.ndarray, zeta3_xk: np.ndarray):
    
#     i_vec = np.arange(7)
#     i_minus1 = i_vec - 1
#     i_minus2 = i_vec - 2
#     eta_pow_i = np.power(eta, i_vec)
#     eta_pow_i_minus1 = np.power(eta, i_minus1)
#     eta_pow_i_minus2 = np.power(eta, i_minus2)
#     zeta3_xjxk = np.outer(zeta3_xk, zeta3_xk)

#     # Construção do I1_xjxk
#     term1 = np.einsum('ijk,i->jk', ai_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', ai_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', ai_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I1_xjxk = term1 + term2 + term3

#     # Construção do I12_xjxk
#     term1 = np.einsum('ijk,i->jk', bi_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', bi_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', bi_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I2_xjxk = term1 + term2 + term3

#     return I1_xjxk, I2_xjxk


# def _compute_m2es3_m2e2s3_xjxk(m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float):
#     mjmk = np.outer(m, m)
#     m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
#     m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

#     return m2es3_xjxk, m2e2s3_xjxk

# def _compute_C1_xjxk(eta: float, m:np.ndarray,  C1: float, C1_xk:np.ndarray, C2_xk: np.ndarray, zeta3_xk: np.ndarray):

#     C2_xj_zeta3_xk = np.outer(C2_xk, zeta3_xk)
#     term1 = C2_xj_zeta3_xk

#     # as funcoes de eta dentro do parenteses
#     u = 8 * eta - 2 * eta**2
#     du = 8 * zeta3_xk - 4 * eta * zeta3_xk
#     v = (1 - eta)**4
#     dv = - 4 * (1 - eta)**3 * zeta3_xk
#     s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
#     ds = 20 * zeta3_xk - 54 * eta * zeta3_xk + 36 * eta**2 * zeta3_xk - 8 * eta**3 * zeta3_xk
#     t = (2 - 3 * eta + eta**2)**2
#     dt = 2 * t**0.5 * (2 * eta * zeta3_xk - 3 * zeta3_xk)
#     aux1_xj = (du * v - u * dv) / v**2
#     aux2_xj = (ds * t - s * dt) / t**2
#     func_aux = u/v - s/t
#     term_aux_xj= 2 * C1 * C1_xk * func_aux + C1**2 * (aux1_xj - aux2_xj)
#     term2 = np.outer(term_aux_xj, m)
#     C1_xjxk = term1 - term2

#     return C1_xjxk

# def _compute_dadisp_xjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, I1_xk:np.ndarray, I2_xk:np.ndarray, 
#                          I1_xjxk:np.ndarray, I2_xjxk:np.ndarray,m2es3:float, m2e2s3:float, m2es3_xk:np.ndarray, m2e2s3_xk:np.ndarray,
#                            m2es3_xjxk:np.ndarray, m2e2s3_xjxk:np.ndarray,
#                          C1:float, C1_xk:np.ndarray, C1_xjxk:np.ndarray):
    
#     m2es3_xj_I1_xk = np.outer(m2es3_xk, I1_xk)
    
#     term1_xj = - 2 * np.pi * rho * (I1_xjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + m2es3_xjxk * I1)

#     aux = I2 * m2e2s3
#     aux_xj = (I2_xk * m2e2s3 + I2 * m2e2s3_xk)
#     aux1_xj = C1_xk * aux + C1 * aux_xj
#     aux1_xj = np.outer(aux1_xj, m)

#     u = m_mean * C1_xk
#     du = np.outer(m, C1_xk) + m_mean * C1_xjxk
#     v = aux
#     dv = aux_xj
#     aux2_xj = du * v + np.outer(dv, u)

#     s = m_mean * C1
#     ds = m * C1 + m_mean * C1_xk
#     t = I2_xk * m2e2s3
#     dt = I2_xjxk * m2e2s3 + np.outer(m2e2s3_xk, I2_xk)
#     aux_3_xj = np.outer(ds, t) + s * dt

#     m = I2 * m2e2s3_xk
#     dm = np.outer(I2_xk, m2e2s3_xk) + I2 * m2e2s3_xjxk
#     aux_4_xj = np.outer(ds, m) + s * dm

#     term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
    
#     dadisp_xjxk = term1_xj + term2_xj
#     return dadisp_xjxk


# def _compute_dlnphik_xj_unc(z:np.ndarray, Z:float, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
#     term1 = ((1 - 1 / Z) * dZ_xk)[:, None]
    
#     term3 = (- np.einsum('ji,i->j', dares_xjxk, z))[:, None]

#     dlnphik_xj_unc = term1 + dares_xjxk + term3

#     return dlnphik_xj_unc
 

# def _compute_dlnphik_xj_cons(z:np.ndarray, dlnphik_xj_unc:np.ndarray):
#     N = len(z)

#     A_ij = np.eye(N) - z[:, None]
#     M_ik = dlnphik_xj_unc
#     dlnphik_xj_cons = np.einsum('ij,ik->jk', A_ij, M_ik)
#     return dlnphik_xj_cons

# def _compute_dlnphik_nj(z:np.ndarray, dlnphik_xj_cons:np.ndarray, n:float=100.0):
#     sum_term = - np.einsum('i,ki->k', z, dlnphik_xj_cons)[:, None]
#     n_dlnphik_nj = dlnphik_xj_cons + sum_term
#     dlnphik_nj = n_dlnphik_nj / n

#     # print(n_dlnphik_nj)


# coisasa para o hard chain


# # coisas para o dispersion
# def _compute_detaI1I2_dxk(eta: float, am: np.ndarray, bm: np.ndarray, am_xk: np.ndarray, bm_xk: np.ndarray, zeta3_xk: np.ndarray):
#     j = np.arange(7)[:, None]
#     detaI1_deta_xk = np.sum((j + 1) * (am_xk.T * eta**j + j * eta**(j - 1) * am[:, None] * zeta3_xk.T), axis=0)
#     detaI2_deta_xk = np.sum((j + 1) * (bm_xk.T * eta**j + j * eta**(j - 1) * bm[:, None] * zeta3_xk.T), axis=0)
   
#     return detaI1_deta_xk, detaI2_deta_xk

# def _compute_C2_xk(m: np.ndarray, m_mean: float, eta: float, zeta3_xk: np.ndarray, C1: float, C1_xk: np.ndarray):

#     u = m_mean
#     u_xk = m
    
#     o = - 4 * eta**2 + 20 * eta + 8
#     o_xk = -8 * eta * zeta3_xk + 20 * zeta3_xk
#     p = (1 - eta)**5
#     p_xk = - 5 * (1 - eta)**4 * zeta3_xk
#     v = o / p
#     v_xk = (o_xk * p - o * p_xk) / p**2


#     a = (1 - m_mean)
#     a_xk = - m
    
#     o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
#     o_xk = 6 * eta**2 * zeta3_xk + 24 * eta * zeta3_xk - 48 * zeta3_xk
#     p = (eta**2 - 3 * eta + 2)**3
#     p_xk = 3 * (2 * eta * zeta3_xk - 3 * zeta3_xk) * (eta**2 - 3 * eta + 2)**2

#     b = o / p
#     b_xk = (o_xk * p - o * p_xk) / p**2

#     s = - C1**2
#     s_xk = - 2 * C1 * C1_xk

#     t = u * v + a * b
#     t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

#     C2_xk = s_xk * t + s * t_xk
    
#     return C2_xk

# def _compute_dZdisp_dxk(rho: float, eta: float, detaI1_eta: float, detaI2_eta: float, detaI1_xk: np.ndarray, detaI2_xk: np.ndarray,
#                         m: np.ndarray, m_mean: float, C1: float, C2: float, C1_xk: np.ndarray, C2_xk: np.ndarray,
#                         m2es3: float, m2es3_xk: np.ndarray, zeta3_xk: np.ndarray, I2: float, I2_xk: np.ndarray,
#                         m2e2s3: float, m2e2s3_xk: np.ndarray):
    
#     termo_1 = - 2 * np.pi * rho * (detaI1_xk * m2es3 + detaI1_eta * m2es3_xk)

#     u = m_mean * C1
#     du = (m * C1 + m_mean * C1_xk)
#     v = detaI2_eta * m2e2s3
#     dv = (detaI2_xk * m2e2s3 + detaI2_eta * m2e2s3_xk)
#     termo_2 = - np.pi * rho * (du * v + u * dv)
#     u = m_mean * C2 * eta
#     du = m * (C2 * eta) + m_mean * (C2_xk * eta + C2 * zeta3_xk)
#     v = I2 * m2e2s3
#     dv = I2_xk * m2e2s3 + I2 * m2e2s3_xk
#     termo_3 = - np.pi * rho * (du * v + u * dv)

#     dZdips_xk = termo_1 + termo_2 + termo_3

    

    
#     return dZdips_xk



# def _compute_I1I1_xjxk(eta: float, a: np.ndarray, b: np.ndarray, ai_xk:np.ndarray, ai_xjxk: np.ndarray, 
#                        bi_xk: np.ndarray, bi_xjxk: np.ndarray, zeta3_xk: np.ndarray):
    
#     i_vec = np.arange(7)
#     i_minus1 = i_vec - 1
#     i_minus2 = i_vec - 2
#     eta_pow_i = np.power(eta, i_vec)
#     eta_pow_i_minus1 = np.power(eta, i_minus1)
#     eta_pow_i_minus2 = np.power(eta, i_minus2)
#     zeta3_xjxk = np.outer(zeta3_xk, zeta3_xk)

#     # Construção do I1_xjxk
#     term1 = np.einsum('ijk,i->jk', ai_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', ai_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', ai_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I1_xjxk = term1 + term2 + term3

#     # Construção do I12_xjxk
#     term1 = np.einsum('ijk,i->jk', bi_xjxk, eta_pow_i)
#     aux1 = np.einsum('ji,k->ijk', bi_xk, zeta3_xk)
#     aux2 = np.einsum('ki,j->ijk', bi_xk, zeta3_xk)
#     term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
#     term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

#     I2_xjxk = term1 + term2 + term3

#     return I1_xjxk, I2_xjxk


# def _compute_m2es3_m2e2s3_xjxk(m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float):
#     mjmk = np.outer(m, m)
#     m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
#     m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

#     return m2es3_xjxk, m2e2s3_xjxk

# def _compute_C1_xjxk(eta: float, m:np.ndarray,  C1: float, C1_xk:np.ndarray, C2_xk: np.ndarray, zeta3_xk: np.ndarray):

#     C2_xj_zeta3_xk = np.outer(C2_xk, zeta3_xk)
#     term1 = C2_xj_zeta3_xk

#     # as funcoes de eta dentro do parenteses
#     u = 8 * eta - 2 * eta**2
#     du = 8 * zeta3_xk - 4 * eta * zeta3_xk
#     v = (1 - eta)**4
#     dv = - 4 * (1 - eta)**3 * zeta3_xk
#     s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
#     ds = 20 * zeta3_xk - 54 * eta * zeta3_xk + 36 * eta**2 * zeta3_xk - 8 * eta**3 * zeta3_xk
#     t = (2 - 3 * eta + eta**2)**2
#     dt = 2 * t**0.5 * (2 * eta * zeta3_xk - 3 * zeta3_xk)
#     aux1_xj = (du * v - u * dv) / v**2
#     aux2_xj = (ds * t - s * dt) / t**2
#     func_aux = u/v - s/t
#     term_aux_xj= 2 * C1 * C1_xk * func_aux + C1**2 * (aux1_xj - aux2_xj)
#     term2 = np.outer(term_aux_xj, m)
#     C1_xjxk = term1 - term2

#     return C1_xjxk

# def _compute_dadisp_xjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, I1_xk:np.ndarray, I2_xk:np.ndarray, 
#                          I1_xjxk:np.ndarray, I2_xjxk:np.ndarray,m2es3:float, m2e2s3:float, m2es3_xk:np.ndarray, m2e2s3_xk:np.ndarray,
#                            m2es3_xjxk:np.ndarray, m2e2s3_xjxk:np.ndarray,
#                          C1:float, C1_xk:np.ndarray, C1_xjxk:np.ndarray):
    
#     m2es3_xj_I1_xk = np.outer(m2es3_xk, I1_xk)
    
#     term1_xj = - 2 * np.pi * rho * (I1_xjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + m2es3_xjxk * I1)

#     aux = I2 * m2e2s3
#     aux_xj = (I2_xk * m2e2s3 + I2 * m2e2s3_xk)
#     aux1_xj = C1_xk * aux + C1 * aux_xj
#     aux1_xj = np.outer(aux1_xj, m)

#     u = m_mean * C1_xk
#     du = np.outer(m, C1_xk) + m_mean * C1_xjxk
#     v = aux
#     dv = aux_xj
#     aux2_xj = du * v + np.outer(dv, u)

#     s = m_mean * C1
#     ds = m * C1 + m_mean * C1_xk
#     t = I2_xk * m2e2s3
#     dt = I2_xjxk * m2e2s3 + np.outer(m2e2s3_xk, I2_xk)
#     aux_3_xj = np.outer(ds, t) + s * dt

#     m = I2 * m2e2s3_xk
#     dm = np.outer(I2_xk, m2e2s3_xk) + I2 * m2e2s3_xjxk
#     aux_4_xj = np.outer(ds, m) + s * dm

#     term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
    
#     dadisp_xjxk = term1_xj + term2_xj
#     return dadisp_xjxk


# def _compute_dlnphik_xj_unc(z:np.ndarray, Z:float, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
#     term1 = ((1 - 1 / Z) * dZ_xk)[:, None]
    
#     term3 = (- np.einsum('ji,i->j', dares_xjxk, z))[:, None]

#     dlnphik_xj_unc = term1 + dares_xjxk + term3

#     return dlnphik_xj_unc
 

# def _compute_dlnphik_xj_cons(z:np.ndarray, dlnphik_xj_unc:np.ndarray):
#     N = len(z)

#     A_ij = np.eye(N) - z[:, None]
#     M_ik = dlnphik_xj_unc
#     dlnphik_xj_cons = np.einsum('ij,ik->jk', A_ij, M_ik)
#     return dlnphik_xj_cons

# def _compute_dlnphik_nj(z:np.ndarray, dlnphik_xj_cons:np.ndarray, n:float=100.0):
#     sum_term = - np.einsum('i,ki->k', z, dlnphik_xj_cons)[:, None]
#     n_dlnphik_nj = dlnphik_xj_cons + sum_term
#     dlnphik_nj = n_dlnphik_nj / n

#     # print(n_dlnphik_nj)
#     # print(dlnphik_nj)

#     ni = z * n



# # ---------------- TUDO DO HARDCHAIN
# def _compute_dzhs_dxk(zeta_xk: np.ndarray, zeta: np.ndarray):
#         zeta_aux = 1 - zeta[3]

#         termo_1 = zeta_xk[3, :] / zeta_aux**2

#         u = zeta[1] * zeta[2]
#         du = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
#         v = zeta[0] * zeta_aux**2
#         dv = zeta_xk[0,:] * zeta_aux**2 - 2 * zeta[0] * zeta_aux * zeta_xk[3,:]
#         termo_2 = 3 * (du * v - dv * u) / v**2

#         u = 3 * zeta[2]**3 - zeta[3] * zeta[2]**3
#         du = 9 * zeta[2]**2 * zeta_xk[2,:] - (zeta_xk[3,:] * zeta[2]**3 + 3 * zeta[3] * zeta[2]**2 * zeta_xk[2,:])
#         v = zeta[0] * zeta_aux**3
#         dv = zeta_xk[0,:] * zeta_aux**3 - 3 * zeta[0] * zeta_aux**2 * zeta_xk[3,:]
#         termo_3 = (du * v - u * dv) / v**2

#         dzhs_dxk = termo_1 + termo_2 + termo_3 

#         return dzhs_dxk
#         # print('dzhs_dxi: ', dzhs_dxi)


# def _compute_dgij_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
#     """
#     eu acho que ja tenho..?
#     Returns: 
#         Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
#     """
#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta[3]

#     termo_1 = zeta_xk[3,:] / zeta_aux**2

#     u = 3 * zeta[2]
#     du = 3 * zeta_xk[2,:]
#     v = zeta_aux**2
#     dv = - 2 * zeta_aux * zeta_xk[3,:]
#     termo_2 = (du * v - u * dv) / v**2

#     u = 2 * zeta[2]**2
#     du = 4 * zeta[2] * zeta_xk[2,:]
#     v = zeta_aux**3
#     dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
#     termo_3 = (du * v - u * dv) / v**2

#     dgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
#     return dgij_dxk

# def _compute_drhodhji_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
#     """
    
#     Returns: 
#         Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
#     """
#     _, _, zeta2, zeta3 = zeta
#     _, _, zeta2_xk, zeta3_xk = zeta_xk

#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta3

#     termo_1 = (zeta3_xk * zeta_aux**2 - (zeta3) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4

#     termo_2 = ((3 * zeta2_xk) * zeta_aux**2 - (3 * zeta2) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4
    
#     termo_2 += ((6 * (zeta2_xk * zeta3 + zeta2 * zeta3_xk)) * zeta_aux**3 - (6 * zeta2 * zeta3) * (- 3 *zeta_aux**2 * zeta3_xk)) / zeta_aux**6

#     termo_3 = ((8 * zeta2 * zeta2_xk) * zeta_aux**3 - (4 * zeta2**2) * ((- 3 *zeta_aux**2 * zeta3_xk))) / zeta_aux**6

#     u = 6 * zeta2**2 * zeta3
#     du = 6 * (2 * zeta2 * zeta3 * zeta2_xk + zeta2**2 * zeta3_xk)
#     v = zeta_aux**4
#     dv = - 4 * zeta_aux**3 * zeta3_xk
#     termo_3 += (du * v - u * dv) / v**2

#     drhodgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
#     return drhodgij_dxk


# def _compute_dZhc_dxk(z: np.ndarray, m_mean: float, m:np.ndarray, Zhs: float, dZhs_dk: np.ndarray, gij: np.ndarray, rho_dgij_drho: np.ndarray,
#                       dgij_dxk: np.ndarray, drhodhji_dxk: np.ndarray):

#     termo_1_vec = m * Zhs + m_mean * dZhs_dk

#     # Termos do somatorio
#     gkk = np.diagonal(gij)
#     rho_dgkk_drho = np.diagonal(rho_dgij_drho)
#     sum_1_factor = (m - 1) * rho_dgkk_drho / gkk
    
#     dgii_dxk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
#     drhodhii_dxk = np.diagonal(drhodhji_dxk, axis1=0, axis2=1).T
#     gii = gkk
#     rho_dgii_drho = rho_dgkk_drho

#     factor_aux = (m - 1) * z * rho_dgii_drho / (-gii**2)
#     sum_2_factor = np.sum(factor_aux[:, None] * dgii_dxk, axis=0)

#     factor_aux = (m - 1) * z / gii
#     sum_3_factor = np.sum(factor_aux[:, None] * drhodhii_dxk, axis=0)
    
#     termo_2_vec = sum_1_factor + sum_2_factor + sum_3_factor

#     dZhc_dxk = termo_1_vec - termo_2_vec

    
#     return dZhc_dxk

# def _compute_dZhs_dT(zeta:np.ndarray, dzeta_dT:np.ndarray):
#     zeta0, zeta1, zeta2, zeta3 = zeta
#     dzeta1_dT, dzeta2_dT, dzeta3_dT = dzeta_dT
#     zeta_aux = 1 - zeta3
#     u = zeta3
#     du = dzeta3_dT
#     v = zeta_aux
#     dv = -dzeta3_dT
#     f1 = u / v
#     df1_dT = (du * v - u * dv) / v**2

#     u = 3 * zeta1 * zeta2
#     du = 3 * (dzeta1_dT * zeta2 + zeta1 * dzeta2_dT)
#     v = zeta0 * zeta_aux**2
#     dv = 0.0 - 2 * zeta0 * zeta_aux * dzeta3_dT
#     f2 = u / v
#     df2_dT = (du * v - u * dv) / v**2

#     u = 3 * zeta2**3 - zeta3 * zeta2**3
#     du = 9 * zeta2**2 * dzeta2_dT - (dzeta3_dT * zeta2**3 + 3 * zeta3 * zeta2**2 * dzeta2_dT)
#     v = zeta0 * zeta_aux**3
#     dv = 0.0 - 3 * zeta0 * zeta_aux**2 * dzeta3_dT
#     f3 = u / v
#     df3_dT = (du * v - u * dv) / v**2

#     dZhs_dT = df1_dT + df2_dT + df3_dT
#     return dZhs_dT

# def _compute_dgij_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
#     _, _, zeta2, zeta3 = zeta
#     _, dzeta2_dT, dzeta3_dT = dzeta_dT

#     zeta_aux = 1 - zeta3
    
#     df1_dT = dzeta3_dT / zeta_aux**2

#     u =  3 * zeta2
#     v = zeta_aux**2
#     f2 = u /v
#     df2_dT = ((3 * dzeta2_dT) * v - u * (- 2 * zeta_aux * dzeta3_dT)) / v**2

#     u = 2 * zeta2**2
#     v = zeta_aux**3
#     f3 = u / v
#     df3_dT = (4 * zeta2 * dzeta2_dT * v - u * (- 3 * zeta_aux**2 * dzeta3_dT)) / v**2

#     dgij_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
#     return dgij_dT

# def _compute_drho_dgij_rho_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
#     _, _, zeta2, zeta3 = zeta
#     _, dzeta2_dT, dzeta3_dT = dzeta_dT

#     zeta_aux = 1 - zeta3

#     f1 = zeta3 / zeta_aux**2
#     df1_dT = (dzeta3_dT * zeta_aux**2 - zeta3 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

#     f21 = 3 * zeta2 / zeta_aux**2
#     df21_dT = (3 * dzeta2_dT * zeta_aux**2 - 3 * zeta2 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

#     f22 = 6 * zeta2 * zeta3 / zeta_aux**3
#     df22_dT = (6 * (dzeta2_dT * zeta3 + zeta2 * dzeta3_dT) * zeta_aux**3 - 6 * zeta2 * zeta3 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6

#     f2 = f21 + f22
#     df2_dT = df21_dT + df22_dT

#     f31 = 4 * zeta2**2 / zeta_aux**3
#     df31_dT = (8 * zeta2 * dzeta2_dT * zeta_aux**3 - 4 * zeta2**2 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6
    
#     f32 = 6 * zeta2**2 * zeta3 / zeta_aux**4
#     df32_dT = (6 * (2 * zeta2 * dzeta2_dT * zeta3 + zeta2**2 * dzeta3_dT) * zeta_aux**4 - 6 * zeta2**2 * zeta3 * (-4 * zeta_aux**3 * dzeta3_dT)) / zeta_aux**8

#     f3 = f31 + f32
#     df3_dT = df31_dT + df32_dT

#     drho_dgij_rho_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
#     return drho_dgij_rho_dT

# def _compute_dZhc_dT(z:np.ndarray, m:np.ndarray, m_mean:float, dZhs_dT:float, gij:np.ndarray, rho_dgij_drho:np.ndarray,
#                         dgij_dT:np.ndarray, drho_dgij_drho_dT:np.ndarray):
#     gii = np.diagonal(gij)
#     dgii_dT = np.diagonal(dgij_dT)
#     rho_dgii_drho = np.diagonal(rho_dgij_drho)
#     drho_dgii_drho_dT = np.diagonal(drho_dgij_drho_dT)

#     term1 = m_mean * dZhs_dT

#     f1 = - gii**-2 * dgii_dT * rho_dgii_drho
#     f2 = gii**-1 * drho_dgii_drho_dT
#     term2 = - np.sum(z * (m - 1) * (f1 + f2))

#     dZhc_dT = term1 + term2
#     return dZhc_dT


# def _compute_dash_xjxk(zeta: np.ndarray, zeta_xk: np.ndarray, ahs: float, dahs_xk: np.ndarray):
#     zeta_aux = 1 - zeta[3]
#     termo_1 = np.outer(zeta_xk[0,:], zeta_xk[0,:]) * ahs / zeta[0]**2 - np.outer(dahs_xk, zeta_xk[0,:]) / zeta[0]

#     # T1
#     u = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
#     du = np.outer(zeta_xk[2,:], zeta_xk[1,:]) + np.outer(zeta_xk[1,:], zeta_xk[2,:])
#     v = zeta_aux
#     dv = - zeta_xk[3,:]
#     T1 = 3 * u / v
#     T1_xj = 3 * (du * v - np.outer(dv, u)) / v**2

#     # T2
#     u = 3 * zeta[1] * zeta[2] * zeta_xk[3,:]
#     v = zeta_aux**2
#     du = 3 * (np.outer(zeta_xk[1,:], zeta_xk[3,:]) * zeta[2] + np.outer(zeta_xk[2,:],zeta_xk[3,:]) * zeta[1])
#     dv = - 2 * zeta_aux * zeta_xk[3,:]
#     T2 = u / v
#     T2_xj = (du * v - np.outer(dv, u)) / v**2

#     # T3
#     u = 3 * zeta[2]**2 * zeta_xk[2,:]
#     du = 6 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[2,:])
#     v = zeta[3] * zeta_aux**2
#     dv = zeta_xk[3,:] * zeta_aux**2 - 2 * zeta[3] * zeta_aux * zeta_xk[3,:]
#     T3 = u / v
#     T3_xj = (du * v - np.outer(dv, u)) / v**2

#     # T4
#     u = zeta[2]**3 * zeta_xk[3,:] * (3 * zeta[3] - 1)
#     du = 3 * zeta[2]**2 * (3 * zeta[3] - 1) * np.outer(zeta_xk[2,:], zeta_xk[3,:]) + 3 * zeta[2]**3 * np.outer(zeta_xk[3,:], zeta_xk[3,:])
#     v = zeta[3]**2 * zeta_aux**3
#     dv = 2 * zeta[3] * zeta_aux**3 * zeta_xk[3,:] - 3 * zeta[3]**2 * zeta_aux**2 * zeta_xk[3,:]
#     T4 = u / v
#     T4_xj = (du * v - np.outer(dv, u)) / v**2

#     # T5
#     u = 3 * zeta[2]**2 * zeta[3] * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta_xk[3,:]
#     v = zeta[3]**3
#     du = 6 * zeta[2] * zeta[3] * np.outer(zeta_xk[2,:], zeta_xk[2,:]) + 3 * zeta[2]**2 * np.outer(zeta_xk[3,:], zeta_xk[2,:]) - 6 * zeta[2]**2 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     dv = 3 * zeta[3]**2 * zeta_xk[3,:]

#     T5_1 = (u / v) - zeta_xk[0,:]
#     T5_1xj = (du * v - np.outer(dv, u)) / v**2
#     T5_2 = np.log(zeta_aux)
#     T5_2xj = - zeta_xk[3,:] / zeta_aux
#     T5 = T5_1 * T5_2
#     T5_xj = T5_1xj * T5_2 + np.outer(T5_2xj, T5_1)

#     # T6
#     T6_1 = zeta[0] - zeta[2]**3 / zeta[3]**2
#     T6_1xj = zeta_xk[0,:] - (3 * zeta[2]**2 * zeta[3]**2 * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta[3] * zeta_xk[3,:]) / zeta[3]**4
#     T6_2 = zeta_xk[3,:] / zeta_aux
#     T6_2xj = np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**2
#     T6 = T6_1 * T6_2
#     T6_xj = np.outer(T6_1xj, T6_2) + T6_1 * T6_2xj

#     T = T1 + T2 + T3 + T4 + T5 + T6
#     T_xj = T1_xj + T2_xj + T3_xj + T4_xj + T5_xj + T6_xj
   
#     termo_2 = - np.outer(zeta_xk[0,:], T) / zeta[0]**2 + T_xj / zeta[0]

#     dahs_xjxk = termo_1 + termo_2
    
#     return dahs_xjxk


# def _compute_dgij_xjxk(d: np.ndarray, zeta: np.ndarray, zeta_xk: np.ndarray):
#     Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
#     zeta_aux = 1 - zeta[3]
#     termo_1 = 2 * np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**3


#     termo_21 = (6 / zeta_aux**3) * np.outer(zeta_xk[3,:], zeta_xk[2,:])

#     u = 6 * zeta[2] * zeta_xk[3,:]
#     du = 6 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     v = zeta_aux**3
#     dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
#     termo_22 = (du * v - np.outer(dv, u)) / v**2

#     termo_2 = termo_21 + termo_22

#     u = 4 * zeta[2] * zeta_xk[2,:]
#     du = 4 * np.outer(zeta_xk[2,:], zeta_xk[2,:])
#     termo_31 = (du * v - np.outer(dv, u)) / v**2
    
#     u = 6 * zeta[2]**2 * zeta_xk[3,:]
#     du = 12 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[3,:])
#     v = zeta_aux**4
#     dv = - 4 * zeta_aux**3 *zeta_xk[3,:]
#     termo_32 = (du * v - np.outer(dv, u)) / v**2

#     termo_3 = termo_31 + termo_32
#     dgij_xjxk = termo_1[None, None, :, :] + Dij[:, :, None, None] * termo_2[None, None, :, :] + Dij[:, :, None, None]**2 * termo_3[None, None, :, :]
#     # AQUI TEM QUE VER MESMO SE GERA UM TENSOR (n,n,n,n)
#     return dgij_xjxk

# def _dahc_xjxk(z: np.ndarray, m: np.ndarray, m_mean: float, gij: np.ndarray, dgij_xk: np.ndarray, dgij_xjxk: np.ndarray,
#                dahs_xk: np.ndarray, dahs_xjxk: np.ndarray):
#     gii = np.diagonal(gij)
#     gii_inv = 1 / gii
#     gii_inv_sq = gii_inv**2

#     dgii_xk = np.diagonal(dgij_xk, axis1=0, axis2=1).T
#     # dgii_xjxk = np.einsum('iikj->jk', dgij_xjxk)

#     # Termo 1: mₖ (∂ãʰˢ/∂xⱼ) -> Matriz [j, k] = mₖ * (∂ãʰˢ/∂xⱼ)
#     term1 = np.outer(dahs_xk, m)

#     # Termo 2: mₖ (∂ãʰˢ/∂xₖ) -> Matriz [k, j] = mⱼ * (∂ãʰˢ/∂xₖ)
#     term2 = np.outer(m, dahs_xk)

#     # Termo 3: m̄ (∂²ãʰˢ/∂xⱼ∂xₖ) -> Escalar * Matriz
#     term3 = m_mean * dahs_xjxk

#     # Termo 4a: - (mⱼ-1)(gⱼⱼ)⁻¹ (∂gⱼⱼ/∂xₖ) -> Matriz [j, k]
#     term4a = - (m - 1.0)[:, None] * gii_inv[:, None] * dgii_xk

#     # Termo 4b: + Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻² (∂gᵢᵢ/∂xⱼ) (∂gᵢᵢ/∂xₖ) -> Matriz [j, k]
#     sum_4b = z * (m - 1.0) * gii_inv_sq # Vetor (N,)
#     term4b = np.einsum('i,ij,ik->jk', sum_4b, dgii_xk, dgii_xk)

#     # Termo 4c: - Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻¹ (∂²gᵢᵢ/∂xⱼ∂xₖ) -> Matriz [j, k]
#     dgii_xjxk = np.diagonal(dgij_xjxk, axis1=0, axis2=1).transpose(2, 0, 1)
#     sum_4c = z * (m - 1.0) * gii_inv # Vetor (N,)
#     term4c = -np.einsum('i,ijk->jk', sum_4c, dgii_xjxk)

#     # Termo 5: - (mₖ-1)(gₖₖ)⁻¹ (∂gₖₖ/∂xⱼ) -> Matriz [j, k]
#     # dgii_dxk.T[k, j] = ∂gₖₖ/∂xⱼ
#     term5 = - (m - 1.0)[None, :] * gii_inv[None, :] * dgii_xk.T

#     # 3. Soma Final -> Matriz NxN
#     dahc_xjxk = term1 + term2 + term3 + term4a + term4b + term4c + term5
#     return dahc_xjxk

# from time import time
# from copy import deepcopy
# if __name__ == '__main__':
#     T_0 = time()
#     T = 200 # K
#     P = 30e5 # Pa
#     # T = 350 # K
#     # P = 9.4573e5 # Pa
#     butano = Component(
#         name='Butano',
#         Tc=None,
#         Pc=None,
#         omega=None,
#         sigma=3.7086,
#         epsilon=222.88,
#         segment=2.3316
#     )

#     nitrogenio = Component(
#         name='N2',
#         Tc=None,
#         Pc=None,
#         omega=None,
#         sigma=3.3130,
#         epsilon=90.96,
#         segment=1.2053
#     )

#     metano = Component(
#         name='CH4',
#         Tc=None,
#         Pc=None,
#         omega=None,
#         sigma=3.7039,
#         epsilon=150.03,
#         segment=1.000
#     )

#     mixture = Mixture(
#         components=[nitrogenio, metano],
#         k_ij=0.0,
#         l_ij=0.0
#         )
    
#     state_trial = State(
#         mixture=mixture,
#         z=np.array([0.4, 0.6]),
#         T=T,
#         P=P
#     )


#     # seta a calc
#     pc_saft_engine = PCSaft(workers=None)
#     pc_saft_engine.calculate_from_TP(state=state_trial, is_vapor=True)
#     pc_saft_engine.calculate_fugacity(state=state_trial)
#     t_F = time()
#     # print(t_F - T_0)
    

#     # Testando numericamente
#     # print(160*'*')
#     # print('----Comparação numérica com analítica----')
#     h_x = 0.00001
#     t_num_0 = time()
#     z_pos = np.array([state_trial.z[0] + h_x, state_trial.z[1]])
#     state_xpos = deepcopy(state_trial)
#     state_xpos.z = z_pos
#     pc_saft_engine.update_parameters(state=state_xpos)
#     pc_saft_engine.calculate_fugacity(state=state_xpos)


#     z_neg = np.array([state_trial.z[0] - h_x, state_trial.z[1]])
#     state_xneg = deepcopy(state_trial)
#     state_xneg.z = z_neg
#     pc_saft_engine.update_parameters(state=state_xneg)
#     pc_saft_engine.calculate_fugacity(state=state_xneg)

#     # dxhs_dxi
    
#     # Calculos numericos (estimar tempo)
    
#     dzhs_dxk_num = (state_xpos.core_model.hc_results.Z_hs - state_xneg.core_model.hc_results.Z_hs) / (2 * h_x)
#     dgij_dxk_num = (state_xpos.core_model.hc_results.gij_hs - state_xneg.core_model.hc_results.gij_hs) / (2 * h_x)
#     drhodhji_dxk_num = (state_xpos.core_model.hc_results.rho_dgij_drho - state_xneg.core_model.hc_results.rho_dgij_drho) / (2 * h_x)
#     dZhc_dxk_num = (state_xpos.core_model.hc_results.Z_hc - state_xneg.core_model.hc_results.Z_hc) / (2 * h_x)
#     detaI1_deta_xk_num = (state_xpos.core_model.disp_results.detaI1_eta - state_xneg.core_model.disp_results.detaI1_eta) / (2 * h_x)
#     detaI2_deta_xk_num = (state_xpos.core_model.disp_results.detaI2_eta - state_xneg.core_model.disp_results.detaI2_eta) / (2 * h_x)
#     C2_xk_num = (state_xpos.core_model.disp_results.C2 - state_xneg.core_model.disp_results.C2) / (2 * h_x)
#     dZdisp_xk_num = (state_xpos.core_model.disp_results.Z_disp - state_xneg.core_model.disp_results.Z_disp) / (2 * h_x)
#     dZ_xk_num = (state_xpos.Z - state_xneg.Z) / (2 * h_x)



   
    

    
#     # Calculos analiticos (estimar tempo)
#     zeta_xk = state_trial.core_model.params.zeta_xk
#     zeta = state_trial.core_model.params.zeta
#     z = state_trial.z
#     d = state_trial.core_model.params.d
#     m_mean = state_trial.core_model.params.m_mean
#     m = state_trial.core_model.params.m
#     Zhs = state_trial.core_model.hc_results.Z_hs
#     eta=state_trial.eta
#     am=state_trial.core_model.coeff.am
#     bm=state_trial.core_model.coeff.bm
#     am_xk=state_trial.core_model.coeff.ai_xk
#     bm_xk=state_trial.core_model.coeff.bi_xk
#     gij = state_trial.core_model.hc_results.gij_hs
#     dahs_xk = state_trial.core_model.hc_results.derivatives.dahs_dxk

#     ai_xjxk = state_trial.core_model.coeff.ai_xjxk
#     bi_xjxk = state_trial.core_model.coeff.bi_xjxk


#     detaI1_eta=state_trial.core_model.disp_results.detaI1_eta
#     detaI2_eta=state_trial.core_model.disp_results.detaI2_eta

#     m2es3=state_trial.core_model.disp_results.m2es3
#     m2es3_xk=state_trial.core_model.disp_results.derivatives.dm2es3_dxk

#     C1=state_trial.core_model.disp_results.C1
#     C1_xk=state_trial.core_model.disp_results.derivatives.dC1_dxk
#     C2 = state_trial.core_model.disp_results.C2
#     I1 = state_trial.core_model.disp_results.I1 
#     I2 = state_trial.core_model.disp_results.I2 
#     I2_xk = state_trial.core_model.disp_results.derivatives.dI2_dxk

#     m2e2s3=state_trial.core_model.disp_results.m2e2s3
#     m2e2s3_xk=state_trial.core_model.disp_results.derivatives.dm2e2s3_dxk

#     dzhs_dxk_anal = _compute_dzhs_dxk(zeta_xk=zeta_xk,
#                       zeta=zeta)
#     dgij_dxk_anal = _compute_dgij_dxk(d=d,
#                           zeta_xk=zeta_xk,
#                       zeta=zeta)
#     drhodhji_dxk_anal = _compute_drhodhji_dxk(d=d,
#                           zeta_xk=zeta_xk,
#                       zeta=zeta)
#     dZhc_dxk_anal = _compute_dZhc_dxk(z=state_trial.z,
#                                 m_mean=m_mean,
#                                 m=m,
#                                 Zhs=Zhs,
#                                 dZhs_dk=dzhs_dxk_anal,
#                                 gij=state_trial.core_model.hc_results.gij_hs,
#                                 rho_dgij_drho=state_trial.core_model.hc_results.rho_dgij_drho,
#                                 dgij_dxk=dgij_dxk_anal,
#                                 drhodhji_dxk=drhodhji_dxk_anal)
#     detaI1_deta_xk_anal, detaI2_deta_xk_anal = _compute_detaI1I2_dxk(eta=eta,
#                                                                      am=am,
#                                                                      bm=bm,
#                                                                      am_xk=am_xk,
#                                                                      bm_xk=bm_xk,
#                                                                      zeta3_xk=zeta_xk[3,:])
    
#     C2_xk_anal = _compute_C2_xk(m=m,
#                                 m_mean=m_mean,
#                                 eta=state_trial.eta,
#                                 zeta3_xk=zeta_xk[3,:],
#                                 C1=C1,
#                                 C1_xk=C1_xk,
#                                 )
#     t_anal_f = time()
#     dZdisp_xk_anal = _compute_dZdisp_dxk(rho=state_trial.rho,
#                                          eta=state_trial.eta,
#                                          detaI1_eta=detaI1_eta,
#                                          detaI2_eta=detaI2_eta,
#                                          detaI1_xk=detaI1_deta_xk_anal,
#                                          detaI2_xk=detaI2_deta_xk_anal,
#                                          m2es3=m2es3,
#                                          m2es3_xk=m2es3_xk,
#                                          m=m,
#                                          m_mean=m_mean,
#                                          C1=C1,
#                                          C1_xk=C1_xk,
#                                          C2=C2,
#                                          C2_xk=C2_xk_anal,
#                                          zeta3_xk=zeta_xk[3,:],
#                                          I2=I2,
#                                          I2_xk=I2_xk,
#                                          m2e2s3=m2e2s3,
#                                          m2e2s3_xk=m2e2s3_xk
#                                          )
#     dZ_xk_anal = dZdisp_xk_anal + dZhc_dxk_anal

#     # print("tempo analitico: ", (t_anal_f - t_anal_0))
#     # print("tempo numerico: ", (t_num_f - t_num_0))
#     # print('--- dzhs_dxk ---')
#     # print('dzhs_dxi_num: ', dzhs_dxk_num)
#     # print('dzhs_dxi_anal: ', dzhs_dxk_anal[0])
#     # print('--- dgij_dxk ---')
#     # print('dgij_dxk_num: ', dgij_dxk_num)
#     # print('dgij_dxk_anal: ', dgij_dxk_anal[:, :, 0])
#     # print('--- drhodhji_dxk ---')
#     # print('drhodhji_dxk_num: ', drhodhji_dxk_num)
#     # print('drhodhji_dxk_anal: ', drhodhji_dxk_anal[:, :, 0])
#     # print('--- dZhc_dxk ---')
#     # print('dZhc_dxk_num: ', dZhc_dxk_num)
#     # print('dZhc_dxk_anal: ', dZhc_dxk_anal[0])
#     # print('--- detaI1_deta_xk ---')
#     # print('detaI1_deta_xk_num: ', detaI1_deta_xk_num)
#     # print('detaI1_deta_xk_anal: ', detaI1_deta_xk_anal[0])
#     # print('--- detaI2_deta_xk ---')
#     # print('detaI2_deta_xk_num: ', detaI2_deta_xk_num)
#     # print('detaI2_deta_xk_anal: ', detaI2_deta_xk_anal[0])
#     # print('--- C2_xk ---')
#     # print('C2_xk_num: ', C2_xk_num)
#     # print('C2_xk_anal: ', C2_xk_anal[0])
#     # print('--- dZdisp_xk ---')
#     # print('dZdisp_xk_num: ', dZdisp_xk_num)
#     # print('dZdisp_xk_anal: ', dZdisp_xk_anal[0])
#     # print('--- dZ_xk ---')
#     # print('dZ_xk_num: ', dZ_xk_num)
#     # print('dZ_xk_anal: ', dZ_xk_anal)

#     # teste para a segundas derivadas de helmhotz
#     a = time()
#     h = 0.00001
    
#     j = 1
#     k = 0
#     state_j_pos = deepcopy(state_trial)
#     state_j_pos.z[j] += h
    
#     pc_saft_engine.update_parameters(state=state_j_pos, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_j_pos, teste=True)

#     state_j_neg = deepcopy(state_trial)
#     state_j_neg.z[j] -= h
#     pc_saft_engine.update_parameters(state=state_j_neg, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_j_neg, teste=True)


#     dahs_xk_pos = state_j_pos.core_model.hc_results.derivatives.dahs_dxk
#     dahs_xk_neg = state_j_neg.core_model.hc_results.derivatives.dahs_dxk
#     dahs_xjxk_num = (dahs_xk_pos[0] - dahs_xk_neg[0]) / (2 * h)


#     dgij_xk_pos = state_j_pos.core_model.hc_results.derivatives.dgij_dxk
#     dgij_xk_neg = state_j_neg.core_model.hc_results.derivatives.dgij_dxk
#     dgij_xjxk_num = (dgij_xk_pos[0,1] - dgij_xk_neg[0,1]) / (2 * h)

#     dahc_xk_pos = state_j_pos.core_model.hc_results.derivatives.dahc_dxk
#     dahc_xk_neg = state_j_neg.core_model.hc_results.derivatives.dahc_dxk
#     dahc_xjxk_num = (dahc_xk_pos - dahc_xk_neg) / (2 * h)

#     ai_xjxk_pos = state_j_pos.core_model.coeff.ai_xk
#     ai_xjxk_neg = state_j_neg.core_model.coeff.ai_xk
#     ai_xjxk_num = (ai_xjxk_pos - ai_xjxk_neg) / (2 * h)

#     bi_xjxk_pos = state_j_pos.core_model.coeff.bi_xk
#     bi_xjxk_neg = state_j_neg.core_model.coeff.bi_xk
#     bi_xjxk_num = (bi_xjxk_pos - bi_xjxk_neg) / (2 * h)

#     I1_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dI1_dxk
#     I1_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dI1_dxk
#     I1_xjxk_num = (I1_xjxk_pos - I1_xjxk_neg) / (2 * h)

#     I2_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dI2_dxk
#     I2_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dI2_dxk
#     I2_xjxk_num = (I2_xjxk_pos - I2_xjxk_neg) / (2 * h)

#     m2es3_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dm2es3_dxk
#     m2es3_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dm2es3_dxk
#     m2es3_xjxk_num = (m2es3_xjxk_pos - m2es3_xjxk_neg) / (2 * h)

#     m2e2s3_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dm2e2s3_dxk
#     m2e2s3_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dm2e2s3_dxk
#     m2e2s3_xjxk_num = (m2e2s3_xjxk_pos - m2e2s3_xjxk_neg) / (2 * h)

#     C1_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dC1_dxk
#     C1_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dC1_dxk
#     C1_xjxk_num = (C1_xjxk_pos - C1_xjxk_neg) / (2 * h)


#     dadisp_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dadisp_dxk
#     dadisp_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dadisp_dxk
#     dadisp_xjxk_num = (dadisp_xjxk_pos - dadisp_xjxk_neg) / (2 * h)

#     # dares_xjxk_pos = state_j_pos.fugacity_result.deletar_depois
#     # dares_xjxk_neg = state_j_neg.fugacity_result.deletar_depois
#     # dares_xjxk_num = (dares_xjxk_pos - dares_xjxk_neg) / (2 * h)

#     dlnphik_xj_pos = state_j_pos.fugacity_result.ln_phi
#     dlnphik_xj_neg = state_j_neg.fugacity_result.ln_phi
#     dlnphik_xj_num = (dlnphik_xj_pos - dlnphik_xj_neg) / (2 * h)


#     # dmuk_xj_pos = state_j_pos.fugacity_result.mu
#     # dmuk_xj_neg = state_j_neg.fugacity_result.mu
#     # dmuk_xj_num = (dmuk_xj_pos - dmuk_xj_neg) / (2 * h)

#     dash_xjxk_anal = _compute_dash_xjxk(zeta=zeta,
#                                         zeta_xk=zeta_xk,
#                                         ahs=state_trial.core_model.hc_results.ar_hs,
#                                         dahs_xk=state_trial.core_model.hc_results.derivatives.dahs_dxk)


#     dgij_xjxk_anal = _compute_dgij_xjxk(d=d,
#                                    zeta=zeta,
#                                    zeta_xk=zeta_xk
#                                    )

#     dahc_xjxk_anal = _dahc_xjxk(z=z, m=m, m_mean=m_mean, gij=gij, dgij_xk=dgij_dxk_anal, dgij_xjxk=dgij_xjxk_anal,
#                           dahs_xk=dahs_xk, dahs_xjxk=dash_xjxk_anal)
    
#     I1_xjxk, I2_xjxk = _compute_I1I1_xjxk(eta=eta, a=am, b=bm, ai_xk=am_xk, bi_xk=bm_xk, ai_xjxk=ai_xjxk, bi_xjxk=bi_xjxk, zeta3_xk=zeta_xk[3,:])
    
#     I1_xk = state_trial.core_model.disp_results.derivatives.dI1_dxk
#     I2_xk = state_trial.core_model.disp_results.derivatives.dI2_dxk

   
    
#     eij = state_trial.core_model.params.eij
#     sij = state_trial.core_model.params.sij
#     T = state_trial.T
#     m2es3_xjxk, m2e2s3_xjxk = _compute_m2es3_m2e2s3_xjxk(m=m, eij=eij, sij=sij, T=T)
    
#     C1_xjxk = _compute_C1_xjxk(eta=eta, m=m, C1=C1, C1_xk=C1_xk, C2_xk=C2_xk_anal, zeta3_xk=zeta_xk[3,:])


#     dadisp_xjxk = _compute_dadisp_xjxk(rho=state_trial.rho, m=m, m_mean=m_mean, I1=I1, I2=I2, I1_xk=I1_xk, I2_xk=I2_xk, 
#                          I1_xjxk=I1_xjxk, I2_xjxk=I2_xjxk, m2es3=m2es3, m2e2s3=m2e2s3, m2es3_xk=m2es3_xk, m2e2s3_xk=m2e2s3_xk,
#                            m2es3_xjxk=m2es3_xjxk, m2e2s3_xjxk=m2e2s3_xjxk,
#                          C1=C1, C1_xk=C1_xk, C1_xjxk=C1_xjxk)

#     dares_xjxk = dahc_xjxk_anal + dadisp_xjxk
#     print(80*'*')
#     print('dahc_xjxk_anal=',dahc_xjxk_anal)
#     print('dadisp_xjxk=',dadisp_xjxk)
#     print(80*'*')



#     # Quero obter o del rho la

#     h = 0.000001
    
#     state_rho_pos = deepcopy(state_trial)
#     state_rho_pos.rho += h  
#     pc_saft_engine.update_parameters_rho(state=state_rho_pos, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_rho_pos, teste=True)


#     state_rho_neg = deepcopy(state_trial)
#     state_rho_neg.rho -= h  
#     pc_saft_engine.update_parameters_rho(state=state_rho_neg, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_rho_neg, teste=True)

#     # dares_rhoxk_pos =  state_rho_pos.fugacity_result.deletar_depois
#     # dares_rhoxk_neg =  state_rho_neg.fugacity_result.deletar_depois
#     # dares_rhoxk_num = (dares_rhoxk_pos - dares_rhoxk_neg) / (2 * h)

#     Z_rho_pos =  state_rho_pos.pressure_result.Z
#     Z_rho_neg =  state_rho_neg.pressure_result.Z
#     Z_rho_num = (Z_rho_pos - Z_rho_neg) / (2 * h)

    
#     Zhc_rho_pos =  state_rho_pos.core_model.hc_results.Z_hc
#     Zhc_rho_neg =  state_rho_neg.core_model.hc_results.Z_hc
#     Zhc_rho_num = (Zhc_rho_pos - Zhc_rho_neg) / (2 * h)

#     Zdisp_rho_pos =  state_rho_pos.core_model.disp_results.Z_disp
#     Zdisp_rho_neg =  state_rho_neg.core_model.disp_results.Z_disp
#     Zdisp_rho_num = (Zdisp_rho_pos - Zdisp_rho_neg) / (2 * h)


#     # muk_rho_pos =  state_rho_pos.fugacity_result.mu
#     # muk_rho_neg =  state_rho_neg.fugacity_result.mu
#     # muk_rho_num = (muk_rho_pos - muk_rho_neg) / (2 * h)

#     # x = np.sum(state_trial.z * dZ_xk_anal)
#     # print(state_trial.rho * (1 - 1 / state_trial.Z)*Z_rho_num + (state_trial.Z - 1) + dZ_xk_anal - x)
#     # print(state_trial.rho * ln_phi_rho_num)

#     z = state_trial.z
#     Z = state_trial.Z

#     dlnphik_xj_unc = _compute_dlnphik_xj_unc(z=z, Z=Z, dZ_xk=dZ_xk_anal, dares_xjxk=dares_xjxk)

#     dlnphik_xj_cons = _compute_dlnphik_xj_cons(z=z, dlnphik_xj_unc=dlnphik_xj_unc)
#     _compute_dlnphik_nj(z=z, dlnphik_xj_cons=dlnphik_xj_cons)

   
#     def _compute_dF_dV_Tn(rho_mol:float, Z:float):
#         dF_dV_Tn = - rho_mol * (Z - 1)
#         return dF_dV_Tn
    
#     rho_mol = (state_trial.rho * (1e10)**3 / NAVOGRADO)
#     V = 100 / rho_mol
#     Z = state_trial.Z


#     def _compute_dF_dVV_Tn(rho_mol:float, Z:float, eta:float, dZ_eta:float, n:float=100.0):
#         dF_dVV = rho_mol**2 * (eta * dZ_eta + (Z - 1)) / n
#         return dF_dVV


#     eta = state_trial.eta
#     dZ_eta = state_trial.pressure_result.dZ_eta

#     z = state_trial.z
#     dZ_xk = dZ_xk_anal
#     def _compute_dF_dVnk_Tn(rho_mol:float, z:np.ndarray, Z:float, eta:float, dZ_eta:float, dZ_xk:np.ndarray, n:float=100.0):
#         print(50*'-')
#         sum_xidZxi = - np.sum(z * dZ_xk)
#         print(sum_xidZxi)
#         print(dZ_eta, dZ_xk, Z, eta)
#         dF_dVnk = - rho_mol * ((Z-1) + eta * dZ_eta + dZ_xk + sum_xidZxi) / n
#         print(50*'-')
#         return dF_dVnk

#     dF_dVnk = _compute_dF_dVnk_Tn(rho_mol=rho_mol, z=z, Z=Z, eta=eta, dZ_eta=dZ_eta, dZ_xk=dZ_xk) 
#     print('dF_dVnk = ', dF_dVnk)

#     def _compute_rho_dmuk_rho(z:np.ndarray, Z:float, eta:float, dZ_eta:float, dZ_xk:np.ndarray):
#         sum_xi_dZ_xi = - np.sum(z * dZ_xk)
#         rho_dmuk_rho = (Z - 1) + eta * dZ_eta + dZ_xk + sum_xi_dZ_xi
#         return rho_dmuk_rho
   
#     def _compute_dmuk_xj(z:np.ndarray, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
#         print('DENTRO DA DMUK XJ')
#         print(dares_xjxk)
#         sum_xi_dares_xjxi = - np.einsum('i,ji->j', z, dares_xjxk)
#         dmuk_xj = dZ_xk[:,None] + dares_xjxk + sum_xi_dares_xjxi[:,None]
#         return dmuk_xj
    
#     def _compute_dF_njnk(z:np.ndarray, rho_dmuk_rho:np.ndarray, dmuk_xj:np.ndarray, n:float=100.00):
#         sum_xi_dmuk_xi = - np.einsum('i,ik->k', z, dmuk_xj)

#         dF_njnk = (1.0 / n) * (rho_dmuk_rho + dmuk_xj + sum_xi_dmuk_xi)
#         return dF_njnk

    
#     rho_dmuk_rho = _compute_rho_dmuk_rho(z=z, Z=Z, eta=eta, dZ_eta=dZ_eta, dZ_xk=dZ_xk)
#     dF_dV = _compute_dF_dV_Tn(rho_mol=rho_mol, Z=Z)
#     dF_dVV = _compute_dF_dVV_Tn(rho_mol=rho_mol, Z=Z, eta=eta, dZ_eta=dZ_eta)

#     dmuk_xj = _compute_dmuk_xj(z=z, dZ_xk=dZ_xk, dares_xjxk=dares_xjxk)
#     dF_njnk = _compute_dF_njnk(z=z, rho_dmuk_rho=rho_dmuk_rho, dmuk_xj=dmuk_xj)

#     def _teste_final(dF_njnk:np.ndarray, dF_dV: float, dF_dVV: float, dF_dVnk:np.ndarray, V:float, T:float, n:float=100.00):
#         dP_dV = - RGAS_SI * T *dF_dVV - n * RGAS_SI * T / V**2
#         dP_dnk = - RGAS_SI * T  * dF_dVnk + RGAS_SI * T / V
#         dP_dnk_dP_dnj = np.outer(dP_dnk, dP_dnk)

#         n_dlnphik_dnj = n * dF_njnk + 1  + (n / (RGAS_SI * T)) * dP_dnk_dP_dnj / dP_dV
        
#         return n_dlnphik_dnj, n_dlnphik_dnj / n  

#     print(50*'-')
#     print('rho_dmuk_rho=',rho_dmuk_rho)
#     print('dmuk_xj=',dmuk_xj)
#     print('dF_njnk=',dF_njnk)
#     print(50*'-')
#     # 
#     # print(30*'#', ' Obtencao do dZ/dT ', 30*'#')
#     s = state_trial.core_model.params.sij
#     e = state_trial.core_model.params.eij
#     T = state_trial.T
#     rho_dgij_drho = state_trial.core_model.hc_results.rho_dgij_drho


#     ddi_dT = state_trial.core_model.params.ddi_dT
#     dzeta_dT = state_trial.core_model.params.dzeta_dT

#     # dDij_dT
#     u = np.outer(d, d)
#     du = np.outer(ddi_dT, d) + np.outer(d, ddi_dT)
#     v = d[:, None] + d
#     dv = ddi_dT[:, None] + ddi_dT
#     dDij_dT = (du * v - u * dv) / v**2
#     Dij = u / v
    


        

#     def _compute_ddetaI12_deta_dT(dzeta3_dT:float, eta:float, a:np.ndarray, b:np.ndarray):
#         j = np.arange(7)
#         ddetaI1_deta_dT = np.sum(a * (j + 1) * j * dzeta3_dT * eta**(j - 1))
#         ddetaI2_deta_dT = np.sum(b * (j + 1) * j * dzeta3_dT * eta**(j - 1))

        
#         return ddetaI1_deta_dT, ddetaI2_deta_dT

#     def _compute_dI12_dT(a:np.ndarray, b:np.ndarray, dzeta3_dT:float, eta:float):
#         j = np.arange(7)
#         dI1_dT = np.sum(a * j * dzeta3_dT * eta**(j - 1))
#         dI2_dT = np.sum(b * j * dzeta3_dT * eta**(j - 1))

#         return dI1_dT, dI2_dT

#     def _compute_dC2_dT(eta:float, dzeta3_dT:float, C1:float, C2:float, m_mean:float):
        
#         dC1_dT = dzeta3_dT * C2

#         u = - 4 * eta**2 + 20 * eta + 8
#         du = - 8 * eta * dzeta3_dT + 20 * dzeta3_dT
#         v = (1 - eta)**5
#         dv = - 5 * (1 - eta)**4 * dzeta3_dT
#         f1 = u / v
#         df1_dT = (du * v - u * dv) / v**2

#         u = 2 * eta**3 + 12 * eta**2 - 48 * eta + 40
#         du = 6 * eta**2 * dzeta3_dT + 24 * eta * dzeta3_dT - 48 * dzeta3_dT
#         v = (2 - 3 * eta + eta**2)**3
#         dv = 3 * (2 - 3 * eta + eta**2)**2 * (- 3 * dzeta3_dT + 2 * eta * dzeta3_dT)
#         f2 = u / v
#         df2_dT = (du * v - u * dv) / v**2

#         f = m_mean * f1 + (1 - m_mean) * f2
#         df_dT = m_mean * df1_dT + (1 - m_mean) * df2_dT

#         dC2_dT = - 2 * C1 * dC1_dT * f - C1**2 * df_dT

#         return dC2_dT


#     def _compute_dZdisp_dT(rho:float, eta:float, detaI1_deta:float, detaI2_deta:float, ddetaI1_deta_dT:float, ddetaI2_deta_dT:float,
#                            I2:float, C1:float, C2:float, dC2_dT:float, m2es3:float, m2e2s3:float, m_mean:float, T:float, dzeta3_dT:float,
#                            dI2_dT:float):
        
#         dC1_dT = dzeta3_dT * C2

#         term1 = - 2 * np.pi * rho * (ddetaI1_deta_dT - detaI1_deta / T) * m2es3

#         f1 = C1 * detaI2_deta
#         df1_dT = dC1_dT * detaI2_deta + C1 * ddetaI2_deta_dT
#         f2 = C2 * eta * I2
#         df2_dT = (dC2_dT * eta + C2 * dzeta3_dT) * I2 + C2 * eta * dI2_dT

#         f = f1 + f2
#         df_dT = df1_dT + df2_dT

#         term2 = - rho * np.pi * m_mean * (df_dT - 2 * f / T) * m2e2s3

#         dZdisp_dT = term1 + term2
#         return dZdisp_dT
        


#     # A perturbação na temperatura 
#     h = 0.0001

#     state_T_pos = deepcopy(state_trial)
#     state_T_pos.T += h
#     pc_saft_engine.update_parameters(state=state_T_pos, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_T_pos, teste=True)

#     state_T_neg = deepcopy(state_trial)
#     state_T_neg.T -= h
#     pc_saft_engine.update_parameters(state=state_T_neg, teste=True)
#     pc_saft_engine.calculate_fugacity(state=state_T_neg, teste=True)


#     # ddi_dT
#     di_pos = state_T_pos.core_model.params.d
#     di_neg = state_T_neg.core_model.params.d
#     ddi_dT_num = (di_pos - di_neg) / (2 * h)
#     # print('---- ddi_dT ----')
#     # print('numerico = ', ddi_dT_num)
#     # print('Analitico = ', ddi_dT)

#     # dZeta_dT
#     zeta_pos = state_T_pos.core_model.params.zeta
#     zeta_neg = state_T_neg.core_model.params.zeta
#     zeta_num = (zeta_pos - zeta_neg) / (2 * h)
#     # print('---- dzeta_dT ----')
#     # print('numerico = ', zeta_num)
#     # print('Analitico = ', dzeta_dT)


#     Zhs_dT_pos = state_T_pos.core_model.hc_results.Z_hs
#     Zhs_dT_neg = state_T_neg.core_model.hc_results.Z_hs
#     dZhs_dT_num = (Zhs_dT_pos - Zhs_dT_neg) / (2 * h)
#     dZhs_dT_anal = _compute_dZhs_dT(zeta=zeta, dzeta_dT=dzeta_dT)
#     # print('---- dZh_dT ----')
#     # print('numerico = ', dZhs_dT_num)
#     # print('Analitico = ', dZhs_dT_anal)


#     gij_pos = state_T_pos.core_model.hc_results.gij_hs
#     gij_neg = state_T_neg.core_model.hc_results.gij_hs
#     dgij_dT_num = (gij_pos - gij_neg) / (2 * h)
#     dgij_dT_anal = _compute_dgij_dT(zeta=zeta, dzeta_dT=dzeta_dT, Dij=Dij, dDij_dT=dDij_dT)
#     # print('---- dgij_dT ----')
#     # print('numerico = ', dgij_dT_num)
#     # print('Analitico = ', dgij_dT_anal)


#     rho_dgij_drho_pos = state_T_pos.core_model.hc_results.rho_dgij_drho
#     rho_dgij_drho_neg = state_T_neg.core_model.hc_results.rho_dgij_drho
#     drho_dgij_drho_dT_num = (rho_dgij_drho_pos - rho_dgij_drho_neg) / (2 * h)
#     drho_dgij_drho_dT_anal = _compute_drho_dgij_rho_dT(zeta=zeta, dzeta_dT=dzeta_dT, Dij=Dij, dDij_dT=dDij_dT)
#     # print('---- dgij_dT ----')
#     # print('numerico = ', drho_dgij_drho_dT_num)
#     # print('Analitico = ', drho_dgij_drho_dT_anal)

#     Zh_pos = state_T_pos.core_model.hc_results.Z_hc
#     Zh_neg = state_T_neg.core_model.hc_results.Z_hc
#     dZhc_dT_num = (Zh_pos - Zh_neg) / (2 * h)
#     dZhc_dT_anal = _compute_dZhc_dT(z=z, m=m, m_mean=m_mean, dZhs_dT=dZhs_dT_anal, gij=gij, rho_dgij_drho=rho_dgij_drho,
#                                      dgij_dT=dgij_dT_anal, drho_dgij_drho_dT=drho_dgij_drho_dT_anal)
#     # print('---- dZhc_dT ----')
#     # print('numerico = ', dZhc_dT_num)
#     # print('Analitico = ', dZhc_dT_anal)

#     detaI1_eta_pos = state_T_pos.core_model.disp_results.detaI1_eta
#     detaI1_eta_neg = state_T_neg.core_model.disp_results.detaI1_eta
#     ddetaI1_eta_dT_num = (detaI1_eta_pos - detaI1_eta_neg) / (2 * h)
#     detaI2_eta_pos = state_T_pos.core_model.disp_results.detaI2_eta
#     detaI2_eta_neg = state_T_neg.core_model.disp_results.detaI2_eta
#     ddetaI2_eta_dT_num = (detaI2_eta_pos - detaI2_eta_neg) / (2 * h)
#     ddetaI1_eta_dT_anal, ddetaI2_eta_dT_anal = _compute_ddetaI12_deta_dT(dzeta3_dT=dzeta_dT[2], eta=eta, a=am, b=bm)
#     # print('---- ddetaI1_eta_dT ----')
#     # print('numerico = ', ddetaI1_eta_dT_num)
#     # print('Analitico = ', ddetaI1_eta_dT_anal)
#     # print('---- ddetaI2_eta_dT ----')
#     # print('numerico = ', ddetaI2_eta_dT_num)
#     # print('Analitico = ', ddetaI2_eta_dT_anal)

#     I1_pos = state_T_pos.core_model.disp_results.I1
#     I1_neg = state_T_neg.core_model.disp_results.I1
#     dI1_dT_num = (I1_pos - I1_neg) / (2 * h)
#     I2_pos = state_T_pos.core_model.disp_results.I2
#     I2_neg = state_T_neg.core_model.disp_results.I2
#     dI2_dT_num = (I2_pos - I2_neg) / (2 * h)
#     dI1_dT_anal, dI2_dT_anal = _compute_dI12_dT(dzeta3_dT=dzeta_dT[2], eta=eta, a=am, b=bm)
#     # print('---- dI1_dT ----')
#     # print('numerico = ', dI1_dT_num)
#     # print('Analitico = ', dI1_dT_anal)
#     # print('---- dI2_dT ----')
#     # print('numerico = ', dI2_dT_num)
#     # print('Analitico = ', dI2_dT_anal)


#     C2_pos = state_T_pos.core_model.disp_results.C2
#     C2_neg = state_T_neg.core_model.disp_results.C2
#     dC2_dT_num = (C2_pos - C2_neg) / (2 * h)
#     dC2_dT_anal = _compute_dC2_dT(eta=eta, dzeta3_dT=dzeta_dT[2], C1=C1, C2=C2, m_mean=m_mean)   
#     # print('---- dC2_dT ----')
#     # print('numerico = ', dC2_dT_num)
#     # print('Analitico = ', dC2_dT_anal)


#     Zdisp_dT_pos = state_T_pos.core_model.disp_results.Z_disp
#     Zdisp_dT_neg = state_T_neg.core_model.disp_results.Z_disp
#     dZdisp_dT_num = (Zdisp_dT_pos - Zdisp_dT_neg) / (2 * h)
#     dZdisp_dT_anal = _compute_dZdisp_dT(rho=state_trial.rho, eta=eta, detaI1_deta=detaI1_eta, detaI2_deta=detaI2_eta,
#                                         ddetaI1_deta_dT=ddetaI1_eta_dT_anal, ddetaI2_deta_dT=ddetaI2_eta_dT_anal, I2=I2,
#                                           C1=C1, C2=C2, dC2_dT=dC2_dT_anal, m2es3=m2es3, m2e2s3=m2e2s3, m_mean=m_mean, 
#                                           T=state_trial.T, dzeta3_dT=dzeta_dT[2], dI2_dT=dI2_dT_anal)
#     # print('---- dZdisp_dT ----')
#     # print('numerico = ', dZdisp_dT_num)
#     # print('Analitico = ', dZdisp_dT_anal)

#     Z_T_pos = state_T_pos.Z
#     Z_T_neg = state_T_neg.Z
#     dZ_dT_num = (Z_T_pos - Z_T_neg) / (2 * h)
#     dZ_dT_anal = dZhc_dT_anal + dZdisp_dT_anal
#     # print('---- dZ_dT ----')
#     # print('numerico = ', dZ_dT_num)
#     # print('Analitico = ', dZ_dT_anal)