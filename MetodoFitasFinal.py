# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:01:24 2024

@author: vitin & Thominhas
"""
import numpy as np
import matplotlib.pyplot as plt

def fita(BETA_0, dados):
    
    # Inicializando os inputs na função passados pelo dicionário 'dados'
    DISCRETIZACAO_TEMPO = dados['DISCRETIZACAO_TEMPO']
    DISCRETRIZAO_ENV = dados['DISCRETRIZAO_ENV']
    deltay = dados['deltay']
    rho = dados['rho']
    THETA_BARRA_W = dados['THETA_BARRA_W']
    THETA_BARRA_A = dados['THETA_BARRA_A']
    OMEGA_RAD = dados['OMEGA_RAD']
    GAMMA_MAIUSCULO = dados['GAMMA_MAIUSCULO']
    t = dados['t']
    dt = dados['dt']
    ALPHA_0 = dados['ALPHA_0']
    n_s = dados['n_s']
    c = dados['c']
    y = dados['y']
    U = dados['U']
    AR = dados['AR']
    mi = dados['mi']
    ALPHA_STALL_MAX = dados['ALPHA_STALL_MAX']
    Cdcf = dados['Cdcf']
    
    # Inicializar alguns vetores e matrizes. As matrizes são funções multivariáveis da posição y na envergadura e t no tempo.
    h = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); dTheta = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));THETA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));h_PONTO = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); THETA_PONTO = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO))
    ALPHA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); GAMMA = np.zeros(DISCRETIZACAO_TEMPO); THETA_2PONTO = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); ALPHA_PONTO = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO))
    k = np.zeros(DISCRETRIZAO_ENV);  F_LINHA =np.zeros(DISCRETRIZAO_ENV); G_LINHA =np.zeros(DISCRETRIZAO_ENV); C_JONES =np.zeros(DISCRETRIZAO_ENV)
    C_LINHA = np.zeros(DISCRETRIZAO_ENV); dCN = np.zeros(DISCRETRIZAO_ENV);
    ALPHA_LINHA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));dCN = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); V = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO))
    V2_PONTO = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DNA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); CN = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO))
    DNC = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DN = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DDCAMBER = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    DTS = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); RN = np.zeros(DISCRETRIZAO_ENV);
    CDF = np.zeros(DISCRETRIZAO_ENV); DDF = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DFX = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    DL = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DT = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); Vn = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    Vx = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DNcsep = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); DNasep = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    Vchap = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); VERIFICAR = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    CASOS = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
    
    # Determinar o THETA_BARRA em radianos
    THETA_BARRA = THETA_BARRA_W + THETA_BARRA_A # Ângulo médio de arfagem da corda do aerofólio em relação à velocidade do escoamento livre em graus (deg)
    THETA_BARRA_A_RAD = THETA_BARRA_A *np.pi/180  # Ângulo médio da corda em relação ao eixo de oscilação em radianos
    THETA_BARRA_RAD = THETA_BARRA*np.pi/180  # Ângulo médio de arfagem da corda do aerofólio em relação à velocidade do escoamento livre em radianos
    
    # Determinando o h, dTheta e THETA.
    for i in range(0,len(y)):
        for j in range(0,len(t)):
            h[i][j] = -GAMMA_MAIUSCULO * y[i] * np.cos (OMEGA_RAD*t[j]) # Movimento de plugging (mergulho) 
            dTheta[i][j] = -BETA_0 * y[i] *np.sin(OMEGA_RAD*t[j]) # Ângulo de ataque relativo instantâneo em radianos
            THETA[i][j] = THETA_BARRA_RAD + dTheta[i][j] # Ângulo total de arfagem da corda do aerofólio em relação à velocidade do escoamento livre em radianos
    
    # Calculando as derivadas temporais
    h_PONTO = np.gradient(h, dt, axis = 1)   # Derivada temporal de h
    THETA_PONTO = np.gradient(THETA, dt, axis = 1) # Derivada temporal de THETA
    THETA_2PONTO = np.gradient(THETA_PONTO, dt, axis = 1) # Derivada segunda temporal de THETA
    
    # Cálculo do ângulo de ataque alpha
    for i in range(0,len(y)):
        for j in range(0,len(t)):
            ALPHA[i][j] = ((h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD))+((3/4)*c[i]*THETA_PONTO[i][j])+ (U*(THETA[i][j] - THETA_BARRA_RAD)))/U  # Ângulo de ataque relativo na posição de 3/4 da corda em radianos
            GAMMA[j] = GAMMA_MAIUSCULO*np.cos(OMEGA_RAD*t[j]) # 
            
    ALPHA_PONTO = np.gradient(ALPHA, dt, axis = 1) # Derivada temporal do ângulo de ataque
    
    DOWNWASH = (2*(ALPHA_0+THETA_BARRA_RAD))/(2+AR)  # Componente do downwash
    
    # Encontrar os parâmetros para o Theodorsen modificado
    for i in range(0, len(y)):
        k[i] = (c[i]*OMEGA_RAD)/(2*U)   
        C1 = (0.5*AR)/(2.32+AR)    
        C2 = 0.181+(0.772/AR)  
        F_LINHA[i] = 1 - (C1*k[i]**2)/(k[i]**2+C2**2) 
        G_LINHA[i] = -(C1*C2*k[i])/((C2**2)+k[i]**2)  
        C_LINHA[i] = F_LINHA[i]+ 1j*G_LINHA[i]  
        C_JONES[i] = (AR*C_LINHA[i])/(2+AR)  
    
    for i in range(0,len(y)):
        for j in range(0,len(t)):
            # Calcular o alpha linha e a condição para o stall  
            ALPHA_LINHA[i][j] = (AR/(2+AR))*((F_LINHA[i]*ALPHA[i][j])+((c[i]/(2*U))*((G_LINHA[i]*ALPHA_PONTO[i][j])/k[i])))-DOWNWASH  #
            VERIFICAR[i][j] = ALPHA_LINHA[i][j] + THETA_BARRA_RAD - 0.75*(c[i]*THETA_PONTO[i][j]/U)
            if VERIFICAR[i][j] <= ALPHA_STALL_MAX: 
                CASOS[i][j] = 1
                dCN[i][j] = 2*np.pi*C_JONES[i]*ALPHA[i][j]  #
                V[i][j] = (((U*np.cos(THETA[i][j]))-(h_PONTO[i][j]*np.sin(THETA[i][j]-THETA_BARRA_A_RAD)))**2 + ((U*(ALPHA_LINHA[i][j]+THETA_BARRA_RAD))- ((c[i]*THETA_PONTO[i][j])/2))**2)**(1/2)  #
                V2_PONTO[i][j] = (U*ALPHA_PONTO[i][j])-((c[i]*THETA_2PONTO[i][j])/4)
                DNA[i][j] = ((rho*np.pi*c[i]**2)/4)*V2_PONTO[i][j]*deltay
                CN[i][j] = 2*np.pi*(ALPHA_LINHA[i][j]+ALPHA_0+THETA_BARRA_RAD)   #
                DNC[i][j] = (rho*U*V[i][j]*CN[i][j]*c[i]*deltay)/2     #
                Vn[i][j] = h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD)+0.5*c[i]*THETA_PONTO[i][j]+U*np.sin(THETA[i][j])
                Vx[i][j] = U*np.cos(THETA[i][j])-h_PONTO[i][j]*np.sin(THETA[i][j] - THETA_BARRA_A_RAD)
                Vchap[i][j] = (Vx[i][j]**2+Vn[i][j]**2)**(0.5)
                DDCAMBER[i][j] = -2*np.pi*ALPHA_0*(ALPHA_LINHA[i][j]+THETA_BARRA_RAD)*0.5*rho*U*V[i][j]*c[i]*deltay
                DTS[i][j] = n_s*2*np.pi*(((ALPHA_LINHA[i][j]+THETA_BARRA_RAD)-(0.25*c[i]*THETA_PONTO[i][j]/U))**2)*0.5*rho*U*V[i][j]*c[i]*deltay
                RN[i] = (rho*U*c[i])/mi
                CDF[i] = 0.89/((np.log10(RN[i]))**2.58)
                DDF[i][j] = CDF[i]*0.5*rho*(Vx[i][j]**2)*c[i]*deltay
                DN[i][j] = DNA[i][j]+DNC[i][j]
                DFX[i][j] = DTS[i][j] - DDCAMBER[i][j] - DDF[i][j]
                DL[i][j] = DN[i][j]*np.cos(THETA[i][j]) + DFX[i][j]*np.sin(THETA[i][j])
                DT[i][j] = DFX[i][j]*np.cos(THETA[i][j]) - DN[i][j]*np.sin(THETA[i][j])
            if VERIFICAR[i][j] > ALPHA_STALL_MAX:
                CASOS[i][j] = 2
                DDCAMBER[i][j] = 0
                DTS[i][j] = 0 
                DDF[i][j] = 0
                Vn[i][j] = h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD)+0.5*c[i]*THETA_PONTO[i][j]+U*np.sin(THETA[i][j])
                Vx[i][j] = U*np.cos(THETA[i][j])-h_PONTO[i][j]*np.sin(THETA[i][j] - THETA_BARRA_A_RAD)
                Vchap[i][j] = (Vx[i][j]**2+Vn[i][j]**2)**(0.5)
                V2_PONTO[i][j] = (U*ALPHA_PONTO[i][j])-((c[i]*THETA_2PONTO[i][j])/4)
                DNcsep[i][j] = 0.5*Cdcf*rho*Vchap[i][j]*Vn[i][j]*c[i]*deltay
                DNasep[i][j] = 0.5*((rho*np.pi*c[i]**2)/4)*V2_PONTO[i][j]*deltay
                DN[i][j] = DNcsep[i][j] + DNasep[i][j]
                DFX[i][j] = DTS[i][j]- DDCAMBER[i][j] - DDF[i][j]
                DL[i][j] = DN[i][j]*np.cos(THETA[i][j]) + DFX[i][j]*np.sin(THETA[i][j])
                DT[i][j] = DFX[i][j]*np.cos(THETA[i][j]) - DN[i][j]*np.sin(THETA[i][j])
    L_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); # sustentacao 
    T_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); L_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
    T_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
    L_SOMA = 0
    T_SOMA = 0
    
    
    for j in range(0,len(t)):  
        for i in range(0,len(y)):
            L_FITA[i][j] = np.cos(GAMMA[i])*DL[i][j]
            L_SOMA = L_SOMA + (np.cos(GAMMA[j])*DL[i][j])
            #L_TOTAL[j] = np.sum(L_FITA[i])
            T_FITA[i][j] = DT[i][j]
            T_SOMA = T_SOMA + DT[i][j]
            #T_TOTAL[j] = np.sum(T_FITA[i])
    L_MEDIO = (2/DISCRETIZACAO_TEMPO)*L_SOMA
    T_MEDIO = (2/DISCRETIZACAO_TEMPO)*T_SOMA
    
    T_TOTAL_EM_CADA_TEMPO = np.sum(T_FITA, axis=0)
    L_TOTAL_EM_CADA_TEMPO = np.sum(L_FITA, axis=0)
    
    return L_MEDIO, T_MEDIO, L_FITA, T_FITA, L_TOTAL_EM_CADA_TEMPO, T_TOTAL_EM_CADA_TEMPO, CASOS


# -------------------------- INPUTS -------------------------- #

DISCRETIZACAO_TEMPO = 20 # Discretização do tempo
DISCRETRIZAO_ENV = 12 # Discretização da envergadura
b = 5.48  # Envergadura em metros
deltay = b / (2 * DISCRETRIZAO_ENV)  # Passo na envergadura em metros
rho = 1.225 # Densidade do ar, assumindo a nível do mar no modelo ISA
THETA_BARRA_W = 0  # Ângulo do eixo de oscilação em relação à velocidade média do escoaamento em graus (deg)
THETA_BARRA_A = 7.5  # Ângulo médio da corda em relação ao eixo de oscilação em graus (deg)
OMEGA = 1.2  # Frequência da oscilação em Hz
OMEGA_RAD = 2 * np.pi * OMEGA
GAMMA_MAIUSCULO = 20 * np.pi / 180  # Amplitude do movimento de mergulho em radianos
BETA_0_TODOS = np.linspace(0, 10, 10)  # Vetor de BETA_0 em graus/m
t = np.linspace(0, 2 * np.pi / OMEGA_RAD, DISCRETIZACAO_TEMPO)  # Tempo discretizado em segundos
dt = t[1] - t[0]  # Diferencial de tempo em segundos
ALPHA_0 = 0.5 * np.pi / 180  # Ângulo de ataque de sustentação nula em radianos
n_s = 0.98
C_mac = 0.025
c = [0.744, 0.607, 0.515, 0.452, 0.416, 0.411, 0.424, 0.363, 0.309, 0.289, 0.231, 0.127]  # corda
y = np.linspace(0, b / 2, DISCRETRIZAO_ENV)
U = 13.411  # velocidade em m/s
AR = 14  # aspect ratio
mi = 1.5 * 10**(-5)
DELTA_GAMMA_MAX = 4 * np.pi / 180
QUISI = 0
ALPHA_STALL_MAX = 13 * np.pi / 180
Cdcf = 1.98

# Armazenando os inputs em um dicionário
dados = {
    'DISCRETIZACAO_TEMPO': DISCRETIZACAO_TEMPO,
    'DISCRETRIZAO_ENV': DISCRETRIZAO_ENV,
    'deltay': deltay,
    'rho': rho,
    'THETA_BARRA_W': THETA_BARRA_W,
    'THETA_BARRA_A': THETA_BARRA_A,
    'OMEGA_RAD': OMEGA_RAD,
    'GAMMA_MAIUSCULO': GAMMA_MAIUSCULO,
    'BETA_0_TODOS': BETA_0_TODOS,
    't': t,
    'dt': dt,
    'ALPHA_0': ALPHA_0,
    'n_s': n_s,
    'C_mac': C_mac,
    'c': c,
    'y': y,
    'U': U,
    'AR': AR,
    'mi': mi,
    'DELTA_GAMMA_MAX': DELTA_GAMMA_MAX,
    'QUISI': QUISI,
    'ALPHA_STALL_MAX': ALPHA_STALL_MAX,
    'Cdcf': Cdcf
}

BETA_0_VETORES = np.linspace(0, 10, 21)*np.pi/180
VETOR_L_MEDIO_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_T_MEDIO_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_T_TOTAL_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_L_TOTAL_TODOS = np.zeros(len(BETA_0_VETORES))
L_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
T_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
L_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
T_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));

for i in range(0, len(BETA_0_VETORES)):
    VETOR_L_MEDIO_TODOS[i], VETOR_T_MEDIO_TODOS[i], L_FITA, T_FITA, L_TOTAL_EM_CADA_TEMPO, T_TOTAL_EM_CADA_TEMPO, CASOS = fita(BETA_0_VETORES[i], dados)
    
plt.figure(1)
plt.plot(BETA_0_VETORES*180/np.pi, VETOR_L_MEDIO_TODOS)
plt.scatter(BETA_0_VETORES*180/np.pi, VETOR_L_MEDIO_TODOS)
plt.title('Sustentação em função de Beta0 para 12 fitas e tempo discretizado em 20')
plt.ylabel('Sustentação média (N)')
plt.xlabel('Beta0 (Deg/m)')
plt.grid()

plt.figure(2)
plt.plot(BETA_0_VETORES*180/np.pi, VETOR_T_MEDIO_TODOS)
plt.scatter(BETA_0_VETORES*180/np.pi, VETOR_T_MEDIO_TODOS)
plt.title('Empuxo em função de Beta0 para 12 fitas e tempo discretizado em 20')
plt.ylabel('Empuxo médio (N)')
plt.xlabel('Beta0 (Deg/m)')
plt.grid()

plt.show()