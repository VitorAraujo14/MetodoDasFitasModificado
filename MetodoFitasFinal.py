# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:01:24 2024

@author: vitin & Thominhas
"""
import numpy as np
import matplotlib.pyplot as plt

def fita(BETA_0, dados):
    '''
    Na função fita, é implementada a teoria das fitas aerodinâmicas a partir da discretização da envergadura de uma 
    asa ou superfície sustentadora perfilada. Simula um carregamento aerodinâmico não estacionário que inclue o movi-
    mento de mergulho e de torção da estrutura. Cada fita possui parâmetros aerodinâmicos distintos, o que permite 
    avaliar o carregamento da estrutura para diferentes cenários de torção e mergulho da asa em regime não permanente
    de escoamento. O método se destaca pela simplicidade e permitir estipular um valor próximo do carregamento com bai-
    xo custo computacional. A função recebe um dicionário de inputs que são utilizados para os cálculos da teoria das 
    fitas aerodinâmicas. A flexão pode ser adicionada na função sem perdas de precisão, porém nessa função não foi im-
    plementada.
    Inputs:
    DISCRETIZACAO_TEMPO: Discretização do tempo
    DISCRETRIZAO_ENV: Discretização da envergadura
    deltay: Passo na envergadura em metros
    rho: Densidade do fluido
    THETA_BARRA_W: Ângulo do eixo de oscilação em relação à velocidade média do escoaamento em graus (deg)
    THETA_BARRA_A: Ângulo médio da corda em relação ao eixo de oscilação em graus (deg)
    OMEGA_RAD: Frequência da oscilação em rad/s
    GAMMA_MAIUSCULO: Amplitude do movimento de mergulho em radianos
    t: Vetor do tempo discretizado em segundos
    dt: Diferencial de tempo em segundos
    ALPHA_0: Ângulo de ataque de sustentação nula em radianos
    n_s: Eficiência da sucção do bordo de ataque. Se adotado a teoria do escoamento potencial, a eficência é 1.
    c: Vetor corda discretizado
    y: Vetor da meia envergadura discretizada em metros, considerando a oridem como a raiz da asa.
    U: Velocidade de escoamento livre em m/s.
    AR: Aspect Ratio da asa.
    mi: Viscosidade do fluido em Pa . s
    ALPHA_STALL_MAX: Ângulo de stall máximo da asa em radianos
    Cdcf: Coeficiente de força normal de pós-estol
    BETA_0: Fator linear da torção ao longo da asa
        
    Outputs:
    L_MEDIO: O valor médio da sustentação durante toda a simulação. Dado em N
    T_MEDIO: O valor médio da tração durrante toda a simulação. Dado em N
    L_FITA: Uma matriz que retorna os valores de sustentação em cada instante e em cada ponto da meia envergadura. Dado em N
    T_FITA: Uma matriz que retorna os valores de tração em cada instante e em cada ponto da meia envergadura. Dado em N
    L_TOTAL_EM_CADA_TEMPO: Um vetor que retorna a sustentação total na estrutura em cada instante de tempo discretizado. Dado em N
    T_TOTAL_EM_CADA_TEMPO: Um vetor que retorna a tração total na estrutura em cada instante de tempo discretizado. Dado em N
    CASOS: Uma matriz que retorna para cada tempo e cada ponto na envergadura o caso em que o escoamento se encontra na simulação, 1 para o caso de 'attached flow' e 2 para 'totally separated flow'.
    '''
    
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
    CASOS = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); L_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));  
    T_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO)); L_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
    T_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
    
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
    
    
    for i in range(0,len(y)):
        for j in range(0,len(t)):
            ALPHA[i][j] = ((h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD))+((3/4)*c[i]*THETA_PONTO[i][j])+ (U*(THETA[i][j] - THETA_BARRA_RAD)))/U  # Ângulo de ataque relativo na posição de 3/4 da corda para cada ponto da meia envergadura e em cada instante de tempo. Medido em radianos
            GAMMA[j] = GAMMA_MAIUSCULO*np.cos(OMEGA_RAD*t[j]) # Diedro instantâneo da fita
            
    ALPHA_PONTO = np.gradient(ALPHA, dt, axis = 1) # Derivada temporal do ângulo de ataque
    
    DOWNWASH = (2*(ALPHA_0+THETA_BARRA_RAD))/(2+AR)  # Componente do downwash
    
    # Encontrar os parâmetros para o Theodorsen modificado
    for i in range(0, len(y)):
        k[i] = (c[i]*OMEGA_RAD)/(2*U)   # Frequência reduzida
        # Aproximações de Scherer para F_LINHA, G_LINHA, C1 e C2.
        C1 = (0.5*AR)/(2.32+AR)    
        C2 = 0.181+(0.772/AR)  
        F_LINHA[i] = 1 - (C1*k[i]**2)/(k[i]**2+C2**2) 
        G_LINHA[i] = -(C1*C2*k[i])/((C2**2)+k[i]**2)  
        C_LINHA[i] = F_LINHA[i]+ 1j*G_LINHA[i]  # Termo complexo para o cálculo da função de Theodorsen modificada
        C_JONES[i] = (AR*C_LINHA[i])/(2+AR)  # Função de Theodorsen modificada
    
    for i in range(0,len(y)):
        for j in range(0,len(t)):
            # Calcular o alpha linha e a condição para o 'attached flow' para verificar nos if-statements
            ALPHA_LINHA[i][j] = (AR/(2+AR))*((F_LINHA[i]*ALPHA[i][j])+((c[i]/(2*U))*((G_LINHA[i]*ALPHA_PONTO[i][j])/k[i])))-DOWNWASH  # Ângulo de ataque relativo ao escoamento
            VERIFICAR[i][j] = ALPHA_LINHA[i][j] + THETA_BARRA_RAD - 0.75*(c[i]*THETA_PONTO[i][j]/U) # Valor para verificar a condição do escoamento na fita
            if VERIFICAR[i][j] <= ALPHA_STALL_MAX: # Condição para o escoamento nessa fita ser 'attached flow'
                CASOS[i][j] = 1 # Armazenar o caso de 'attached flow' na matriz de casos.
                dCN[i][j] = 2*np.pi*C_JONES[i]*ALPHA[i][j]  # Coeficiente de força normal não estacionário
                V[i][j] = (((U*np.cos(THETA[i][j]))-(h_PONTO[i][j]*np.sin(THETA[i][j]-THETA_BARRA_A_RAD)))**2 + ((U*(ALPHA_LINHA[i][j]+THETA_BARRA_RAD))- ((c[i]*THETA_PONTO[i][j])/2))**2)**(1/2)  # Velocidade do escoamento a 1/4 da corda
                V2_PONTO[i][j] = (U*ALPHA_PONTO[i][j])-((c[i]*THETA_2PONTO[i][j])/4) # Derivada da velocidade normal na metade da corda.
                DNA[i][j] = ((rho*np.pi*c[i]**2)/4)*V2_PONTO[i][j]*deltay # Força normal devido o efeito da massa aparente
                CN[i][j] = 2*np.pi*(ALPHA_LINHA[i][j]+ALPHA_0+THETA_BARRA_RAD)   # Coeficiente de força normal
                DNC[i][j] = (rho*U*V[i][j]*CN[i][j]*c[i]*deltay)/2     # Força normal devido a circulação do escoamento
                Vn[i][j] = h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD)+0.5*c[i]*THETA_PONTO[i][j]+U*np.sin(THETA[i][j]) # Componente vertical da velocidade na metade da corda
                Vx[i][j] = U*np.cos(THETA[i][j])-h_PONTO[i][j]*np.sin(THETA[i][j] - THETA_BARRA_A_RAD) # Velocidade do escoamento tangencial à fita
                DDCAMBER[i][j] = -2*np.pi*ALPHA_0*(ALPHA_LINHA[i][j]+THETA_BARRA_RAD)*0.5*rho*U*V[i][j]*c[i]*deltay # Força no sentido da corda devido ao arqueamento
                DTS[i][j] = n_s*2*np.pi*(((ALPHA_LINHA[i][j]+THETA_BARRA_RAD)-(0.25*c[i]*THETA_PONTO[i][j]/U))**2)*0.5*rho*U*V[i][j]*c[i]*deltay # Força no sentido da corda devido aa sucção no bordo de ataque
                RN[i] = (rho*U*c[i])/mi # Reynolds local instantâneo 
                CDF[i] = 0.89/((np.log10(RN[i]))**2.58) # Coeficiente de atrito local instantâneo 
                DDF[i][j] = CDF[i]*0.5*rho*(Vx[i][j]**2)*c[i]*deltay # Arrasto viscoso na direção da corda
                DN[i][j] = DNA[i][j]+DNC[i][j] # Força normal total instantânea na fita 
                DFX[i][j] = DTS[i][j] - DDCAMBER[i][j] - DDF[i][j] # Força na direção da corda total instantânea na fita
                DL[i][j] = DN[i][j]*np.cos(THETA[i][j]) + DFX[i][j]*np.sin(THETA[i][j]) # Sustentação instantânea 
                DT[i][j] = DFX[i][j]*np.cos(THETA[i][j]) - DN[i][j]*np.sin(THETA[i][j]) # Tração instantâneo
            if VERIFICAR[i][j] > ALPHA_STALL_MAX: # Condição para o 'totally separated flow'
                CASOS[i][j] = 2 # Armazenar o caso de 'totally separated flow' na matriz de casos.
                DDCAMBER[i][j] = 0 # Força no sentido da corda devido aa sucção no bordo de ataque.
                DTS[i][j] = 0 # Força no sentido da corda devido aa sucção no bordo de ataque
                DDF[i][j] = 0 # Arrasto viscoso na direção da corda
                # Nesse caso, são nulos DDCAMBER, DTS e DDF, devido a separação do escoamento na fita.
                Vn[i][j] = h_PONTO[i][j]*np.cos(THETA[i][j]-THETA_BARRA_A_RAD)+0.5*c[i]*THETA_PONTO[i][j]+U*np.sin(THETA[i][j]) # Componente vertical da velocidade na metade da corda
                Vx[i][j] = U*np.cos(THETA[i][j])-h_PONTO[i][j]*np.sin(THETA[i][j] - THETA_BARRA_A_RAD) # Velocidade do escoamento tangencial à fita
                Vchap[i][j] = (Vx[i][j]**2+Vn[i][j]**2)**(0.5) # Velocidade resultante na metade da corda
                V2_PONTO[i][j] = (U*ALPHA_PONTO[i][j])-((c[i]*THETA_2PONTO[i][j])/4) # Derivada da velocidade normal na metade da corda.
                DNcsep[i][j] = 0.5*Cdcf*rho*Vchap[i][j]*Vn[i][j]*c[i]*deltay # Força normal devido a circulação do escoamento quando o escoamento está totalmente descolado da fita
                DNasep[i][j] = 0.5*((rho*np.pi*c[i]**2)/4)*V2_PONTO[i][j]*deltay # Força normal devido o efeito da massa aparente quando o escoamento está totalmente descolado da fita
                DN[i][j] = DNcsep[i][j] + DNasep[i][j] # Força normal total instantânea na fita 
                DFX[i][j] = DTS[i][j]- DDCAMBER[i][j] - DDF[i][j] # Força na direção da corda total instantânea na fita
                DL[i][j] = DN[i][j]*np.cos(THETA[i][j]) + DFX[i][j]*np.sin(THETA[i][j]) # Sustentação instantânea
                DT[i][j] = DFX[i][j]*np.cos(THETA[i][j]) - DN[i][j]*np.sin(THETA[i][j]) # Tração instantâneo
    
    L_SOMA = 0 # Inicializar a variável para a soma da sustentação
    T_SOMA = 0 # Inicializar a variável para a soma da tração
    
    # Cálculo da sustentação/tração médias, totais para cada instante de tempo e as matrizes ao longo da envergadura e tempo de cada carregamento
    for j in range(0,len(t)):  
        for i in range(0,len(y)):
            L_FITA[i][j] = np.cos(GAMMA[i])*DL[i][j] # Valor da sustentação instantânea em cada fita
            L_SOMA = L_SOMA + (np.cos(GAMMA[j])*DL[i][j]) # Somatório de toda a sustentação ao longo da meia envergadura no intervalo de tempo avaliado
            T_FITA[i][j] = DT[i][j] # Valor da tração instantânea em cada fita
            T_SOMA = T_SOMA + DT[i][j] # Somatório de toda a tração ao longo da meia envergadura no intervalo de tempo avaliado
    L_MEDIO = (2/DISCRETIZACAO_TEMPO)*L_SOMA # Obtenção da sustentação média no intervalo de tempo avaliado
    T_MEDIO = (2/DISCRETIZACAO_TEMPO)*T_SOMA # Obtenção da tração média no intervalo de tempo avaliado
    
    T_TOTAL_EM_CADA_TEMPO = np.sum(T_FITA, axis=0) # Somatório de todos os valores da tração de cada fita para cada instante de tempo. É um vetor com a tração total para cada instante de tempo
    L_TOTAL_EM_CADA_TEMPO = np.sum(L_FITA, axis=0) # Somatório de todos os valores da sustentação de cada fita para cada instante de tempo. É um vetor com a sustentação total para cada instante de tempo
    
    return L_MEDIO, T_MEDIO, L_FITA, T_FITA, L_TOTAL_EM_CADA_TEMPO, T_TOTAL_EM_CADA_TEMPO, CASOS # Outputs do método da fita


# -------------------------- INPUTS -------------------------- #

DISCRETIZACAO_TEMPO = 20 # Discretização do tempo
DISCRETRIZAO_ENV = 12 # Discretização da envergadura
b = 5.48  # Envergadura em metros
deltay = b / (2 * DISCRETRIZAO_ENV)  # Passo na envergadura em metros
rho = 1.225 # Densidade do ar, assumindo a nível do mar no modelo ISA
THETA_BARRA_W = 0  # Ângulo do eixo de oscilação em relação à velocidade média do escoaamento em graus (deg)
THETA_BARRA_A = 7.5  # Ângulo médio da corda em relação ao eixo de oscilação em graus (deg)
OMEGA = 1.2  # Frequência da oscilação em Hz
OMEGA_RAD = 2 * np.pi * OMEGA # Frequência da oscilação em rad/s
GAMMA_MAIUSCULO = 20 * np.pi / 180  # Amplitude do movimento de mergulho em radianos
BETA_0_TODOS = np.linspace(0, 10, 10)  # Vetor de BETA_0 em graus/m
t = np.linspace(0, 2 * np.pi / OMEGA_RAD, DISCRETIZACAO_TEMPO)  # Tempo discretizado em segundos
dt = t[1] - t[0]  # Diferencial de tempo em segundos
ALPHA_0 = 0.5 * np.pi / 180  # Ângulo de ataque de sustentação nula em radianos
n_s = 0.98 # Eficiência da sucção do bordo de ataque.
c = [0.744, 0.607, 0.515, 0.452, 0.416, 0.411, 0.424, 0.363, 0.309, 0.289, 0.231, 0.127]  # Corda discretizada em metros
y = np.linspace(0, b / 2, DISCRETRIZAO_ENV) # Discretização da meia envergadura em metros
U = 13.411  # Velocidade do escoamento livre em m/s
AR = 14  # Aspect Ratio
mi = 1.5 * 10**(-5) # Viscosidade do ar em Pa . s
ALPHA_STALL_MAX = 13 * np.pi / 180 # Ângulo de stall máximo da asa em radianos
Cdcf = 1.98 # Coeficiente de força normal de pós-estol

BETA_0_VETORES = np.linspace(0, 10, 21)*np.pi/180 # Definindo os valores de Beta0 em um vetor. 
# O objetivo é analisar a sustentação e a tração média para vários valores de Beta0.

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
    'c': c,
    'y': y,
    'U': U,
    'AR': AR,
    'mi': mi,
    'ALPHA_STALL_MAX': ALPHA_STALL_MAX,
    'Cdcf': Cdcf
}


# Inicializando os vetores e matrizes para armazenar os outputs da função fita
VETOR_L_MEDIO_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_T_MEDIO_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_T_TOTAL_TODOS = np.zeros(len(BETA_0_VETORES))
VETOR_L_TOTAL_TODOS = np.zeros(len(BETA_0_VETORES))
L_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
T_FITA = np.zeros((DISCRETRIZAO_ENV,DISCRETIZACAO_TEMPO));
L_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));
T_TOTAL_EM_CADA_TEMPO = np.zeros((DISCRETIZACAO_TEMPO));

# Realizando as simulações para vários valores de Beta0 e armazenando nos vetores e matrizes inicializados acima
for i in range(0, len(BETA_0_VETORES)):
    VETOR_L_MEDIO_TODOS[i], VETOR_T_MEDIO_TODOS[i], L_FITA, T_FITA, L_TOTAL_EM_CADA_TEMPO, T_TOTAL_EM_CADA_TEMPO, CASOS = fita(BETA_0_VETORES[i], dados)
    
# Plotando a sustentação média para os vários valores de Beta0
plt.figure(1)
plt.plot(BETA_0_VETORES*180/np.pi, VETOR_L_MEDIO_TODOS)
plt.scatter(BETA_0_VETORES*180/np.pi, VETOR_L_MEDIO_TODOS)
plt.title('Sustentação em função de Beta0')
plt.ylabel('Sustentação média (N)')
plt.xlabel('Beta0 (Deg/m)')
plt.grid()

# Plotando a tração média para os vários valores de Beta0
plt.figure(2)
plt.plot(BETA_0_VETORES*180/np.pi, VETOR_T_MEDIO_TODOS)
plt.scatter(BETA_0_VETORES*180/np.pi, VETOR_T_MEDIO_TODOS)
plt.title('Tração média em função de Beta0')
plt.ylabel('Tração média (N)')
plt.xlabel('Beta0 (Deg/m)')
plt.grid()

# Escolhendo alguns valores de BETA_0 para plotar a sustentação e a tração totais ao longo do tempo
betas_escolhidos = [BETA_0_VETORES[0], BETA_0_VETORES[5], BETA_0_VETORES[10], BETA_0_VETORES[15], BETA_0_VETORES[20]]

plt.figure(3)
# Plotando a sustentação total para cada instante ao longo do tempo
for i in range(0, len(betas_escolhidos)):
    _, _, _, _, L_TOTAL_EM_CADA_TEMPO, _ , _ = fita(betas_escolhidos[i], dados)
    legenda = 'Beta0: '+str(betas_escolhidos[i]*180/np.pi)+'°'
    plt.plot(t,L_TOTAL_EM_CADA_TEMPO, label=legenda)
    plt.title('Sustentação total na asa ao longo do tempo')
    plt.ylabel('Sustentação total (N)')
    plt.xlabel('Tempo (s)')
    plt.legend()
    plt.grid()

plt.figure(4)
# Plotando a tração total para cada instante ao longo do tempo
for i in range(0, len(betas_escolhidos)):
    _, _, _, _, _ , T_TOTAL_EM_CADA_TEMPO, _ = fita(betas_escolhidos[i], dados)
    legenda = 'Beta0: '+str(betas_escolhidos[i]*180/np.pi)+'°'
    plt.plot(t,T_TOTAL_EM_CADA_TEMPO, label=legenda)
    plt.title('Tração total na asa ao longo do tempo')
    plt.ylabel('Tração total (N)')
    plt.xlabel('Tempo (s)')
    plt.legend()
    plt.grid()
    
plt.show() # Plotando todos os gráficos