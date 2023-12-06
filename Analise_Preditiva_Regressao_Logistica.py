
'''
import pandas as pd
import numpy as np

# Configuração para tornar a geração de números aleatórios reproduzível
np.random.seed(42)

# Gerando dados fictícios
num_samples = 1000

# Idade da peça em dias (variando de 1 a 365)
idade_peca = np.random.randint(1, 366, num_samples)

# Temperatura em graus Celsius (variando de 0 a 100)
temperatura = np.random.uniform(0, 100, num_samples)

# Pressão em PSI (variando de 50 a 150)
pressao = np.random.uniform(50, 150, num_samples)

# Velocidade em metros por segundo (variando de 1 a 10)
velocidade = np.random.uniform(1, 10, num_samples)

# Vibração em unidades arbitrárias (variando de 0 a 1)
vibracao = np.random.uniform(0, 1, num_samples)

# Nível de lubrificação (variando de 0 a 1)
lubrificacao = np.random.uniform(0, 1, num_samples)

# Desgaste da peça (variando de 0 a 1)
desgaste = np.random.uniform(0, 1, num_samples)

# Criando uma coluna de falha (1 se falhou, 0 se não falhou)
falha = np.random.choice([0, 1], num_samples)

# Criando um DataFrame
dados_completos = pd.DataFrame({
    'Idade_Peca': idade_peca,
    'Temperatura': temperatura,
    'Pressao': pressao,
    'Velocidade': velocidade,
    'Vibracao': vibracao,
    'Lubrificacao': lubrificacao,
    'Desgaste': desgaste,
    'Falha': falha
})

# Exibindo as primeiras linhas do DataFrame
dados_completos.to_excel('C:\\Users\\vinic\\OneDrive\\Área de Trabalho\\base_treino.xlsx', index=False) #Informar caminho para salvar o arquivo 
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

modelo = LogisticRegression() #Criando Modelo

#Treino
df_treino = pd.read_excel('') # Caminho do conjunto de dados de treino

X_treino = df_treino[['Idade_Peca','Temperatura','Pressao','Velocidade','Vibracao','Lubrificacao','Desgaste']]
y_treino = df_treino['Falha'] # O que estamos buscando prever! quebra ou não quebra?


weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_treino), y=y_treino)
modelo = LogisticRegression(class_weight={0: weights[0], 1: weights[1]}) # definie pesos para cada situação para evitar desbalanceamento

modelo.fit(X_treino, y_treino)


#Teste
df_teste = pd.read_excel('') # Caminho do conjunto de dados de teste

X_teste = df_teste[['Idade_Peca','Temperatura','Pressao','Velocidade','Vibracao','Lubrificacao','Desgaste']] #pode ser subistituido por qualquer parâmetro dentro do contexto
y_teste = df_teste['Falha'] # O que estamos buscando prever! quebra ou não quebra?
y_pred = modelo.predict(X_teste)


accuracy = accuracy_score(y_teste, y_pred)
report = classification_report(y_teste, y_pred)
matrix = confusion_matrix(y_teste, y_pred)

print(f'Precisão:{accuracy*100}% \n\n {report} \n\n {matrix} \n\n') # Mostra os resutados sobre a precisão do modelo 


#Previsão de novos dados
df_novo = pd.read_excel('') #Caminho do arquivo dos novos dados
X_novos = df_novo[['Idade_Peca','Temperatura','Pressao','Velocidade','Vibracao','Lubrificacao','Desgaste']] # Parâmetros

previsao = modelo.predict(X_novos)

resultado = pd.DataFrame({
    'Idade_Peca': X_novos['Idade_Peca'],
    'Temperatura': X_novos['Temperatura'],
    'Pressao': X_novos['Pressao'],
    'Velocidade': X_novos['Velocidade'],
    'Vibracao': X_novos['Vibracao'],
    'Lubrificacao': X_novos['Lubrificacao'],
    'Desgaste': X_novos['Desgaste'],
    'Previsão': previsao,
    
})

print(resultado)
resultado.to_excel('')