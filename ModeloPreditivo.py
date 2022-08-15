#Problema do negócio: O objetivo é desenvolver uma metodologia para previsão do valor do plano de saúde para seus beneficiários.
#Análise exploratória
#importando as bibliotecas:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Carregando o dataset:
train_data = pd.read_csv('Train_data.csv')
#Checando os 5 primeiros:
train_data.head()

#Verificando o shape do dataset:
train_data.shape
#Checando se existem valores nulos:
train_data.isnull().sum()
#Insights do dataset:
train_data.info()
#Verificando a descrição do dataset (numérica):
train_data.describe()
#Verificando a descrição do dataset (categorica):
train_data.describe(include=['O'])

#Histograma das taxas de seguro médico:
plt.figure(figsize=(8,5))
sns.histplot(train_data['charges'], kde=True)
plt.title('Despesas com plano de Saúde', fontsize=20)
plt.show()

#Boxplot das taxas com seguro médico:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['charges'])
plt.title('Despesas com plano de saúde (boxplot)', fontsize=20)
plt.show()

#histograma das idades:
plt.figure(figsize=(8,5))
sns.histplot(train_data['age'], kde=True)
plt.title('Idade', fontsize=20)
plt.show()

#Boxplot das idades:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['age'])
plt.title('Idade (boxplot)', fontsize=20)
plt.show()

#histograma das imc:
plt.figure(figsize=(8,5))
sns.histplot(train_data['bmi'], kde=True)
plt.title('Índice de Massa Corporal', fontsize=20)
plt.show()

#Boxplot das imc:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['bmi'])
plt.title('Índice de Massa Corporal (boxplot)', fontsize=20)
plt.show()

#Histograma dos filhos:
plt.figure(figsize=(8,5))
sns.histplot(train_data['children'], kde=True)
plt.title('Filhos', fontsize=20)
plt.show()

#Boxplot dos filhos:
plt.figure(figsize=(8,5))
sns.boxplot(train_data['children'])
plt.title('Filhos (boxplot)', fontsize=20)
plt.show()

#Histograma do sexo:
plt.figure(figsize=(6,4))
sns.countplot(train_data['sex'])
plt.title('Sexo', fontsize=20)
plt.show()

#Contagem de valores fumantes
print('Smokers        :', train_data['smoker'].value_counts()[1])
print('Non-Smokers    :', train_data['smoker'].value_counts()[0])
#Vizualização
sns.countplot(train_data['smoker'])
sns.countplot(train_data['smoker'])
plt.title('Smoker', fontsize=20)
plt.show()

#Contagem de valores regiões
print('South-East region:', train_data['region'].value_counts()[0])
print('North-West region:', train_data['region'].value_counts()[1])
print('South-West region:', train_data['region'].value_counts()[2])
print('North-East region:', train_data['region'].value_counts()[3])
#Vizualização
sns.countplot(train_data['region'])
sns.countplot(train_data['region'])
plt.title('Regiões', fontsize=20)
plt.show()

#5 primeiras aparições
train_data.head()

#Pré-processamento de dados
#Arredondando a variável age:
train_data['age'] = round(train_data['age'])
#5 primeiros pós arredondamento:
train_data.head()

#OHEnconding: Transformar variáveis em númericas
train_data = pd.get_dummies(train_data, drop_first=True)
#Dois primeiros depois do encoding:
train_data.head(2)

#Colunas do dataset
train_data.columns
#Reorganizando as colunas para uma melhor visualização
train_data = train_data[['age', 'sex_male','smoker_yes', 'bmi', 'region_northwest', 'region_southeast', 'region_southwest', 'charges']]
train_data.head(2)

#Separando 
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
#2 primeiros das características independentes
X.head(2)
#2 primeiros das características dependentes
y.head(2)

#Separando em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Construção e avaliação da Máquina Preditiva
#Criando a metodologia de previsão do valor do custo do plano de saúde que é a própria máquina
#Importando as métricas de avaliação:
from sklearn.metrics import mean_squared_error, r2_score

#MP com regressão linear
#Regressão linear
from sklearn.linear_model import LinearRegression
LinearRegression = LinearRegression()
LinearRegression = LinearRegression.fit(X_train, y_train)
#Predição:
y_pred = LinearRegression.predict(X_test)
#Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#MP com Regressão Ridge
#Ridge
from sklearn.linear_model import Ridge
Ridge = Ridge()
Ridge = Ridge.fit(X_train, y_train)
#Predição:
y_pred = Ridge.predict(X_test)
#Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#MP com Regressão Lasso
#Lasso:
from sklearn.linear_model import Lasso
Lasso = Lasso()
Lasso = Lasso.fit(X_train, y_train)
#Predição 
y_pred = Lasso.predict(X_test)
#Scores
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#MP com Random Forest
#Randon Forest Regressor
from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor = RandomForestRegressor()
RandomForestRegressor = RandomForestRegressor.fit(X_train, y_train)
#Predição
y_pred = RandomForestRegressor.predict(X_test)
#Scores
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#Salvamento da Máquina Preditiva para Deploy ou implementação
#Criando um arquivo pickle para o classificador
import pickle
filename = 'MedicalInsuranceCost.pkl'
pickle.dump(RandomForestRegressor, open(filename, 'wb'))
