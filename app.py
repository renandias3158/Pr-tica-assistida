
#slide 50
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

def get_data():
  return pd.read_csv('data.csv')

def train_model():
  data = get_data()
  selected_features = ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'PTRATIO']
  x = data[selected_features]
  y = data['MEDV']
  rf = RandomForestRegressor(random_state=42)
  rf.fit(x, y)
  return rf

#slide 51

data = get_data()

model = train_model()

#slide 52

st.title('AppIA - Previsão de Preço de Imóveis')

st.markdown('Este é um AppIA treinado para prover preços de imóveis da cidade de Boston feito pela equipe da creAIfia, os integrantes sendo João paz, Davi gleristone, Juan Vila Nova Rojas Moreno, Matheus Luciano, Todos alunos do 2B do IFPE CJBG')

st.subheader('Amostra dos dados - selecione os atributos da tabela')

defaultcols = ['RM','PTRATIO','CRIM','MEDV']

cols = st.multiselect('Atributos', data.columns.tolist(), default=defaultcols)

#slide 53

st.dataframe(data[cols].head(10))

st.subheader('Distribuição de imóveis por preço')

faixa_valores = st.slider('Selecione a faixa de preço', float(data.MEDV.min()), 150., (10.0, 100.0))

dados = data[(data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1]))]

fig = px.histogram(dados, x='MEDV', nbins=100, title='Distribuição de Preços')
fig.update_xaxes(title='MEDV')
fig.update_yaxes(title='Total imóveis')
st.plotly_chart(fig)

#slide 54

st.sidebar.subheader('Entre com as informações do imóvel a ser avaliado')

CRIM = st.sidebar.number_input('Taxa de criminalidade', value=data.CRIM.mean())

INDUS = st.sidebar.number_input('Proporção de hectares de negócios', value=data.INDUS.mean())

NOX = st.sidebar.number_input('Concentração de óxido nítrico', value=data.NOX.mean())

RM = st.sidebar.number_input('Número de quartos', value=1.)

AGE = st.sidebar.number_input('Proporção de unidades ocupadas pelos proprietários construídas antes de 1940', value=data.AGE.mean())

PTRATIO = st.sidebar.number_input('Índice de alunos para professores', value=data.PTRATIO.mean())

CHAS = st.sidebar.selectbox('Faz limite com o rio?', ('Sim', 'Não'))

if CHAS == 'Sim':
  CHAS = 1
else:
  CHAS = 0

btn_predict = st.sidebar.button('Realizar Previsão')

if btn_predict:
  result = model.predict([[CRIM, INDUS, CHAS, NOX, RM, AGE, PTRATIO]])
  st.subheader('O valor previsto para o imóvel é:')
  result = 'US $ ' + str(round(result[0]*1000,2))
  st.write(result)
