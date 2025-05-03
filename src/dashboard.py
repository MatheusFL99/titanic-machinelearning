import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# configuracao
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# load dos dados
@st.cache_data
def load_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    return train, test

@st.cache_resource
def load_model():
    return joblib.load('../model/titanic_model.pkl')

train_df, test_Df = load_data()
model = load_model()

# filtros
with st.sidebar:
    st.header("Configurações")

    # filtro de dados
    st.subheader("Filtro de dados")
    selected_class = st.multiselect('Classe', options=[1,2,3], default=[1,2,3])
    selected_sex = st.multiselect('Sexo', options=['male', 'female'], default=['male', 'female'])

    # previsao em tempo real
    st.subheader("Previsão em Tempo Real")
    pclass = st.selectbox('Classe', [1, 2, 3], key='pred_class')
    sex = st.selectbox('Sexo', ['male', 'female'], key='pred_sex')
    age = st.number_input('Idade', min_value=0, max_value=100, value=25)
    sibsp = st.number_input('Irmãos/Cônjuge', min_value=0, max_value=8, value=0)
    parch = st.number_input('Pais/Filhos', min_value=0, max_value=6, value=0)
    fare = st.number_input('Tarifa', min_value=0.0, max_value=512.0, value=32.0)
    embarked = st.selectbox('Porto de Embarque', ['S', 'C', 'Q'])

    predict_button = st.button('Prever Sobrevivência')

# processando os filtros
filtered_data = train_df[
    (train_df['Pclass'].isin(selected_class)) &
    (train_df['Sex'].isin(selected_sex))
]


# layout
st.title("Titanic Dashboard - Análise e Previsões")
st.markdown("Explore os dados e faça previsões em tempo real")

# metricas
st.subheader("Metricas")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Passageiros", filtered_data.shape[0])

with col2:
    try:
        survival_rate = filtered_data['Survived'].mean() * 100
        st.metric("Taxa de Sobrevivência", f"{survival_rate:.2f}%")
    except KeyError:
        st.error("Dados não contêm coluna 'Survived'")
    except ZeroDivisionError:
        st.warning("Nenhum dado disponível para cálculo")

with col3:
    avg_age = filtered_data['Age'].mean()
    st.metric("Idade Média", f"{avg_age:.0f} anos")

# visualização
st.subheader("Visualização de Dados")
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("### Sobrevivência por Classe")
    survival_class = filtered_data.groupby(['Pclass', 'Survived']).size().unstack()
    survival_class['Survival Rate'] = survival_class[1] / (survival_class[0] + survival_class[1]) * 100
    fig = px.bar(survival_class, x=survival_class.index, y='Survival Rate',
                 color_discrete_sequence=['#2ca02c'],
                 labels={'x': 'Classe', 'y': 'Taxa de Sobrevivência (%)'})
    st.plotly_chart(fig, use_container_width=True)

with fig_col2:
    st.markdown("### Distribuição de Idades por Sobrevivência")
    fig = px.histogram(filtered_data, x='Age', nbins=20, color='Survived',
                       color_discrete_sequence=['#d62728', '#2ca02c'],
                       labels={'Survived': 'Sobreviveu'},
                       category_orders={'Survived': [0, 1]})
    st.plotly_chart(fig, use_container_width=True)

# heatmap
st.subheader("Correlação entre Variáveis")
corr = filtered_data.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# previsão
if predict_button:
    st.subheader("Resultado da Previsão")
    
    # preprocessamento
    input_data = {
        'Pclass': pclass,
        'Sex': 0 if sex == 'male' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0
    }
    
    input_df = pd.DataFrame([input_data])
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    input_df = input_df[features]
    
    # fazendo previsao
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]
    

    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("### Previsão Final")
        if prediction[0] == 1:
            st.success("## Sobreviveria")
        else:
            st.error("## Não Sobreviveria")
        
        st.metric("Confiança da Previsão", 
                 f"{max(proba)*100:.2f}%",
                 delta_color="off")

    with result_col2:
        st.markdown("### Probabilidades Detalhadas")
        fig = px.pie(
            names=['Não Sobrevivência', 'Sobrevivência'],
            values=proba,
            color=['Não Sobrevivência', 'Sobrevivência'],
            color_discrete_sequence=['#d62728', '#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)

# dados 
st.subheader("📁 Dados Filtrados")
if st.checkbox('Mostrar dados brutos'):
    st.write(filtered_data)

# rodapé
st.markdown("---")
st.markdown("Desenvolvido por Matheus Faustino Lima | [Titanic - Prevendo Sobrevivência com Machine Learning](https://github.com/MatheusFL99/titanic-machinelearning)")