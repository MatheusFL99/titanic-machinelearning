# 🛳️ Titanic - Prevendo Sobrevivência com Machine Learning

Este projeto utiliza aprendizado de máquina para prever a sobrevivência de passageiros no famoso desastre do Titanic, com base em dados reais.

## Objetivo

O objetivo principal é construir um modelo de classificação que possa prever se um passageiro sobreviveu ou não, utilizando variáveis como idade, sexo, classe da cabine, entre outras. E criar um dashboard interativo para análise exploratória e um sistema de previsão em tempo real

## Estrutura do Projeto

- `data/`: Deve conter os conjuntos de dados utilizados no projeto. Disponiveis em https://www.kaggle.com/competitions/titanic
- `notebooks/`: Notebook Jupyter com a análise exploratória e desenvolvimento do modelo.
- `README.md`: Documentação do projeto.
- `model/`: Modelo treinado

## 📦 Tecnologias utilizadas

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Plotly
- Streamlit

## 🔍 Principais Funcionalidades

**Dashboard Interativo**

- Visualizações dinâmicas com filtros em tempo real
- Gráficos interativos de sobrevivência por classe/idade
- Matriz de correlação entre variáveis
- Exibição de dados brutos

**Sistema de Previsão**

- Interface para entrada de dados
- Previsão instantânea com porcentagem de confiança
- Visualização de probabilidades detalhadas
- Suporte para múltiplos cenários

## 📊 Etapas realizadas

- Análise exploratória e estatísticas descritivas
- Tratamento de valores ausentes
- Transformação de variáveis categóricas (dummies)
- Separação entre treino e teste
- Criação e treinamento de modelo de Regressão Logística
- Avaliação de desempenho (acurácia, matriz de confusão, relatório)
- Desenvolvimento de dashboard interativo

## 🚀 Como rodar o projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/MatheusFL99/titanic-machinelearning.git
   cd titanic-machinelearning
   ```
2. Crie um ambiente virtual python:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute o notebook para treinar o modelo:

   ```bash
   jupyter lab notebooks/titanic_classification.ipynb
   ```

5. Execute o dashboard interativo:
   ```bash
   streamlit run src/dashboard.py
   ```
