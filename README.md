# 🛳️ Titanic - Prevendo Sobrevivência com Machine Learning

Este projeto utiliza aprendizado de máquina para prever a sobrevivência de passageiros no famoso desastre do Titanic, com base em dados reais.

## Objetivo

O objetivo principal é construir um modelo de classificação que possa prever se um passageiro sobreviveu ou não, utilizando variáveis como idade, sexo, classe da cabine, entre outras.

## Estrutura do Projeto

- `data/`: Deve conter os conjuntos de dados utilizados no projeto. Disponiveis em https://www.kaggle.com/competitions/titanic
- `notebooks/`: Notebook Jupyter com a análise exploratória e desenvolvimento do modelo.
- `README.md`: Documentação do projeto.

## 📦 Tecnologias utilizadas

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📊 Etapas realizadas

- Análise exploratória e estatísticas descritivas
- Tratamento de valores ausentes
- Transformação de variáveis categóricas (dummies)
- Separação entre treino e teste
- Criação e treinamento de modelo de Regressão Logística
- Avaliação de desempenho (acurácia, matriz de confusão, relatório)

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
4. Execute os notebooks ou scripts para treinar e avaliar o modelo.
