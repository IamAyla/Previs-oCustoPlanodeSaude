# Modelo Preditivo de custo de Plano de Saúde

Problema do negócio: O objetivo desse projeto é desenvolver uma aplicação na web para uma operadora de planos de saúde para prever, a partir das características do beneficiário, o valor anual do custo do plano de saúde.

#Sobre o projeto

Com a base de dados pública, respeitando todos os aspectos da LGPD, inicei realizado uma análise explorátoria dos dados para entender as informações trazidas pelo dataset. Verifiquei que as características dos beneficiários disponíveis são: idade, sexo, índice de massa corporal, região, número de filhos e se o beneficiário é fumante. Após isso, verifiquei se haviam valores nulos, com a função info verifiquei possíveis insigths e analisei, a partir da função descrição, o dataset de forma categórica e numérica.

Para entender melhor os dados criei os gráficos mais adequados de todas as categorias e dos gastos médicos dos beneficiários. Com isso, realizei no pré-processamento dos dados o arredondamento das idades e o enconding, transformando as variáveis em numéricas e reorganizei as novas variáveis criadas.

Além disso, s características dependentes e independentes e de treino e teste.

Foram treinadas e testadas quatro algoritmos de Machine Learning, esteS são: Regressão Linear, Regressão Ridge, Regressão Lasso e Random Forest com as métricas de valiação sendo Mean squared error e Coeficiente de determinação. 

Nos testes, o algoritmo que apresentou melhor perfomance foi Random Forest chegando a 89,35%.

Por fim, realizei o salvamento da máquina preditiva para o futuro deploy ou implementação criando um arquivo pickle.


