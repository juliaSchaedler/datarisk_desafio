# Desafio da Datarisk
## Contexto Geral: 

Este projeto foi desenvolvido como solução para o desafio proposto pela Datarisk. O objetivo é atuar como um cientista de dados encarregado de desenvolver um modelo preditivo para o acompanhamento de transações financeiras. O modelo deve ser capaz de estimar a probabilidade de inadimplência de clientes, definida como um atraso de 5 dias ou mais no pagamento de uma cobrança.

A solução utiliza um conjunto de dados realista, incluindo informações cadastrais, dados mensais de acompanhamento e histórico de pagamentos, para treinar um modelo de Machine Learning que identifique, com antecedência, clientes com maior risco de atraso. O resultado final é um arquivo de submissão com as probabilidades para um novo conjunto de cobranças.

## Estrutura do projeto:

![image](https://github.com/user-attachments/assets/8da2aaa3-cff8-4d60-9972-d551eb2aefb7)

## Tecnologias e bibliotecas:

Neste projeto foi utilizado Python (3.10+) e as bibliotecas e suas versões estão descritas no **requirements.txt**

## Guia de execução:

Clone o repositório e entre na pasta do projeto:

1.      git clone <https://github.com/juliaSchaedler/datarisk_desafio.git>    
2.      cd datarisk-case

Crie e ative um ambiente virtual:

3.      python -m venv .venv
4.      .venv\Scripts\activate


Instale as biblitecas necessárias:

5.       pip install -r requirements.txt

Execute o arquivo .py principal:

6.      python main.py

## Saídas do projeto (Outputs)
Ao final da execução, os seguintes arquivos serão gerados:

- submissao_case.csv: O arquivo final com as previsões de probabilidade para a base de teste, localizado na pasta raiz do projeto.

- Gráficos de Avaliação: Na pasta /output, você encontrará:

    - confusion_matrix.png: Visualização dos acertos e erros do modelo.

    - precision_recall_curve.png: Curva que demonstra o *trade-off* entre precisão e *recall*, essencial para problemas com classes desbalanceadas.

    - feature_importance.png: Gráfico com as 30 *features* mais importantes para as decisões do modelo.

## Metodologia Aplicada:

O pipeline de desenvolvimento seguiu as seguintes etapas:

- **Carregamento e Limpeza:** Os dados foram carregados, tratando o separador de ponto e vírgula (;) e convertendo os tipos de dados, especialmente as datas.

- **Engenharia de Atributos:** Foram criadas novas variáveis (features) para enriquecer o modelo, incluindo:

  - *Features de Data:* PRAZO_COBRANCA (dias entre emissão e vencimento) e DIAS_DESDE_CADASTRO.

  - *Features Categóricas:* TIPO_DOMINIO_EMAIL (classificação do domínio de e-mail em 'comum', 'corporativo', etc.).

  - *Features de Comportamento Histórico:* Utilizando groupby() e shift(), foram criadas features que resumem o comportamento passado do cliente, como VALOR_PAGAR_MEDIA_3M e TAXA_INADIMPLENCIA_CLIENTE.

  - *Features de Interação e Tendência:* Como RENDA_POR_FUNCIONARIO e VALOR_PAGAR_VS_MEDIA_3M para capturar sinais de risco mais sutis.

- **Validação do Modelo:** Para obter uma estimativa de performance robusta e evitar data leakage, foi implementada uma estratégia de Validação Cruzada Estratificada (StratifiedKFold) com 5 folds. O modelo foi treinado e avaliado 5 vezes, e a média das métricas foi considerada como o desempenho final esperado.

- **Modelagem:** O algoritmo LightGBM foi escolhido por sua eficiência, velocidade e alta performance em dados tabulares, além de sua capacidade de lidar nativamente com valores ausentes.

## Resultados do projeto:

O modelo final demonstrou uma performance excelente e confiável, alcançando uma média de AUC (Area Under Curve) de **0.9591** durante a validação cruzada. Este valor indica uma capacidade muito alta de distinguir corretamente entre clientes que se tornarão inadimplentes e os que pagarão em dia.  A análise do Relatório de Classificação e da Matriz de Confusão mostra um bom equilíbrio entre precision e recall. Isso significa que o modelo não só identifica corretamente uma boa parcela dos inadimplentes (**recall de 63% com limiar de 0.5**), mas também mantém um baixo número de alarmes falsos (**precision de 76%**). 

**O limiar de decisão pode ser ajustado pela área de negócio para otimizar o trade-off entre o custo do risco (perder um inadimplente) e o custo da ação (acionar um bom pagador).** 

Nesse cenário, pode até adotar estratégias diferentes, como uma estratégia agressiva (para reduzir o risco a todo custo. A empresa pode decidir: "Quero pegar o máximo de inadimplentes possível, não me importo de ligar para alguns clientes bons por engano". Nesse caso, podemos diminuir o limiar para 0.3 (30%). O resultado é que o *Recall* vai aumentar (vamos pegar mais de 63% dos inadimplentes), mas a *Precision* vai diminuir (teremos mais alarmes falsos).

Outra estratégia seria uma estratégia mais conservadora (de não incomodar bons clientes): A empresa pode dizer: "Só quero agir quando tiver muita certeza, para não incomodar clientes fiéis". Nesse caso, podemos aumentar o limiar para 0.8 (80%). O resultado nesse caso é que a *Precision* vai aumentar (quase todos que o modelo apontar serão inadimplentes), mas o *Recall* vai diminuir muito (vamos perder muitos casos de risco médio).

## Principal desafio: evitar data leakage e overfitting:

O principal problema enfrentado foi o data leakage, trazendo como resultado um modelo com desempenho "perfeito" no ambiente de desenvolvimento, mas que com certeza falharia em produção. Os sinais para isso foram, durante o treinamento do modelo, um AUC Perfeito (1.000), métrica perfeitas (precision, recall, f1-score com 1.00 para as duas classes)  e gráficos estranhos e irreais. 

Solução encontrada: usar previsões "Out-of-Fold" (OOF) para corrigir a geração de gráficos e avaliações no final, garantindo que eles sejam baseados em previsões "limpas", sem vazamento. A solução foi usar o poder da Validação Cruzada não apenas para calcular uma média de score, mas também para gerar um conjunto completo de previsões "fora da amostra" (Out-of-Fold). Essa abordagem garante que a avaliação final e as visualizações  representem fielmente a capacidade de generalização do modelo. Ela usa todo o conjunto de dados para avaliação, mas de uma forma que evita completamente o vazamento de informações,  dando assim uma imagem precisa e confiável do desempenho que podemos esperar em produção!

## Algumas fontes utilizadas:

Para realizar esse desafio, foi feito uso de algumas fontes para decidir sobre a utilização do algoritmo de machine learning, técnicas para realizar o processamento dos dados, bem como soluções para o problema enfretado. Abaixo estão algumas delas:

- Como o LightGBM funciona - SageMaker IA da Amazon. Disponível em: <https://docs.aws.amazon.com/pt_br/sagemaker/latest/dg/lightgbm-HowItWorks.html>.
- MUCCI, T. Data leakage machine learning. Disponível em: <https://www-ibm-com.translate.goog/think/topics/data-leakage-machine-learning?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc>. Acesso em: 8 jul. 2025.
- What is OOF approach in machine learning? Disponível em: <https://stackoverflow.com/questions/52396191/what-is-oof-approach-in-machine-learning>.
- CAETANO, T. ALGORITMOS DE APRENDIZADO DE MÁQUINA NO ESTUDO DA INADIMPLÊNCIA EM UMA INSTITUIÇÃO FINANCEIRA. [s.l: s.n.]. Disponível em: <https://repositorio.ufu.br/bitstream/123456789/41843/1/Projeto_TCC_TatianeMoreira.pdf#page=12.25>. Acesso em: 8 jul. 2025.
