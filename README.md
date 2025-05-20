
 Guia Passo a Passo: Criação e Implementação do Robô de Trading com RL
          (Modelo Global Treinado Offline e Interface Streamlit)
===============================================================================

**Introdução:**

Este documento detalha o processo de desenvolvimento e execução de um sistema de trading automatizado utilizando Aprendizado por Reforço (RL), especificamente o algoritmo A2C (Advantage Actor-Critic), para treinar um modelo global capaz de operar em múltiplos ativos. O sistema inclui um script para treinamento offline deste modelo global e uma interface gráfica interativa construída com Streamlit para testar e visualizar o desempenho do modelo em dados novos. Todas as etapas são projetadas para execução no ambiente Google Colaboratory (Colab).

**Requisitos Gerais de Software (Instalados via Pip):**

* **Python:** Linguagem de programação base.
* **Streamlit:** Para a criação da interface gráfica web interativa.
* **pyngrok:** Para expor o aplicativo Streamlit rodando no Colab para a internet.
* **Gymnasium:** Framework para desenvolver e comparar algoritmos de RL (sucessor do Gym).
* **gym-anytrading:** Biblioteca que fornece ambientes de simulação de trading de ações para Gymnasium.
* **Stable Baselines3:** Biblioteca de implementações de algoritmos de RL de alta qualidade, incluindo A2C.
* **Matplotlib:** Para a geração de gráficos estáticos.
* **NumPy:** Para computação numérica eficiente.
* **Pandas:** Para manipulação e análise de dados tabulares.

**Requisitos de Dados:**

1.  **Arquivo CSV de Treinamento:**
    * Nome do arquivo: Idealmente, nomeado como `dados_desafio_v5.csv` (ou o nome especificado na variável `ARQUIVO_CSV_TREINAMENTO` no script de treino).
    * Conteúdo: Dados históricos de cotações para todos os tickers que serão usados para treinar o modelo global.
    * Colunas Obrigatórias: `Date` (data/datetime), `Ticker` (símbolo do ativo), `Open` (preço de abertura), `High` (preço máximo), `Low` (preço mínimo), `Close` (preço de fechamento), `Volume` (volume negociado).
    * Formato da Data: Deve ser reconhecível pelo `pd.to_datetime()`.

2.  **Arquivo CSV de Teste:**
    * Formato: Idêntico ao arquivo de treinamento (mesmas colunas).
    * Conteúdo: Dados históricos para os tickers que se deseja testar com o modelo global treinado. Idealmente, dados não vistos durante o treinamento.

**Arquivos Gerados pelo Processo:**

1.  `A2C_global_model.zip`: O modelo de RL global treinado e salvo.
2.  `ticker_to_id_map.json`: Um arquivo JSON que mapeia cada ticker (do conjunto de treinamento) a um ID numérico normalizado, usado como feature pelo modelo.

---
**Passo a Passo da Implementação no Google Colab:**
---

**Passo 1: Instalação das Bibliotecas (Célula 1 do Colab)**

* **Funcionalidade:** Esta célula inicial configura o ambiente do Google Colab instalando todas as bibliotecas Python necessárias para o projeto. Utiliza o gerenciador de pacotes `pip` para buscar e instalar as versões mais recentes ou especificadas das dependências.
* **Requisitos:** Acesso à internet pela máquina virtual do Colab para baixar os pacotes.
* **Execução:** Deve ser executada uma vez no início de cada sessão do Colab.

---

**Passo 2: Criação do `backtesting_engine.py` (Célula 2 do Colab)**

* **Funcionalidade:** Este bloco de código define e salva o arquivo `backtesting_engine.py`. Este módulo Python é o núcleo do sistema, contendo:
    * **`MyCustomEnv` e `add_signals`:** Definição do ambiente de negociação personalizado para RL, que processa os dados de mercado e adiciona indicadores técnicos (SMA, RSI, OBV) e uma feature de identificação de ticker normalizada (`ticker_id_norm`) às observações do agente.
    * **`format_brl`:** Função utilitária para formatar valores numéricos como moeda brasileira (R$).
    * **`carregar_dados_csv_master`:** Função para carregar e realizar uma validação inicial no arquivo CSV de dados.
    * **`preprocess_and_feature_engineer_for_ticker`:** Função que realiza o pré-processamento detalhado dos dados de um ticker específico (tratamento de NaNs, conversão de tipos, cálculo de indicadores).
    * **`testar_ticker_com_modelo_global`:** Função principal que será chamada pelo Streamlit. Ela carrega o modelo global pré-treinado, processa os dados de teste para um ticker específico (adicionando o `ticker_id_norm` correto usando um mapa), executa a simulação de avaliação e retorna os resultados (métricas, gráfico, logs).
* **Requisitos:** As bibliotecas listadas no Passo 1 devem estar instaladas.
* **Execução:** Executar esta célula para criar o arquivo `backtesting_engine.py` no ambiente do Colab. Este arquivo será importado pelos scripts subsequentes.

---

**Passo 3: Criação do `train_global_model.py` (Célula 3 do Colab)**

* **Funcionalidade:** Este bloco cria o script `train_global_model.py`, responsável por orquestrar o treinamento offline do modelo A2C global. Suas principais tarefas são:
    * Importar as funções necessárias do `backtesting_engine.py`.
    * Definir a função `criar_dados_globais_treinamento`: Carrega o CSV de treinamento, processa os dados para cada ticker (usando `preprocess_and_feature_engineer_for_ticker`), adiciona a feature `ticker_id_norm` a cada conjunto de dados de ticker e cria um mapa (`ticker_to_id_map`) que associa cada string de ticker a seu ID numérico normalizado.
    * No bloco `if __name__ == '__main__':`:
        * Carrega os dados de treinamento.
        * Chama `criar_dados_globais_treinamento` para obter os DataFrames processados por ticker e o mapa de IDs.
        * Salva o `ticker_to_id_map.json`.
        * Inicializa um modelo A2C.
        * Itera sobre cada ticker com dados processados, configurando um ambiente `MyCustomEnv` específico para os dados daquele ticker (que já incluem `ticker_id_norm`).
        * Treina o modelo A2C global (`model_global.learn()`) iterativamente com os dados de cada ticker por um número definido de `timesteps_por_ticker_iteracao`. O aprendizado é acumulado no mesmo objeto de modelo.
        * Salva o modelo global treinado como `A2C_global_model.zip`.
* **Requisitos:** O arquivo `backtesting_engine.py` deve ter sido criado (Passo 2).
* **Execução:** Executar esta célula para criar o arquivo `train_global_model.py`.

---

**Passo 4: Upload do Arquivo CSV de TREINAMENTO (Ação Manual)**

* **Funcionalidade:** Fornecer os dados históricos que serão usados para treinar o modelo global.
* **Requisitos:** Um arquivo CSV formatado conforme especificado (com colunas `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`). O nome do arquivo deve corresponder ao valor da variável `ARQUIVO_CSV_TREINAMENTO` no script `train_global_model.py` (padrão: `dados_desafio_v5.csv`).
* **Execução:** Utilizar o painel "Arquivos" à esquerda no Google Colab para fazer o upload do seu arquivo CSV de treinamento para o diretório raiz (`/content/`).
* **OBS: No CSV enviado no Github será necessário mudar o nome da coluna 'Datetime' para 'Date' e 'Tiket' para 'Ticker' para que o modelo funcione.
---

**Passo 5: Execução do Treinamento Offline do Modelo Global (Célula 5 do Colab)**

* **Funcionalidade:** Esta célula executa o script `train_global_model.py` (`!python train_global_model.py`), que efetivamente treina o modelo A2C global.
* **Requisitos:**
    * O arquivo `backtesting_engine.py` deve existir (criado no Passo 2).
    * O arquivo `train_global_model.py` deve existir (criado no Passo 3).
    * O arquivo CSV de treinamento deve ter sido carregado (Passo 4).
* **Execução:** Executar esta célula. O processo pode ser demorado. Ao final, os arquivos `A2C_global_model.zip` e `ticker_to_id_map.json` devem ser criados no diretório `/content/`. Acompanhe os logs na saída da célula para verificar o progresso e possíveis erros.

---

**Passo 6: Criação do `app_streamlit_global_tester.py` (Célula 6 do Colab)**

* **Funcionalidade:** Este bloco cria o arquivo `app_streamlit_global_tester.py`, que define a interface gráfica do usuário (GUI) para testar o modelo global. Suas principais funcionalidades são:
    * Importar funções necessárias do `backtesting_engine.py`.
    * Definir a função `formatar_reais` para exibição de moeda.
    * Configurar a página Streamlit (título, layout).
    * Gerenciar o estado da sessão para armazenar dados carregados e resultados.
    * Criar a barra lateral (sidebar) com controles para:
        * Upload do arquivo CSV de **TESTE**.
        * Seleção de um ticker do CSV de teste.
        * Input para o "Valor de Investimento Inicial (R$)" para a simulação de teste.
        * Input para o caminho do arquivo do modelo global salvo (com valor padrão).
        * Botão para testar um ticker individualmente.
        * Botão para testar todos os tickers do CSV de teste em lote, com barra de progresso.
    * Na área principal, exibir os resultados detalhados do teste para o ticker selecionado (métricas, gráfico de simulação, tabela qualitativa, logs).
    * Exibir um resumo geral da performance de todos os tickers testados na sessão.
* **Requisitos:** O arquivo `backtesting_engine.py` deve existir.
* **Execução:** Executar esta célula para criar o arquivo `app_streamlit_global_tester.py`.

---

**Passo 7: Upload do Arquivo CSV de TESTE e Verificação dos Artefatos do Modelo (Ação Manual)**

* **Funcionalidade:** Preparar o ambiente para a execução da interface de teste.
* **Requisitos:**
    * Um arquivo CSV formatado para ser usado como dados de **TESTE**.
    * Os arquivos `A2C_global_model.zip` e `ticker_to_id_map.json` (gerados no Passo 5) devem estar presentes no diretório `/content/` (ou no caminho especificado no `app_streamlit_global_tester.py`).
* **Execução:**
    1.  Faça o upload do seu arquivo CSV de **TESTE** para o diretório raiz (`/content/`) do Colab.
    2.  Verifique no painel "Arquivos" se `A2C_global_model.zip` e `ticker_to_id_map.json` estão presentes. Se você treinou em uma sessão anterior e reiniciou o Colab, precisará fazer o upload desses arquivos novamente ou tê-los salvos no Google Drive e montar o Drive.

---

**Passo 8: Execução da Interface Streamlit com `ngrok` (Célula 7 ou 8 do Colab)**

* **Funcionalidade:** Esta célula lança o aplicativo Streamlit (`app_streamlit_global_tester.py`) e o torna acessível através de um link público na internet usando `ngrok`.
* **Requisitos:**
    * Todos os arquivos `.py` necessários (`backtesting_engine.py`, `app_streamlit_global_tester.py`) devem ter sido criados.
    * Os artefatos do modelo (`A2C_global_model.zip`, `ticker_to_id_map.json`) devem estar no local esperado.
    * Seu authtoken do `ngrok` deve ser inserido corretamente no script.
* **Execução:**
    1.  Execute esta célula. Ela irá:
        * Configurar o `ngrok` com seu authtoken.
        * Tentar finalizar processos `streamlit` ou `ngrok` anteriores.
        * Iniciar o servidor Streamlit em segundo plano.
        * Aguardar um breve período para o servidor iniciar (o `time.sleep(15)` foi aumentado para dar mais tempo).
        * Criar um túnel `ngrok` e exibir o link público.
    2.  Clique no link público gerado para abrir o dashboard Streamlit no seu navegador.
    3.  Dentro do aplicativo, carregue seu CSV de TESTE, selecione um ticker, defina o investimento e clique em "Testar Modelo Global" (ou "Testar TODOS os Tickers").
    4.  Se encontrar o erro `ERR_NGROK_108`, siga as instruções na saída da célula (verificar seu painel `ngrok.com/agents` para encerrar sessões ativas ou reiniciar a sessão do Colab e repetir os passos de criação de arquivos e execução do `ngrok`).

---
