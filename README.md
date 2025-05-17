# Desafio-Investe.AI
Desafio 8: Investe.AI - Grupo: Vazios e Nulos
___________________________________________________________________________________________________________________________________________________________________

===============================================================================
Guia de Criação e Implementação do Robô de Trading com RL e Streamlit
===============================================================================

Este guia descreve as etapas envolvidas na construção e execução do sistema de trading automatizado usando Aprendizado por Reforço (A2C) e uma interface gráfica interativa com Streamlit.

-------------------------------------------------------------------------------
## Etapa 1: Configuração Inicial e Coleta de Dados
-------------------------------------------------------------------------------

1.  **Bibliotecas Necessárias:**
    * O projeto utiliza Python e requer a instalação de várias bibliotecas. No Google Colab, a primeira célula do notebook é dedicada a isso:
        `!pip install streamlit pyngrok gymnasium gym_anytrading stable-baselines3 matplotlib numpy pandas --upgrade`
    * Principais bibliotecas:
        * `gymnasium`: Para criar os ambientes de RL (atualização do `gym`).
        * `gym_anytrading`: Fornece um ambiente base para simulação de trading de ações.
        * `stable-baselines3`: Para implementar e treinar agentes de RL como o A2C.
        * `numpy`, `pandas`: Para manipulação de dados.
        * `matplotlib`: Para gerar gráficos.
        * `streamlit`: Para criar a interface gráfica web.
        * `pyngrok`: Para expor o aplicativo Streamlit rodando no Colab para a internet.

2.  **Arquivo de Dados de Entrada:**
    * O sistema espera um arquivo CSV contendo dados históricos de cotações.
    * Colunas obrigatórias: `Date`, `Ticker` (símbolo do ativo), `Open`, `High`, `Low`, `Close`, `Volume`.
    * A coluna `Date` deve estar em um formato que o Pandas possa converter para datetime.

-------------------------------------------------------------------------------
## Etapa 2: `backtesting_engine.py` - O Coração da Lógica de Trading
-------------------------------------------------------------------------------
Este arquivo Python contém toda a lógica central para processar os dados, treinar o agente e avaliá-lo para um ticker específico. Ele é projetado para ser importado e usado pelo aplicativo Streamlit.

**A. Pré-processamento de Dados:**
    1.  **Carregamento do DataFrame Completo:** Uma função (`carregar_dados_csv_master`) carrega o CSV principal e converte a coluna `Date` para o formato datetime.
    2.  **Filtragem por Ticker:** A função principal de análise (`analisar_ticker_pipeline_completo`) recebe o DataFrame completo e o símbolo do ticker. Ela primeiro filtra os dados para obter apenas as informações do ticker especificado.
    3.  **Indexação por Data:** A coluna `Date` do DataFrame filtrado é definida como o índice. O DataFrame é então ordenado por este índice.
    4.  **Validação e Limpeza de Colunas OHLCV:**
        * Verifica se as colunas `Open`, `High`, `Low`, `Close`, `Volume` existem.
        * Converte a coluna `Volume` e as colunas OHLC para o tipo numérico.
        * Trata quaisquer valores ausentes (NaNs) nas colunas OHLCV usando preenchimento progressivo (`ffill`) e regressivo (`bfill`).
        * Garante que o índice de data seja único (removendo duplicatas, se houver).

**B. Engenharia de Features (Indicadores Técnicos):**
    * Para fornecer mais informações ao agente de RL, indicadores técnicos são calculados manualmente usando Pandas:
        * **SMA (Simple Moving Average):** Média Móvel Simples do preço de fechamento (ex: período de 12 dias).
        * **RSI (Relative Strength Index):** Índice de Força Relativa (ex: período de 14 dias).
        * **OBV (On-Balance Volume):** Saldo de Volume.
    * Após o cálculo, quaisquer NaNs gerados (especialmente no início da série devido às janelas dos indicadores) são preenchidos com 0.

**C. Ambiente de Aprendizado por Reforço (RL):**
    1.  **Classe `MyCustomEnv`:**
        * Uma classe de ambiente personalizada é criada herdando de `StocksEnv` da biblioteca `gym_anytrading`.
        * Isso permite modificar o processamento de dados que o ambiente fornece ao agente.
    2.  **Função `add_signals`:**
        * Esta função é usada pela `MyCustomEnv` para preparar os dados de observação para o agente.
        * Ela seleciona as features relevantes do DataFrame (ex: `Low`, `Volume`, `SMA`, `RSI`, `OBV`) e as retorna como um array NumPy.
    3.  **Criação dos Ambientes:**
        * São criadas instâncias de `MyCustomEnv` para treinamento e avaliação.
        * Os ambientes são configurados com o DataFrame processado, o tamanho da janela de observação (`window_size`) e os limites de dados (`frame_bound`).
        * O "valor de investimento inicial" fornecido pelo usuário no Streamlit é usado externamente para calcular o lucro em R$ e pode ser usado internamente pelo ambiente se a versão do `gym-anytrading` suportar (na nossa implementação atual, o ambiente usa seu saldo padrão, e o cálculo em R$ é uma projeção).

**D. Treinamento do Agente (A2C):**
    1.  **DummyVecEnv:** O ambiente de treinamento é encapsulado em um `DummyVecEnv` para compatibilidade com `Stable Baselines3`.
    2.  **Modelo A2C:** Um agente A2C (Advantage Actor-Critic) é inicializado com uma política `MlpPolicy`.
    3.  **Aprendizado (`model.learn()`):** O agente é treinado por um número especificado de `total_timesteps`.

**E. Avaliação do Agente:**
    1.  Após o treinamento, o modelo é avaliado em um conjunto de dados de avaliação.
    2.  O modelo toma decisões de forma determinística.
    3.  São extraídas informações como `total_reward` e `total_profit` (lucro percentual).
    4.  **Cálculo do Lucro em R$:** O lucro percentual é usado com o `valor_investimento_inicial_reais` (do Streamlit) para calcular o lucro monetário.
    5.  **Geração do Gráfico:** Um gráfico Matplotlib da simulação é gerado.

**F. Função Principal `analisar_ticker_pipeline_completo`:**
    * Encapsula as sub-etapas A-E para um ticker.
    * Recebe o ticker, DataFrame completo, valor de investimento inicial e timesteps de treino.
    * Retorna um dicionário com resultados (gráficos, métricas, logs, etc.).

**G. Função `format_brl` (ou `formatar_reais` no Streamlit):**
    * Utilizada para formatar números como moeda brasileira (R$).

-------------------------------------------------------------------------------
## Etapa 3: `app_streamlit_online.py` - A Interface Gráfica Interativa
-------------------------------------------------------------------------------
Este arquivo Python cria a interface web usando Streamlit, permitindo ao usuário interagir com o motor de backtesting.

**A. Configuração Inicial do App:**
    * Importa bibliotecas, incluindo `streamlit` e funções de `backtesting_engine.py`.
    * Define funções auxiliares (ex: `formatar_reais`).
    * Configura o layout da página e exibe título/legenda.

**B. Gerenciamento de Estado da Sessão (`st.session_state`):**
    * Para manter informações (dados carregados, resultados) persistentes entre interações do usuário.
    * Armazena o DataFrame principal, lista de tickers e resultados por ticker.

**C. Interface do Usuário (Sidebar e Área Principal):**
    * **Sidebar (`st.sidebar`):** Contém os controles principais:
        * Upload de arquivo CSV.
        * Seleção de ticker.
        * Input para "Valor de Investimento Inicial (R$)".
        * Input para "Timesteps para Treinamento" (individual e em lote).
        * Botão "Analisar Ticker Individual".
        * Botão "Analisar TODOS os Tickers".
        * Checkbox "Reprocessar tickers já analisados".
    * **Área Principal:** Usada para exibir os resultados e o progresso.

**D. Lógica de Chamada ao Engine:**
    * Ao clicar nos botões de análise, a função `analisar_ticker_pipeline_completo` de `backtesting_engine.py` é chamada.
    * `st.spinner` e `st.progress` mostram feedback durante o processamento.
    * Resultados são armazenados em `st.session_state` e `st.rerun()` atualiza a interface.

**E. Exibição de Resultados:**
    * Selectbox na área principal para escolher qual ticker (dos já analisados) exibir.
    * **Métricas:** Lucro (%, R$), recompensa.
    * **Gráfico:** Simulação da avaliação (`st.pyplot()`).
    * **Tabela Qualitativa:** Resumo de métricas (`st.table()`).
    * **Logs de Processamento:** Em um `st.expander()`.
    * **Resumo Geral:** Se múltiplos tickers foram analisados, mostra tabela comparativa e métricas agregadas.

-------------------------------------------------------------------------------
## Etapa 3.1: Guia de Uso da Interface Streamlit (Fluxo de Usuário)
-------------------------------------------------------------------------------
A interface Streamlit foi projetada para ser intuitiva. Siga estes passos para realizar suas análises:

1.  **Acesso e Carregamento de Dados:**
    * Após iniciar o Streamlit (via `ngrok` no Colab), acesse o link público fornecido no seu navegador.
    * Na **barra lateral esquerda**, você encontrará a seção "⚙️ Configurações e Ações".
    * Clique no botão "1. Carregue seu arquivo CSV (Date, Ticker, OHLCV)". Selecione o seu arquivo de dados.
        * *Nota: O Streamlit geralmente tem um limite de upload de arquivo de cerca de 200MB.*
    * Após o carregamento bem-sucedido, uma mensagem de sucesso e o número de tickers encontrados serão exibidos na barra lateral.

2.  **Análise Individual de Ticker:**
    * Ainda na barra lateral, abaixo do uploader de arquivo:
        * **"2. Escolha um Ticker:"** Selecione o ticker desejado no menu dropdown.
        * **"3. Valor de Investimento Inicial (R$):"** Defina o montante em Reais que você deseja usar como base para o cálculo do lucro monetário desta simulação.
        * **"4. Timesteps para Treinamento (Individual):"** Especifique quantos passos de tempo o modelo deve treinar para este ticker.
    * Clique no botão **"🚀 Analisar Ticker Individual: [Nome do Ticker]"**.
    * Uma mensagem de "processando" (spinner) aparecerá. Aguarde a conclusão (pode levar alguns minutos).
    * Os resultados (gráfico, métricas, tabela, logs) para este ticker serão exibidos na área principal da página.

3.  **Análise em Lote (Todos os Tickers):**
    * Na barra lateral, na seção "Análise em Lote":
        * **"Timesteps para Treinamento (em Lote):"** Defina o número de timesteps de treinamento a ser usado para CADA ticker durante o processamento em lote.
        * **"Reprocessar tickers já analisados nesta sessão?"**: Marque esta caixa se desejar que tickers já analisados anteriormente (nesta mesma sessão do Streamlit) sejam reprocessados. Caso contrário, eles serão pulados para economizar tempo.
    * Clique no botão **"📊 Analisar TODOS os Tickers"**.
    * Na **área principal da página**, uma barra de progresso e mensagens de status indicarão qual ticker está sendo processado. Este processo pode ser bastante demorado.
    * Ao final, todos os resultados estarão disponíveis para visualização individual e no resumo geral.

4.  **Visualização dos Resultados:**
    * Na área principal, use o dropdown **"Mostrar detalhes para o Ticker:"** para alternar entre os resultados dos diferentes tickers que já foram analisados.
    * Para cada ticker, você verá:
        * Lucro/Prejuízo em percentual e em Reais (com base no investimento inicial que você configurou para aquela análise).
        * Recompensa total do agente.
        * O gráfico da simulação do modelo treinado.
        * Uma tabela qualitativa com métricas resumidas.
        * Um expansor "Logs de Processamento" para ver as mensagens de status da análise daquele ticker.
    * Se mais de um ticker foi analisado, uma seção **"Resumo Geral"** aparecerá no final da página, mostrando uma tabela comparativa e métricas agregadas como lucro médio e lucro total em R$.

-------------------------------------------------------------------------------
## Etapa 4: Execução no Google Colab 
-------------------------------------------------------------------------------
(Conteúdo da Etapa 4 como antes, explicando as células de instalação, `%%writefile`, upload de CSV, execução com `ngrok` e resolução do `ERR_NGROK_108`.)

**A. Célula de Instalações:**
    * A primeira célula do notebook deve conter o comando `!pip install ...` (listado na Etapa 1).

**B. Criação dos Arquivos `.py`:**
    * Use o comando mágico `%%writefile nome_do_arquivo.py` no topo de células separadas para salvar o conteúdo do `backtesting_engine.py` e do `app_streamlit_online.py` como arquivos no sistema de arquivos temporário do Colab.

**C. Upload do Arquivo de Dados CSV:**
    * O usuário faz o upload do seu arquivo CSV (ex: `dados_desafio_v5.csv`) para a pasta raiz (`/content/`) do Colab usando o painel "Arquivos", ou diretamente pela interface do Streamlit quando o app estiver rodando.

**D. Célula de Execução do Streamlit com `ngrok`:**
    * Esta célula contém código Python para:
        1.  Importar `pyngrok`.
        2.  Configurar seu authtoken do `ngrok`.
        3.  Usar `os.system("kill ...")` para tentar finalizar processos anteriores.
        4.  Iniciar o servidor Streamlit em segundo plano.
        5.  Usar `ngrok.connect(...)` para criar um túnel público.
        6.  Exibir o link público do `ngrok`.

**E. Resolução de Problemas Comuns no Colab:**
    * **`ERR_NGROK_108`:** Indica limite de 1 sessão `ngrok` simultânea.
        * **Solução:** Verificar painel do `ngrok` (dashboard.ngrok.com/agents) e encerrar sessões ativas; reiniciar sessão do Colab.
        
-------------------------------------------------------------------------------
## Conclusão e Próximos Passos Sugeridos
-------------------------------------------------------------------------------
(Conteúdo da Conclusão como antes, sugerindo otimizações, mais indicadores, etc.)

Este projeto estabelece uma base sólida para um sistema de trading com RL. Possíveis melhorias e próximos passos incluem:

* **Otimização de Hiperparâmetros:** Ajustar os parâmetros do modelo A2C e do ambiente.
* **Mais Indicadores/Features:** Experimentar com diferentes indicadores técnicos ou features alternativas.
* **Funções de Recompensa Avançadas:** Refinar a função de recompensa do ambiente para melhor alinhar com os objetivos de trading (ex: considerar risco, drawdown).
* **Simulação de Custos:** Adicionar custos de transação e slippage ao ambiente.
* **Validação Robusta:** Testar em mais ativos, períodos mais longos e diferentes condições de mercado (out-of-sample, walk-forward optimization).
* **Gerenciamento de Risco:** Implementar lógicas de stop-loss, take-profit, ou dimensionamento de posição.

___________________________________________________________________________________________________________________________________________________________________
