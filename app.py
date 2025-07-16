import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Título do app
st.title("Análise de Probabilidade de Retorno e WOE - Ano de 2020")

# Drag and drop ou botão de upload
st.markdown("### Upload do arquivo de dados (.pkl)")
data_file = st.file_uploader("Arraste ou selecione um arquivo", type=["pkl"])

@st.cache_data(show_spinner=False)
def load_pickle(file):
    return pd.read_pickle(file)

if data_file is not None:
    # Carregar dados com cache
    df = load_pickle(data_file)
    st.success("Arquivo carregado com sucesso!")

    # Seleção das colunas de interesse
    st.markdown("### Selecione as colunas de referência")
    colunas = df.columns.tolist()
    ano_col = st.selectbox("Coluna que representa o ano (ou selecione 'Nenhum')", options=["Nenhum"] + colunas)
    retorno_col = st.selectbox("Coluna que representa o retorno (target binário)", options=colunas)

    # Filtro para o ano de 2020
    if ano_col != "Nenhum":
        try:
            df_2020 = df[df[ano_col] == 2020].copy()
            st.write(f"Total de registros em 2020: {len(df_2020)}")
        except Exception as e:
            st.warning(f"Erro ao filtrar ano: {e}")
            df_2020 = df.copy()
    else:
        df_2020 = df.copy()
        st.info("Análise feita com todos os anos, pois nenhuma coluna de ano foi selecionada.")

    # Exibição inicial da base
    st.subheader("Prévia dos dados")
    st.dataframe(df_2020.head())

    # Seletor de variável para análise de retorno
    var = st.selectbox("Escolha a variável explicativa para análise da probabilidade de retorno:",
                       [col for col in df_2020.columns if col != retorno_col])

    # Plot da taxa de retorno
    st.subheader("Probabilidade de Retorno")

    @st.cache_data(show_spinner=False)
    def calcular_probabilidade(df, var, retorno_col):
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        return sample_df.groupby(var)[retorno_col].mean().reset_index(), len(sample_df)

    try:
        prob_df, n_amostra = calcular_probabilidade(df_2020, var, retorno_col)
        if prob_df.empty:
            st.warning("Não há dados suficientes para gerar o gráfico de probabilidade de retorno.")
        else:
            sns.set(style="whitegrid")
            fig, ax = plt.subplots()
            sns.barplot(data=prob_df, x=var, y=retorno_col, ax=ax, palette='viridis')
            ax.set_ylabel('Taxa média de retorno')
            ax.set_xlabel(var)
            ax.set_title(f"Probabilidade média de retorno por {var} (amostra de {n_amostra} registros)")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Erro ao gerar gráfico: {e}")

    # Cálculo do WOE
    st.subheader("Cálculo do WOE")
    woe_var = st.selectbox("Escolha a variável para cálculo do WOE:",
                           [col for col in df_2020.columns if col != retorno_col])

    @st.cache_data(show_spinner=False)
    def calcular_woe(df, var, target):
        woe_df = df[[var, target]].copy()
        total_event = (woe_df[target] == 1).sum()
        total_non_event = (woe_df[target] == 0).sum()

        woe_table = []
        for val in woe_df[var].dropna().unique():
            sub = woe_df[woe_df[var] == val]
            event = (sub[target] == 1).sum()
            non_event = (sub[target] == 0).sum()

            if event == 0 or non_event == 0:
                woe_value = None
            else:
                woe_value = np.log((event / total_event) / (non_event / total_non_event))

            woe_table.append({
                var: val,
                'WOE': woe_value,
                'n': len(sub),
                'event_rate': event / len(sub)
            })

        return pd.DataFrame(woe_table)

    try:
        woe_result = calcular_woe(df_2020, woe_var, retorno_col)
        st.dataframe(woe_result)

        # Exportar resultados
        csv = woe_result.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="woe_result.csv">🔗 Baixar CSV com resultados</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Erro no cálculo de WOE: {e}")