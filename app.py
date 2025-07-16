

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components

from transformers import pipeline

# Importações corrigidas para LangChain e DeepSeek
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings # Usado para embeddings
from langchain_deepseek import ChatDeepSeek # Usar ChatDeepSeek para o LLM



import re
from collections import Counter
# Configuração inicial
st.set_page_config(
    page_title="Dashboard de Produção de Mandioca - Juruti",
    page_icon="🌱",
    layout="wide",
)

# Carregar dados
@st.cache_data
def load_data():
    # Substitua pelo caminho do seu arquivo
    df = pd.read_csv('Backup_Juriti.csv', delimiter=',', encoding='utf-8')
    
    # Verificar e corrigir nomes de colunas
    col_mapping = {
        'Tamanho da Propriedade (ha)': 'Tamanho_Propriedade_ha',
        'Tamanho da área produtiva (ha)': 'Tamanho_Area_Produtiva_ha',
        'Tamanho da área plantada (ha)': 'Tamanho_Area_Plantada_ha',
        'Qual a renda familiar absoluta/mês em R$?': 'Renda_Familiar',
        'Qual(s) variedade(s) de MANDIOCA?': 'Variedades_Mandioca',
        'Com quantos meses colhe a MANDIOCA?': 'Meses_Colheita_Mandioca',
        'Já teve problema com pragas na mandioca/macaxeira???': 'Teve_Problema_Pragas',
        'Se sim, quais produtos são comercializados?': 'Produtos_Comercializados',
        'Onde é comercializado os produtos?': 'Local_Comercializacao',
        'Qual o preço médio de farinha atualmente (kg)?': 'Preco_Farinha',
        'Quais as dificuldades encontradas na COMERCIALIZAÇÃO da farinha e derivados?': 'Dificuldades_Comercializacao',
        'Quais as principais dificuldades no cultivo mandioca/macaxeira ?': 'Dificuldades_Cultivo',
        'Realiza adubação?': 'Adubacao',
        'Quais as principais dificuldades no PROESSAMENTO da mandioca/macaxeira?': 'Dificuldades_Processamento',
        'Recebe algum tipo de assistência técnica?': 'Assistencia_Tecnica',
        'Qual tamanho da área destinada ao plantio de MANDIOCA (ha)?': 'Area_Mandioca_ha',
        'Qual tamanho da área destinada ao plantio de MACAXEIRA (ha)?': 'Area_Macaxeira_ha',
        'Comunidade':'Comunidade'
    }
    
    # Renomear colunas
    for original, new in col_mapping.items():
        if original in df.columns:
            df.rename(columns={original: new}, inplace=True)
    
    return df

df = load_data()

# Pré-processamento
def preprocess_data(df):
    # Converter colunas numéricas
    numeric_cols = [
        'Tamanho_Propriedade_ha', 'Tamanho_Area_Produtiva_ha', 'Tamanho_Area_Plantada_ha',
        'Idade', 'Meses_Colheita_Mandioca', 'Area_Mandioca_ha', 'Area_Macaxeira_ha'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Converter para string e depois para numérico, tratando vírgulas
            df[col] = df[col].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
    
    # Mapear renda familiar
    if 'Renda_Familiar' in df.columns:
        renda_map = {
            'MENOR QUE UM SALÁRIO MÍNIMO': 1000,
            '1 SALÁRIO MÍNIMO': 1630,
            '1 A 2 SALÁRIOS MÍNIMOS': 3260,
            '2 A 3 SALÁRIOS MÍNIMOS': 4890
        }
        df['Renda_Familiar_R$'] = df['Renda_Familiar'].map(renda_map)
        df['Renda_Familiar_R$'] = pd.to_numeric(df['Renda_Familiar_R$'], errors='coerce')
    
    return df

df = preprocess_data(df)
        
# Configuração do sistema RAG


# preparar o terreno para a IA
# Função para gerar contexto detalhado apenas com dados locais
def generate_comprehensive_context(df):
    """Gera contexto estruturado com todas as colunas e estatísticas relevantes"""
    context_lines = []
    
    if df.empty:
        return "Base de dados vazia."
    
    # Informações gerais
    context_lines.append(f"Total de registros: {len(df)}")
    context_lines.append(f"Colunas disponíveis ({len(df.columns)}): {', '.join(df.columns)}")
    
    # Processamento por coluna
    for col in df.columns:
        col_data = df[col].dropna()
        
        if col_data.empty:
            context_lines.append(f"\nColuna: {col} - SEM DADOS")
            continue
            
        # Tipo de dados
        dtype = str(df[col].dtype)
        context_lines.append(f"\nColuna: {col} - Tipo: {dtype}")
        
        # Dados numéricos
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                'Média': col_data.mean(),
                'Mediana': col_data.median(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Desvio Padrão': col_data.std()
            }
            for stat, value in stats.items():
                context_lines.append(f"  {stat}: {value:.2f}")
                
        # Dados categóricos/texto
        else:
            # Contagem de valores únicos
            unique_count = col_data.nunique()
            context_lines.append(f"  Valores únicos: {unique_count}")
            
            # Amostra de valores
            sample_size = min(10, unique_count)
            sample = col_data.sample(sample_size).unique().tolist()
            context_lines.append(f"  Amostra: {sample}")
            
            # Contagem de valores para poucas categorias
            if unique_count <= 20:
                top_values = col_data.value_counts().head(10)
                for value, count in top_values.items():
                    context_lines.append(f"  '{value}': {count} ocorrências")
    
    return "\n".join(context_lines)

@st.cache_resource
def setup_rag_system(df, api_key):
    # Gerar contexto(transformar o dataframe em string)
    local_context = generate_comprehensive_context(df)
    
    # Configuração do embeddings, para entender as relações entre palavras e contextos
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Criar banco vetorial
    vector_db = FAISS.from_texts([local_context], embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
    
    # Configurar as instruções para geração de respostas
    template = """
    Você é um especialista no Projeto Maniva Tapajós em Juruti, Pará.
    Sua função é responder perguntas com base EXCLUSIVAMENTE nos dados fornecidos no contexto.

    Contexto:
    {context}

    Pergunta: {question}

    Instruções:
    - Responda de forma concisa e direta
    - Baseie-se APENAS nas informações do contexto
    - Se a informação não estiver no contexto, diga "Não tenho dados sobre isso"
    - Para perguntas numéricas, forneça valores exatos quando disponíveis
    - Considere que o contexto contém todas as colunas do dataset

    Resposta:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Inicializar modelo DeepSeek
    model = ChatDeepSeek(
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=1000
    )
    
    # Criar cadeia RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    
    return qa_chain

def consultar_rag_sistema(qa_chain, query, df):
    try:
        result = qa_chain({"query": query})
        return {
            "text": result["result"],
            "source": "DeepSeek RAG System"
        }
    except Exception as e:
        return {
            "text": f"⚠️ Erro no sistema RAG: {str(e)}",
            "source": "Sistema"
        }



def render_plot_from_config(plot_config, df):
    import plotly.express as px

    if not plot_config:
        return None

    plot_type = plot_config["type"]
    params = plot_config["params"]

    if plot_type == "histogram":
        return px.histogram(df, **params)
    elif plot_type == "box":
        return px.box(df, **params)
    elif plot_type == "scatter":
        return px.scatter(df, **params)
    elif plot_type == "bar":
        return px.bar(x=params["x"], y=params["y"], title=params["title"], labels=params["labels"])
    elif plot_type == "pie":
        return px.pie(names=params["names"], values=params["values"], title=params["title"])

    return None


# CSS Global para Responsividade

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

st.markdown(""" 
<style>
    /* Ajustes gerais para mobile */
    @media (max-width: 768px) {
        /* KPIs em coluna única */
        .stMetric {
            margin-bottom: 15px;
        }
        
        /* Abas em scroll horizontal */
        div[data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        
        /* Redução de padding */
        .main .block-container {
            padding: 1rem;
        }
        
        /* Ajuste de tamanho de fonte */
        h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.3rem;
        }
    }
    
    /* Ajustes específicos para celulares */
    @media (max-width: 480px) {
        /* Elementos de filtro sidebar */
        .sidebar .stMultiSelect, 
        .sidebar .stSlider, 
        .sidebar .stSelectbox {
            font-size: 14px;
        }
        
        /* Cards de métricas */
        .stMetric {
            padding: 10px;
        }
        
        /* Rank em coluna única */
        .rank-column {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)
# Sidebar - Filtros
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzENcdjez22ijsES4vSml4F-MkUDG88NNXhw&s", use_container_width=True)
st.sidebar.title("Maniva Tapajós")
st.sidebar.markdown("Use os filtros abaixo para explorar os dados")
st.sidebar.header("Filtros")
comunidades = st.sidebar.multiselect(
    "Selecione as comunidades:",
    options=df['Comunidade'].unique(),
    default=df['Comunidade'].unique()
)

idade_min = int(df['Idade'].min()) if 'Idade' in df.columns and not df['Idade'].isnull().all() else 18
idade_max = int(df['Idade'].max()) if 'Idade' in df.columns and not df['Idade'].isnull().all() else 100

idade_range = st.sidebar.slider(
    "Faixa etária:",
    min_value=idade_min,
    max_value=idade_max,
    value=(idade_min, idade_max)
)

if 'Sexo' in df.columns:
    genero = st.sidebar.multiselect(
        "Gênero:",
        options=df['Sexo'].unique(),
        default=df['Sexo'].unique()
    )
else:
    genero = []

# Filtro por tipo de cultivo
if 'Cultiva macaxeira, mandioca ou as duas?' in df.columns:
    tipo_cultivo = st.sidebar.multiselect(
        "Tipo de Cultivo:",
        options=df['Cultiva macaxeira, mandioca ou as duas?'].unique(),
        default=df['Cultiva macaxeira, mandioca ou as duas?'].unique()
    )
else:
    tipo_cultivo = []

# Aplicar filtros
filtered_df = df.copy().replace('N.A.',np.nan)
if comunidades:
    filtered_df = filtered_df[filtered_df['Comunidade'].isin(comunidades)]
if genero and 'Sexo' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Sexo'].isin(genero)]
if tipo_cultivo and 'Cultiva macaxeira, mandioca ou as duas?' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Cultiva macaxeira, mandioca ou as duas?'].isin(tipo_cultivo)]
if 'Idade' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['Idade'].between(idade_range[0], idade_range[1])
    ]

# Layout principal
st.title("🌱 Impacto do Projeto Maniva Tapajós em Juruti")
st.markdown("Este painel analisa os dados coletados de produtores de mandioca e macaxeira na região de Juruti, "
            "focando em métricas que refletem o impacto de iniciativas de desenvolvimento como o Projeto Maniva Tapajós.")

st.markdown('---')
st.title('Dados Gerais')

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Produtores", len(filtered_df))


with col2:
    if 'Quantas pessoas trabalham no cultivo?' in filtered_df.columns:
        sum_cultivo = filtered_df['Quantas pessoas trabalham no cultivo?'].sum()
        st.metric("Pessoas Trabalhando no Cultivo", f"{sum_cultivo}")

with col3:
    if 'Tamanho_Area_Plantada_ha' in filtered_df.columns:
        area_media = filtered_df['Tamanho_Area_Plantada_ha'].mean()
        st.metric("Área Plantada Média (ha)", f"{area_media:.1f}")
    else:
        st.metric("Área Plantada", "Dado indisponível")

with col4:
    if 'Renda_Familiar_R$' in filtered_df.columns:
        renda_media = filtered_df['Renda_Familiar_R$'].mean()
        st.metric("Renda Familiar Média (R$)", f"{renda_media:,.0f}")
    else:
        st.metric("Renda Familiar", "Dado indisponível")

with col5:
    if 'É associado a alguma entidade?' in filtered_df.columns:
        associados = filtered_df['É associado a alguma entidade?'].value_counts().get('SIM', 0)
        percentual = associados/len(filtered_df)*100 if len(filtered_df) > 0 else 0
        st.metric("Associados", f"{associados} ({percentual:.0f}%)")
    else:
        st.metric("Associados", "Dado indisponível")

st.markdown("---")
st.title("Rede do Maniva Tapajós na Região de Juruti")
network_html = """
                <!DOCTYPE html>
                <html lang="pt-br">

                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Rede Maniva Tapajós - D3.js</title>
                    <script src="https://d3js.org/d3.v7.min.js"></script>
                    <style>
                        body {
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f5f5f5;
                            color: #333;
                        }

                        h1 {
                            text-align: center;
                            color: #5d4a36;
                            margin-bottom: 30px;
                        }

                        #filter-container {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            margin-bottom: 20px;
                            padding: 15px;
                            background-color: #fff;
                            border-radius: 8px;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        }

                        #filter-container label {
                            font-weight: bold;
                            margin-right: 10px;
                            color: #5d4a36;
                        }

                        #comunidade-select {
                            padding: 8px 15px;
                            border: 1px solid #c5b8a8;
                            border-radius: 4px;
                            background-color: #f8f4f0;
                            font-size: 16px;
                            color: #5d4a36;
                            cursor: pointer;
                        }

                        #comunidade-select:focus {
                            outline: none;
                            border-color: #a52a2a;
                        }

                        #graph-container {
                            width: 100%;
                            height: 600px;
                            background-color: #f8f4f0;
                            border-radius: 10px;
                            border: 1px solid #c5b8a8;
                            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                            overflow: hidden;
                        }

                        .node {
                            stroke: #fff;
                            stroke-width: 1.5px;
                            cursor: pointer;
                            transition: r 0.2s ease;
                        }

                        .node:hover {
                            stroke-width: 3px;
                        }

                        .link {
                            stroke: #999;
                            stroke-opacity: 0.6;
                        }

                        .label {
                            font-size: 10px;
                            fill: #333;
                            pointer-events: none;
                            text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
                        }

                        .legend {
                            margin-top: 20px;
                            padding: 15px;
                            background-color: #fff;
                            border-radius: 8px;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                            display: flex;
                            flex-wrap: wrap;
                            justify-content: center;
                            gap: 15px;
                        }

                        .legend-item {
                            display: flex;
                            align-items: center;
                            margin: 0 10px;
                        }

                        .legend-color {
                            width: 20px;
                            height: 20px;
                            border-radius: 50%;
                            margin-right: 8px;
                        }

                        .tooltip {
                            position: absolute;
                            padding: 8px 12px;
                            background: rgba(0, 0, 0, 0.8);
                            color: white;
                            border-radius: 4px;
                            pointer-events: none;
                            font-size: 14px;
                            z-index: 10;
                            opacity: 0;
                            transition: opacity 0.3s;
                        }
                        @media (max-width: 768px) {
                            #graph-container {
                                height: 400px; /* Altura reduzida para tablets */
                            }
                            
                            .label {
                                font-size: 8px; /* Rótulos menores */
                            }
                            
                            .legend {
                                flex-direction: column; /* Legenda em coluna */
                                align-items: flex-start;
                                gap: 5px;
                            }
                            
                            .legend-item {
                                margin: 0;
                            }
                            
                            #filter-container {
                                flex-direction: column; /* Filtro em coluna */
                                align-items: flex-start;
                            }
                            
                            #comunidade-select {
                                width: 100%;
                                margin-top: 10px;
                            }
                        }
                        
                        @media (max-width: 480px) {
                            #graph-container {
                                height: 300px; /* Altura ainda menor para celulares */
                            }
                            
                            .node {
                                r: 4; /* Nós menores */
                            }
                            
                            h1 {
                                font-size: 1.2rem; /* Título menor */
                            }
                        }

                    </style>
                </head>

                <body>
                    <h1>Rede Maniva Tapajós</h1>

                    <div id="filter-container">
                        <label for="comunidade-select">Filtrar por Comunidade:</label>
                        <select id="comunidade-select">
                            <option value="Todos">Todas</option>
                            <option value="Comunidade Café torrado">Comunidade Café torrado</option>
                            <option value="Comunidade Maravilha">Comunidade Maravilha</option>
                            <option value="Castanhal">Castanhal</option>
                            <option value="Comunidade Pau Darco">Comunidade Pau Darco</option>
                        </select>
                    </div>

                    <div id="graph-container"></div>

                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #a52a2a;"></div>
                            <span>Maniva Tapajós</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #5d4a36;"></div>
                            <span>Propriedades</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #d2b48c;"></div>
                            <span>APRAS</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #cd853f;"></div>
                            <span>STTR</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #8b4513;"></div>
                            <span>ACORJUVE</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #a0522d;"></div>
                            <span>ACOGLEC</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #d2691e;"></div>
                            <span>ACOJUV</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #bc8f8f;"></div>
                            <span>Sindicato</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #f4a460;"></div>
                            <span>CONJUV</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #808080;"></div>
                            <span>N.A.</span>
                        </div>
                    </div>

                    <div class="tooltip"></div>

                    <script>
                        // Dados da rede
                        const dados = [
                            ["Sítio 7 irmãos", "APRAS", "Comunidade Café torrado"],
                            ["Sítio Campo Verde", "STTR", "Comunidade Café torrado"],
                            ["Sítio Nova Vida", "STTR", "Comunidade Café torrado"],
                            ["Sítio Santa Rosa", "APRAS", "Comunidade Café torrado"],
                            ["Sítio Lobão", "APRAS", "Comunidade Café torrado"],
                            ["Sítio Terra Preta", "N.A.", "Comunidade Café torrado"],
                            ["Sítio Mãe Liuca", "APRAS", "Comunidade Café torrado"],
                            ["Sítio Pimentel", "N.A.", "Comunidade Café torrado"],
                            ["Fazenda Pingo D'Água", "APRAS", "Comunidade Café torrado"],
                            ["Sítio Coelho", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Sucuri", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Boa Esperança", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Bom Pará", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Nova Lembrança", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio São Bento", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Cumaru", "ACORJUVE", "Comunidade Maravilha"],
                            ["Sítio Nova Vida", "ACOGLEC", "Castanhal"],
                            ["Sítio Salbal", "ACOGLEC", "Castanhal"],
                            ["Sítio Baixa da Serra", "ACOGLEC", "Castanhal"],
                            ["Sítio Bom Viver", "ACOGLEC", "Castanhal"],
                            ["Sítio São Raimundo 2", "ACOGLEC", "Castanhal"],
                            ["Nova Vida", "ACOJUV", "Comunidade Pau Darco"],
                            ["Sítio Santa Rosa", "Sindicato do trabalhador", "Comunidade Pau Darco"],
                            ["Sítio só Um", "ACOJUV", "Comunidade Pau Darco"],
                            ["Sítio Arara azul", "ACOJUV", "Comunidade Pau Darco"],
                            ["Sítio Fé em Deus", "CONJUV", "Comunidade Pau Darco"],
                            ["Sítio Boa Esperança", "CONJUV", "Comunidade Pau Darco"]
                        ];

                        // Cores para as organizações
                        const orgColors = {
                            "APRAS": "#d2b48c",
                            "STTR": "#cd853f",
                            "ACORJUVE": "#8b4513",
                            "ACOGLEC": "#a0522d",
                            "ACOJUV": "#d2691e",
                            "SINDICATO DO TRABALHADOR": "#bc8f8f",
                            "CONJUV": "#f4a460",
                            "N.A.": "#808080"
                        };

                        // Configurações do gráfico
                        const width = document.getElementById('graph-container').clientWidth;
                        const height = document.getElementById('graph-container').clientHeight;

                        // Cria o SVG
                        const svg = d3.select("#graph-container")
                            .append("svg")
                            .attr("width", width)
                            .attr("height", height);

                        // Elemento para tooltip
                        const tooltip = d3.select(".tooltip");

                        // Cria o gráfico inicial
                        createGraph("Todos");

                        // Função para criar o gráfico com base na comunidade selecionada
                        function createGraph(selectedComunidade) {
                            // Limpa o SVG
                            svg.selectAll("*").remove();

                            // Inicializa os arrays para nós e links
                            const nodes = [];
                            const links = [];

                            // Adiciona o nó central
                            nodes.push({
                                id: "MANIVA TAPAJÓS",
                                label: "MANIVA TAPAJÓS",
                                size: 30,
                                color: "#a52a2a",
                                type: "central",
                                comunidade: "Todos"
                            });

                            // Processa os dados e adiciona os nós e links
                            dados.forEach(([prop, org, comunidade]) => {
                                const idProp = prop.toUpperCase().trim();
                                const idOrg = org.toUpperCase().trim();

                                // Verifica se deve incluir este nó baseado no filtro
                                if (selectedComunidade !== "Todos" && comunidade !== selectedComunidade) {
                                    return;
                                }

                                // Adiciona o nó da propriedade
                                if (!nodes.find(n => n.id === idProp)) {
                                    nodes.push({
                                        id: idProp,
                                        label: prop,
                                        size: 7,
                                        color: "#5d4a36",
                                        type: "propriedade",
                                        comunidade: comunidade
                                    });
                                }

                                // Adiciona o nó da organização
                                if (!nodes.find(n => n.id === idOrg)) {
                                    nodes.push({
                                        id: idOrg,
                                        label: org,
                                        size: 15,
                                        color: orgColors[idOrg] || "#696969",
                                        type: "organizacao",
                                        comunidade: comunidade
                                    });
                                }

                                // Adiciona os links
                                links.push({
                                    source: idProp,
                                    target: idOrg,
                                    size: 2,
                                    color: "#aaa"
                                });

                                links.push({
                                    source: idProp,
                                    target: "MANIVA TAPAJÓS",
                                    size: 1,
                                    color: "#00db92"
                                });

                                links.push({
                                    source: idOrg,
                                    target: "MANIVA TAPAJÓS",
                                    size: 3,
                                    color: "#ccc"
                                });
                            });

                            // Cria a simulação de força
                            const simulation = d3.forceSimulation(nodes)
                                .force("link", d3.forceLink(links).id(d => d.id).distance(150))
                                .force("charge", d3.forceManyBody().strength(-300))
                                .force("center", d3.forceCenter(width / 2, height / 2))
                                .force("collide", d3.forceCollide().radius(d => d.size + 5));

                            // Desenha os links
                            const link = svg.append("g")
                                .attr("stroke", "#999")
                                .attr("stroke-opacity", 0.6)
                                .selectAll("line")
                                .data(links)
                                .join("line")
                                .attr("stroke-width", d => d.size)
                                .attr("stroke", d => d.color);

                            // Desenha os nós
                            const node = svg.append("g")
                                .attr("stroke", "#fff")
                                .attr("stroke-width", 1.5)
                                .selectAll("circle")
                                .data(nodes)
                                .join("circle")
                                .attr("r", d => d.size)
                                .attr("fill", d => d.color)
                                .attr("class", "node")
                                .call(d3.drag()
                                    .on("start", dragstarted)
                                    .on("drag", dragged)
                                    .on("end", dragended))
                                .on("mouseover", function (event, d) {
                                    // Aumenta o nó
                                    d3.select(this).attr("r", d.size * 1.5);

                                    // Mostra tooltip
                                    tooltip.style("opacity", 1)
                                        .html(`<strong>${d.label}</strong><br>${d.type === "propriedade" ? "Propriedade" : d.type === "organizacao" ? "Organização" : "Central"}<br>Comunidade: ${d.comunidade}`)
                                        .style("left", (event.pageX + 10) + "px")
                                        .style("top", (event.pageY - 28) + "px");
                                })
                                .on("mouseout", function (event, d) {
                                    // Retorna ao tamanho original
                                    d3.select(this).attr("r", d.size);

                                    // Esconde tooltip
                                    tooltip.style("opacity", 0);
                                });

                            // Adiciona rótulos
                            const label = svg.append("g")
                                .attr("class", "labels")
                                .selectAll("text")
                                .data(nodes)
                                .join("text")
                                .attr("class", "label")
                                .text(d => d.label)
                                .attr("font-size", d => d.type === "central" ? 14 : 10)
                                .attr("dx", d => d.type === "central" ? 20 : 12)
                                .attr("dy", ".35em");

                            
                            simulation.on("tick", () => {
                                link
                                    .attr("x1", d => d.source.x)
                                    .attr("y1", d => d.source.y)
                                    .attr("x2", d => d.target.x)
                                    .attr("y2", d => d.target.y);

                                node
                                    .attr("cx", d => d.x)
                                    .attr("cy", d => d.y);

                                label
                                    .attr("x", d => d.x)
                                    .attr("y", d => d.y);
                            });

                            
                            function dragstarted(event, d) {
                                if (!event.active) simulation.alphaTarget(0.3).restart();
                                d.fx = d.x;
                                d.fy = d.y;
                            }

                            function dragged(event, d) {
                                d.fx = event.x;
                                d.fy = event.y;
                            }

                            function dragended(event, d) {
                                if (!event.active) simulation.alphaTarget(0);
                                d.fx = null;
                                d.fy = null;
                            }
                        }

                        // Adiciona o evento de mudança no seletor de comunidade
                        document.getElementById("comunidade-select").addEventListener("change", function () {
                            createGraph(this.value);
                        });
                    </script>
                </body>

                </html>
    """


components.html(network_html, height=700, scrolling=True)

st.title('Navegue Pelos Dados')

st.markdown("""
<style>
    /* Container principal das abas - espaço entre elas */
    div[data-baseweb="tab-list"] {
        gap: 1rem !important;
        justify-content: space-between !important;
    }
    
    /* Abas individuais - tamanho aumentado e destaque */
    button[data-baseweb="tab"] {
        font-size: 1.2rem !important;
        padding: 1rem 1.5rem !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
        flex: 1 !important;
        text-align: center !important;
        border: 1px solid #5D4037 !important;
        background-color: #b9d306 !important;
    }
    
    /* Efeito hover - destaque ao passar o mouse */
    button[data-baseweb="tab"]:hover {
        background-color: #5D4037 !important;
        transform: translateY(-1px) scale(0.9);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Aba selecionada - destaque máximo */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #a0522d !important;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 4px 12px rgba(93, 64, 55, 0.4);
        border: none !important;
    }
    
    /* Ícones dentro das abas */
    .stTabs [data-testid="stMarkdownContainer"] svg {
        width: 24px !important;
        height: 24px !important;
        vertical-align: middle !important;
        margin-right: 8px !important;
    }
    
    .stTabs [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-border"]{
        visibility: hidden
    }
    .stTabs [data-baseweb="tab-highlight"]{
        visibility: hidden
    }
</style>
""", unsafe_allow_html=True)


maniv_ai_tab, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Maniv.IA","👤 Perfil", "🌱 Cultivo", "💰 Comercialização", 
    "⚠️ Desafios", "📊 Dados Completos"
])



TERRACOTA_PALETTE = [
    "#A52A2A",
    "#667755",
    "#8D6E63",  # Marrom terroso
    "#A1887F",  # Marrom claro
    "#CCCCAA",  # Bege
    "#5D4037",  # Marrom escuro
    "#795548",  # Marrom chocolate
    "#BCAAA4",  # Rosa terroso
    "#4E342E",  # Marrom quase preto
    "#3E2723",  # Terracota escuro
    "#6D4C41",  # Terracota
]

# Adicione no início do seu código
import os
import time

with maniv_ai_tab:
    st.markdown("""
    <style>
        .maniva-ai-container {
            padding: 10px;
            height: calc(100vh - 150px);
            display: flex;
            flex-direction: column;
        }
        
        .input-container {
            padding: 15px;
            position: sticky;
            bottom: 0;
            background: white;
            z-index: 100;
        }
       .stTabs [data-testid="stMarkdownContainer"] {
            display:flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .stTabs [data-testid="stMarkdownContainer"] svg {
            width: 300px !important;
            height: 300px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 209 197">
                <defs>
                    <linearGradient id="Gradiente_sem_nome_6" data-name="Gradiente sem nome 6" x1="104.2" y1="20" x2="104.2" y2="165" gradientUnits="userSpaceOnUse">
                        <stop offset=".05" stop-color="#8cc63f"/>
                        <stop offset=".81" stop-color="#edb973"/>
                        <stop offset="1" stop-color="#ff938d"/>
                    </linearGradient>
                    <linearGradient id="Gradiente_sem_nome_199" data-name="Gradiente sem nome 199" x1="173.35" y1="32.29" x2="173.35" y2="59.02" gradientUnits="userSpaceOnUse">
                        <stop offset=".81" stop-color="#edb973"/>
                        <stop offset="1" stop-color="#ff938d"/>
                    </linearGradient>
                    <linearGradient id="Gradiente_sem_nome_199-2" data-name="Gradiente sem nome 199" x1="157.74" y1="36.07" x2="157.74" y2="82.01" xlink:href="#Gradiente_sem_nome_199"/>
                </defs>
                <path fill="url(#Gradiente_sem_nome_6)" d="M112,131.54v2.23s15.62,0,26.77,11.15c11.15,11.15,20.08,20.08,20.08,20.08,0,0-4.46,0-26.77-8.92-22.31-8.92-26.77-20.08-26.77-20.08h-2.23s-4.46,11.15-26.77,20.08c-22.31,8.92-26.77,8.92-26.77,8.92,0,0,8.92-8.92,20.08-20.08s26.77-11.15,26.77-11.15v-2.23s-33.46,4.46-44.62-4.46-8.92-8.92-17.85-13.38-11.15-6.69-11.15-6.69c0,0,22.31-6.69,29-2.23s44.62,24.54,44.62,24.54l2.23-2.23s-31.23-17.85-40.15-42.38c-8.92-24.54-4.46-35.69-4.46-35.69,0,0,2.23,6.69,4.46,8.92s17.85,6.69,20.08,15.62c2.23,8.92,22.31,49.08,22.31,49.08l2.23-2.23s-17.85-42.38-15.62-55.77c2.23-13.38,13.38-22.31,11.15-29s5.58-15.62,5.58-15.62c0,0,7.81,8.92,5.58,15.62-2.23,6.69,8.92,15.62,11.15,29s-15.62,55.77-15.62,55.77l2.23,2.23s20.08-40.15,22.31-49.08c.98-3.9,4.53-6.96,8.39-9.35l7.48,2.77c1.89.7,3.38,2.2,4.08,4.09l2.42,6.56c-.62,2.23-1.38,4.57-2.3,7.07-8.92,24.54-40.15,42.38-40.15,42.38l2.23,2.23s37.92-20.08,44.62-24.54c6.69-4.46,29,2.23,29,2.23c0,0-2.23,2.23-11.15,6.69s-6.69,4.46-17.85,13.38c-11.15,8.92-44.62,4.46-44.62,4.46Z"/>
                <path fill="url(#Gradiente_sem_nome_199)" d="M185.93,46.77l-7.85,2.92c-.33.11-.58.36-.69.69l-1.7,4.57-1.23,3.28c-.38,1.05-1.85,1.05-2.23,0l-1.96-5.29-.96-2.57c-.11-.33-.36-.58-.69-.69l-4.19-1.56-3.66-1.36c-1.05-.38-1.05-1.85,0-2.23l2.05-.76,5.8-2.16c.33-.11.58-.36.69-.69l2.92-7.85c.38-1.05,1.85-1.05,2.23,0l2.92,7.85c.11.33.36.58.69.69l7.85,2.92c1.05.38,1.05,1.85,0,2.23Z"/>
                <path fill="url(#Gradiente_sem_nome_199-2)" d="M178.62,62.05l-11.67,4.31c-.87.33-1.56,1.03-1.9,1.9l-4.31,11.67c-1.05,2.79-4.97,2.79-6.02,0l-1.78-4.82-2.52-6.85c-.33-.87-1.03-1.56-1.9-1.9l-8.57-3.17-3.10-1.14c-2.79-1.05-2.79-4.97,0-6.02l11.67-4.31c.87-.33,1.56-1.03,1.9-1.9l4.31-11.67c1.05-2.79,4.97-2.79,6.02,0l1.54,4.17-3.68,1.36c-1.83.67-1.83,3.26,0,3.93l6.56,2.43c.36.76.98,1.36,1.78,1.67l1.41.51,3.03,8.16c.67,1.83,3.26,1.83,3.93,0l1.83-4.91,1.47.54c2.79,1.05,2.79,4.97,0,6.02Z"/>
                <text font-family="Montserrat" font-weight="900" fill="#8cc63f" font-size="48" transform="translate(0 177.7)">
                    <tspan x="0" y="0">Mani</tspan>
                    <tspan letter-spacing="-0.03em" x="125.81" y="0">v</tspan>
                    <tspan x="155.28" y="0">AI</tspan>
                </text>
            </svg>
            <p style="color: #6d4c41; margin:0">Assistente digital do Projeto Maniva Tapajós</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("Pergunte sobre os dados do Projeto Maniva Tapajós em Juruti, Pará. O chatbot usará informações da base de dados fornecida para responder.")

    # Input para a API Key do DeepSeek
    deepseek_api_key = st.text_input("Insira sua DeepSeek API Key", type="password", key="deepseek_api_key_input")

    # Inicializar o sistema RAG apenas se a API Key for fornecida e não vazia
    qa_chain = None
    if deepseek_api_key:
        with st.spinner("Configurando sistema RAG..."):
            try:
                qa_chain = setup_rag_system(df, deepseek_api_key)
                st.success("Sistema RAG configurado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao configurar o sistema RAG. Verifique sua API Key e conexão: {e}")
    else:
        st.info("Por favor, insira sua DeepSeek API Key para ativar o chatbot.")

    # Inicializar histórico de chat
    if "web_chat_history" not in st.session_state:
        st.session_state.web_chat_history = [
            {"role": "assistant", "content": "Olá! Sou o Maniv.IA, seu assistente para o Projeto Maniva Tapajós. Como posso ajudar com os dados hoje?"}
        ]

        # Exibir histórico
        for msg in st.session_state.web_chat_history:
            # Garante que 'msg' é um dicionário e possui a chave 'role' antes de tentar acessá-la
            if isinstance(msg, dict) and "role" in msg:
                if msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                        if "plot_config" in msg and msg["plot_config"]:
                            fig = render_plot_from_config(msg["plot_config"], df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                elif msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])


    # Input container
    with st.form(key='chat_form', clear_on_submit=True):
        prompt = st.text_area("Digite sua pergunta:", key="input", height=100)
        submitted = st.form_submit_button("Enviar")

        if submitted and prompt:
            if not deepseek_api_key:
                st.warning("Por favor, insira sua DeepSeek API Key para conversar com o chatbot.")
            elif not qa_chain:
                st.warning("O sistema RAG ainda não foi configurado ou houve um erro. Por favor, verifique a API Key.")
            else:
                st.session_state.web_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Processando..."):
                    response = consultar_rag_sistema(qa_chain, prompt, df)
                    st.markdown(response["text"])
                    st.session_state.web_chat_history.append(response)

                    
    
with tab1:
    st.subheader("Perfil dos Produtores")
    
    if 'Possui Cadastro Ambiental Rural (CAR)?' in filtered_df.columns:
        car_count = filtered_df['Possui Cadastro Ambiental Rural (CAR)?'].value_counts()
        fig = px.bar(car_count,
                     title="Registro de CAR Entre os Produtores",
                     labels={'index': 'Registro de CAR', 'value':'Contagem'},
                     orientation='h',
                     color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
                     )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if 'Sexo' in filtered_df.columns:
            fig = px.pie(
                filtered_df, 
                names='Sexo',
                title='Distribuição por Gênero',
                color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de gênero não disponíveis")
        
        if 'Escolaridade' in filtered_df.columns:
            escolaridade_counts = filtered_df['Escolaridade'].value_counts()
            fig = px.bar(
                escolaridade_counts,
                title='Nível de Escolaridade',
                labels={'index': 'Escolaridade', 'value': 'Contagem'},
                orientation='h',
                color_discrete_sequence=[TERRACOTA_PALETTE[0]]  # Cor principal
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de escolaridade não disponíveis")
    
    with col2:
        if 'Idade' in filtered_df.columns:
            fig = px.histogram(
                filtered_df, 
                x='Idade',
                nbins=10,
                title='Distribuição Etária',
                color='Sexo' if 'Sexo' in filtered_df.columns else None,
                color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de idade não disponíveis")
        
        if 'É associado a alguma entidade?' in filtered_df.columns:
            associacao_counts = filtered_df['É associado a alguma entidade?'].value_counts()
            fig = px.pie(
                associacao_counts,
                names=associacao_counts.index,
                title='Associação a Entidades',
                color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de associação não disponíveis")

with tab2:
    st.subheader("Práticas de Cultivo")
    
    # Bubble Chart
    if 'Area_Mandioca_ha' in filtered_df.columns and 'Area_Macaxeira_ha' in filtered_df.columns:
        filtered_df['Area_Total_ha'] = filtered_df['Area_Mandioca_ha'] + filtered_df['Area_Macaxeira_ha']
        
        fig = px.scatter(
            filtered_df,
            x='Area_Mandioca_ha',
            y='Area_Macaxeira_ha',
            size='Area_Total_ha',
            color='Comunidade',
            hover_name='Nome da propriedade',
            title='Relação entre Área de Mandioca e Macaxeira',
            labels={
                'Area_Mandioca_ha': 'Área de Mandioca (ha)',
                'Area_Macaxeira_ha': 'Área de Macaxeira (ha)',
                'Area_Total_ha': 'Área Total (ha)'
            },
            size_max=50,
            color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Dados de área plantada específica não disponíveis") 
    
    # RANK 
    container_rank = st.container(height=600)
    with container_rank:
        
        
        # Layout modificado com classe
        st.markdown('<div class="rank-column" style="display:flex; gap:20px;">', unsafe_allow_html=True)
        # Cálculo da área total
        filtered_df['Area_Total_ha'] = filtered_df['Area_Mandioca_ha'] + filtered_df['Area_Macaxeira_ha']
        
        st.header('Rank dos Sítios por Área Plantada')
        
        # Dropdown para seleção do tipo de ranking
        ranking_option = st.selectbox(
            'Selecione o ranking:',
            options=['Top 5', 'Top 10', 'Todos'],
            index=0
        )
        
        # Ordena o DataFrame
        sorted_df = filtered_df.sort_values(by='Area_Total_ha', ascending=False)
        
        # Aplica o filtro
        if ranking_option == 'Top 5':
            ranked_df = sorted_df.head(5)
        elif ranking_option == 'Top 10':
            ranked_df = sorted_df.head(10)
        else:
            ranked_df = sorted_df
        
        # CSS para estilização
        st.markdown("""
        <style>
            .gold {
                background-color: #FFD700 !important;
                color: #000;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
            .silver {
                background-color: #C0C0C0 !important;
                color: #000;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
            .bronze {
                background-color: #CD7F32 !important;
                color: #000;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
            .normal {
                background-color: #f0f2f6;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
                color: #000
            }
            .rank-header {

                font-weight: bold;
                margin-bottom: 10px;
            }
        </style>
        """, unsafe_allow_html=True)
        # CSS para mobile
        st.markdown("""
        <style>
            @media (max-width: 768px) {
                .rank-column {
                    flex-direction: column !important;
                    gap: 10px;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Cria colunas
        col_propriedades, col_area, col_comunidade = st.columns(3)
        
        with col_propriedades:
            st.markdown('<p class="rank-header">Propriedade</p>', unsafe_allow_html=True)
            for i, (_, row) in enumerate(ranked_df.iterrows(), start=1):
                css_class = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else "normal"
                st.markdown(f'<div class="{css_class}">{i}º - {row["Nome da propriedade"]}</div>', unsafe_allow_html=True)
                
        with col_area:
            st.markdown('<p class="rank-header">Área Total (ha)</p>', unsafe_allow_html=True)
            for i, (_, row) in enumerate(ranked_df.iterrows(), start=1):
                css_class = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else "normal"
                st.markdown(f'<div class="{css_class}">{row["Area_Total_ha"]:.2f}</div>', unsafe_allow_html=True)
        
        with col_comunidade:
            st.markdown('<p class="rank-header">Comunidade</p>', unsafe_allow_html=True)
            for i, (_, row) in enumerate(ranked_df.iterrows(), start=1):
                css_class = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else "normal"
                st.markdown(f'<div class="{css_class}">{i}º - {row["Comunidade"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        

    # MEDIAS
    st.subheader('Média das Áreas')
    col_media_mandioca,col_media_macaxeira,col_total_media = st.columns(3)
    with col_media_macaxeira:
        if 'Area_Macaxeira_ha' in filtered_df.columns:
            # Criar coluna de área total para o tamanho das bolhas
            media_macaxeira = filtered_df['Area_Macaxeira_ha']
            media_formatada_macaxeira = f'{media_macaxeira.mean():.3f}'
            st.metric('Média total de Área Plantada de Macaxeira', media_formatada_macaxeira)
            
        else:
            st.warning("Dados de área plantada específica não disponíveis")
            
    
    with col_media_mandioca:
        if 'Area_Mandioca_ha' in filtered_df.columns:
            # Criar coluna de área total para o tamanho das bolhas
            media_mandioca = filtered_df['Area_Mandioca_ha']
            media_formatada_mandioca = f'{media_mandioca.mean():.3f}'
            st.metric('Média total de Área Plantada de Mandioca', media_formatada_mandioca)
            
        else:
            st.warning("Dados de área plantada específica não disponíveis")
    with col_total_media:
        if 'Area_Mandioca_ha' in filtered_df.columns and 'Area_Macaxeira_ha' in filtered_df.columns:
            # Criar coluna de área total para o tamanho das bolhas
            filtered_df['Area_Total_ha'] = filtered_df['Area_Mandioca_ha'] + filtered_df['Area_Macaxeira_ha']
            totaldf =filtered_df['Area_Total_ha']
            total_media_formata = totaldf.mean()
            media_formatada_total = f'{total_media_formata.mean():.3f}'
            st.metric('Média total de Área Plantada', media_formatada_total)
            
            
        else:
            st.warning("Dados de área plantada específica não disponíveis")
    

    
    st.subheader("Variedades")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if 'Variedades_Mandioca' in filtered_df.columns:
            try:
                mandioca_variedades = filtered_df['Variedades_Mandioca'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    mandioca_variedades.head(10),
                    title='Variedades de Mandioca Mais Cultivadas',
                    labels={'index': 'Variedade', 'value': 'Contagem'},
                    color_discrete_sequence=[TERRACOTA_PALETTE[3]]  # Nova cor
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar variedades de mandioca")
        else:
            st.warning("Dados de variedades de mandioca não disponíveis")
            
    with col2:
        if 'Qual(s) variedade(s) de MACAXEIRA?' in filtered_df.columns:
            try:
                macaxeira_variedades = filtered_df['Qual(s) variedade(s) de MACAXEIRA?'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    macaxeira_variedades.head(10),
                    title='Variedades de Macaxeira Mais Cultivadas',
                    labels={'index': 'Variedade', 'value': 'Contagem'},
                    color_discrete_sequence=[TERRACOTA_PALETTE[0]]  # Nova cor
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar variedades de macaxeira")
        else:
            st.warning("Dados de variedades de macaxeira não disponíveis")

with tab3:
    st.subheader("Comercialização e Processamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Produtos_Comercializados' in filtered_df.columns:
            try:
                produtos = filtered_df['Produtos_Comercializados'].str.split(', ', expand=True).stack().value_counts()
                fig = px.bar(
                    produtos,
                    title='Produtos Derivados Comercializados',
                    labels={'index': 'Produto', 'value': 'Contagem'},
                    color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar produtos comercializados")
        else:
            st.warning("Dados de produtos comercializados não disponíveis")
        
        st.subheader("Principais Compradores")
        if 'Com quem comercializa os produtos ?' in filtered_df.columns and not filtered_df['Com quem comercializa os produtos ?'].dropna().empty:
            compradores = filtered_df['Com quem comercializa os produtos ?'].dropna().str.split(',').explode().str.strip().str.title().value_counts()
            fig_compradores = px.pie(
                compradores, 
                names=compradores.index, 
                values=compradores.values, 
                title="Para Quem os Produtores Vendem?", 
                hole=0.4,
                color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
            )
            st.plotly_chart(fig_compradores, use_container_width=True)
        else:
            st.warning("Dados de locais de comercialização não disponíveis")
    
    with col2:
        if 'Preco_Farinha' in filtered_df.columns:
            try:
                precos = filtered_df['Preco_Farinha'].apply(lambda x: pd.to_numeric(str(x).replace(',', '.'), errors='coerce'))
                precos = precos.dropna()
                
                if not precos.empty:
                    fig = px.histogram(
                        precos,
                        title='Distribuição de Preços da Farinha (R$/kg)',
                        labels={'value': 'Preço (R$/kg)'},
                        color_discrete_sequence=[TERRACOTA_PALETTE[4]]  # Nova cor
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Nenhum dado numérico válido para preços")
            except:
                st.warning("Erro ao processar preços da farinha")
        else:
            st.warning("Dados de preço da farinha não disponíveis")
        
        st.subheader("Dificuldades na Comercialização")
        if 'Dificuldades_Comercializacao' in filtered_df.columns:
            try:
                dificuldades = filtered_df['Dificuldades_Comercializacao'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    dificuldades,
                    title='Dificuldades na Comercialização',
                    labels={'index': 'Dificuldade', 'value': 'Contagem'},
                    color_discrete_sequence=[TERRACOTA_PALETTE[2]]  # Nova cor
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar dificuldades de comercialização")
        else:
            st.warning("Dados de dificuldades na comercialização não disponíveis")

with tab4:
    st.subheader("Dificuldades no Cultivo")
    
    if 'Dificuldades_Cultivo' in filtered_df.columns:
        try:
            cultivo_dificuldades = filtered_df['Dificuldades_Cultivo'].str.split(',', expand=True).stack().value_counts()
            fig = px.bar(
                cultivo_dificuldades,
                title='Dificuldades no Cultivo',
                labels={'index': 'Dificuldade', 'value': 'Contagem'},
                color_discrete_sequence=[TERRACOTA_PALETTE[1]]  # Nova cor
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Erro ao processar dificuldades no cultivo")
    else:
        st.warning("Dados de dificuldades no cultivo não disponíveis")
    
    if 'Assistencia_Tecnica' in filtered_df.columns:
        assistencia = filtered_df['Assistencia_Tecnica'].value_counts()
        fig = px.pie(
            assistencia,
            names=assistencia.index,
            title='Acesso à Assistência Técnica',
            color_discrete_sequence=TERRACOTA_PALETTE  # Nova cor
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados de assistência técnica não disponíveis")
    
    if 'Dificuldades_Processamento' in filtered_df.columns:
        try:
            processamento_dificuldades = filtered_df['Dificuldades_Processamento'].str.split(',', expand=True).stack().value_counts()
            fig = px.bar(
                processamento_dificuldades,
                title='Dificuldades no Processamento',
                labels={'index': 'Dificuldade', 'value': 'Contagem'},
                color_discrete_sequence=[TERRACOTA_PALETTE[5]]  # Nova cor
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Erro ao processar dificuldades no processamento")
    else:
        st.warning("Dados de dificuldades no processamento não disponíveis")
    if 'Se sim, quais pragas?' in filtered_df:
        pragas = filtered_df['Se sim, quais pragas?'].str.replace(', ',',').str.replace(' E ',',').str.replace('LARGATA','LAGARTA').str.upper().str.split(',').explode().str.strip().value_counts()
        fig = px.bar(
            pragas,
            title='Incidência de Pragas',
            labels={'index': 'Pragas', 'value': 'Contagem'},
            color_discrete_sequence=TERRACOTA_PALETTE)
        st.plotly_chart(fig,use_container_width=True)
with tab5:
    
    start_col = 1  # Skip first column (index 0)
    end_col = -8   # Skip last 8 columns
    filtered_df_to_show = filtered_df.iloc[:, start_col:end_col]
    st.subheader("Dados Completos")
    st.dataframe(filtered_df_to_show, height=600)
    
    
    # Botão para download
    csv = filtered_df_to_show.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Baixar dados filtrados (CSV)",
        data=csv,
        file_name='dados_mandioca_filtrados.csv',
        mime='text/csv'
    )

# Rodapé
st.markdown("---")
st.caption("Dashboard de Produção de Mandioca e Macaxeira em Juruti - Dados coletados em 2025 | Maniva Tapajós | LABCRIA")