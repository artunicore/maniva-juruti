

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import streamlit.components.v1 as components



# Configuração inicial
st.set_page_config(
    page_title="Dashboard de Produção de Mandioca - Juruti",
    page_icon="🌱",
    layout="wide"
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
        'Qual tamanho da área destinada ao plantio de MACAXEIRA (ha)?': 'Area_Macaxeira_ha'
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
            '1 SALÁRIO MÍNIMO': 1300,
            '1 A 2 SALÁRIOS MÍNIMOS': 2500,
            '2 A 3 SALÁRIOS MÍNIMOS': 4000
        }
        df['Renda_Familiar_R$'] = df['Renda_Familiar'].map(renda_map)
        df['Renda_Familiar_R$'] = pd.to_numeric(df['Renda_Familiar_R$'], errors='coerce')
    
    return df

df = preprocess_data(df)

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
filtered_df = df.copy()
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
                <html>

                <head>
                    <meta charset="UTF-8" />
                    <title>Rede Maniva Tapajós</title>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/sigma.js/2.4.0/sigma.min.js"></script>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/graphology/0.25.4/graphology.umd.min.js"></script>

                    <style>
                       body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }

                    #filter-container {
                        display: flex;
                        margin-bottom: 10px;
                        justify-content: center;
                        align-items: center;
                    }
                    select {
                    border: 2px solid #ddd;
                    background: #eee;
                    padding: 10px;
                    transition: 0.4s;
                    }


                    select,
                    ::picker(select) {
                        appearance: base-select;
                    }
                    select:hover,
                    select:focus {
                    background: #ddd;
                    }
                    
                    select::picker-icon {
                    color: #999;
                    transition: 0.4s rotate;
                    }
                    
                    select:open::picker-icon {
                    rotate: 180deg;
                    }

                    #sigma-container {
                        height: 600px;
                        background-color: #f8f4f0;
                        border-radius: 10px;
                        border: 1px solid #c5b8a8;
                    }
                    </style>
                </head>

                <body>

                    <div id="filter-container">
                        <label for="comunidade-select"><strong>Filtrar por Comunidade:</strong></label>
                        <select id="comunidade-select">
                            <option value="Todos">Todas</option>
                            <option value="Comunidade Café torrado">Comunidade Café torrado</option>
                            <option value="Comunidade Maravilha">Comunidade Maravilha</option>
                            <option value="Castanhal">Castanhal</option>
                            <option value="Comunidade Pau Darco">Comunidade Pau Darco</option>
                        </select>
                    </div>

                    <div id="sigma-container"></div>

                    <script>
                        const graph = new graphology.Graph();
                        const container = document.getElementById("sigma-container");

                        // Nó central
                        graph.addNode("MANIVA TAPAJÓS", {
                            label: "MANIVA TAPAJÓS",
                            x: 0,
                            y: 0,
                            size: 30,
                            color: "#a52a2a"
                        });

                        // Dados das propriedades, organizações e comunidades
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
                            ["Sítio Santa Rosa", "Sindicato do trabalhador", "Comunidade Pau Darco"],
                            ["Sítio só Um", "Acojuv", "Comunidade Pau Darco"],
                            ["Sítio Arara azul", "Acojuv", "Comunidade Pau Darco"],
                            ["Sítio Fé em Deus", "Conjuv", "Comunidade Pau Darco"],
                            ["Sítio Boa Esperança", "Conjuv", "Comunidade Pau Darco"]
                        ];

                        const comunidadeMap = {};
                        const orgColors = {
                            "APRAS": "#d2b48c",
                            "STTR": "#cd853f",
                            "ACORJUVE": "#8b4513",
                            "ACOGLEC": "#a0522d",
                            "ACOJUV": "#d2691e",
                            "ACOJUV": "#d2691e",
                            "SINDICATO DO TRABALHADOR": "#bc8f8f",
                            "CONJUV": "#f4a460",
                            "ACT": "#daa520",
                            "N.A.": "#808080"
                        };

                        // Coordenadas base por comunidade
                        const comunidadeOffset = {
                            "Comunidade Café torrado": [-100, 10],
                            "Comunidade Maravilha": [100, 10],
                            "Castanhal": [10, -100],
                            "Comunidade Pau Darco": [10, 100]
                        };

                        dados.forEach(([prop, org, comunidade]) => {
                            const idProp = prop.toUpperCase().trim();
                            const idOrg = org.toUpperCase().trim();

                            comunidadeMap[idProp] = comunidade;

                            const [baseX, baseY] = comunidadeOffset[comunidade] || [0, 0];
                            const jitterX = Math.random() * 100 - 50;
                            const jitterY = Math.random() * 100 - 50;

                            // Adiciona o nó da propriedade
                            if (!graph.hasNode(idProp)) {
                                graph.addNode(idProp, {
                                    label: prop,
                                    size: 7,
                                    x: baseX + jitterX,
                                    y: baseY + jitterY,
                                    color: "#5d4a36"
                                });
                            }

                            // Adiciona o nó da organização
                            if (!graph.hasNode(idOrg)) {
                                graph.addNode(idOrg, {
                                    label: org,
                                    size: 15,
                                    x: baseX + jitterX / 12,
                                    y: baseY + jitterY / 2,
                                    color: orgColors[idOrg] || "#696969"
                                });
                            }

                            // Propriedade → Organização
                            if (!graph.hasEdge(idProp, idOrg)) {
                                graph.addEdge(idProp, idOrg, { size: 2, color: "#aaa" });
                            }

                            // Propriedade → MANIVA
                            if (!graph.hasEdge(idProp, "MANIVA TAPAJÓS")) {
                                graph.addEdge(idProp, "MANIVA TAPAJÓS", { size: 1, color: "#00db92" });
                            }

                            // Organização → MANIVA
                            if (!graph.hasEdge(idOrg, "MANIVA TAPAJÓS")) {
                                graph.addEdge(idOrg, "MANIVA TAPAJÓS", { size: 3, color: "#ccc" });
                            }
                        });

                        // Renderiza com Sigma.js
                        const renderer = new Sigma(graph, container);

                        // Filtro por comunidade
                        const select = document.getElementById("comunidade-select");
                        select.addEventListener("change", () => {
                            const value = select.value;

                            graph.forEachNode((node) => {
                                if (node === "MANIVA TAPAJÓS" || !comunidadeMap[node]) {
                                    graph.setNodeAttribute(node, "hidden", false);
                                    return;
                                }

                                const comunidade = comunidadeMap[node];
                                const visible = value === "Todos" || comunidade === value;
                                graph.setNodeAttribute(node, "hidden", !visible);
                            });

                            graph.forEachEdge((edge, _, source, target) => {
                                const visible = !graph.getNodeAttribute(source, "hidden") && !graph.getNodeAttribute(target, "hidden");
                                graph.setEdgeAttribute(edge, "hidden", !visible);
                            });
                        });
                    </script>
                </body>

                </html>
    """

components.html(network_html, height=700, scrolling=True)

# Legenda
st.markdown("""
                <div style="margin-top:20px; background-color:#f0e6d8; padding:15px; border-radius:10px; border:1px solid #c5b8a8">
                    <h4 style="color:#5d4a36; margin-top:0">Legenda:</h4>
                    <div style="display:flex; flex-wrap:wrap; gap:15px; justify-content: space-between">
                        <div style="display:flex; align-items:center;">
                            <div style="width:15px; height:15px; background-color:#a52a2a; border-radius:50%; margin-right:8px"></div>
                            <span>Maniva Tapajós</span>
                        </div>
                        <div style="display:flex; align-items:center">
                            <div style="width:15px; height:15px; background-color:#d2b48c; border-radius:50%; margin-right:8px"></div>
                            <span>APRAS</span>
                        </div>
                        <div style="display:flex; align-items:center">
                            <div style="width:15px; height:15px; background-color:#cd853f; border-radius:50%; margin-right:8px"></div>
                            <span>STTR</span>
                        </div>
                        <div style="display:flex; align-items:center">
                            <div style="width:15px; height:15px; background-color:#8b4513; border-radius:50%; margin-right:8px"></div>
                            <span>ACORJUVE</span>
                        </div>
                        <div style="display:flex; align-items:center">
                            <div style="width:15px; height:15px; background-color:#a0522d; border-radius:50%; margin-right:8px"></div>
                            <span>ACOGLEC</span>
                        </div>
                        <div style="display:flex; align-items:center">
                            <div style="width:15px; height:15px; background-color:#5d4a36; border-radius:50%; margin-right:8px"></div>
                            <span>Propriedades</span>
                        </div>
                    </div>
                </div>
    """, unsafe_allow_html=True)


# Abas para diferentes seções
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Perfil", "🌱 Cultivo", "💰 Comercialização", 
    "⚠️ Desafios", "📊 Dados Completos"
])

with tab1:
    st.subheader("Perfil dos Produtores")
    
    
    if 'Possui Cadastro Ambiental Rural (CAR)?' in filtered_df.columns:
        car_count = filtered_df['Possui Cadastro Ambiental Rural (CAR)?'].value_counts()
        fig = px.bar(car_count,
                     title="Registro de CAR Entre os Produtores",
                     labels={'index': 'Registro de CAR', 'value':'Contagem'},
                     orientation='h'
                     )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if 'Sexo' in filtered_df.columns:
            fig = px.pie(
                filtered_df, 
                names='Sexo',
                title='Distribuição por Gênero'
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
                color='Sexo' if 'Sexo' in filtered_df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de idade não disponíveis")
        
        if 'É associado a alguma entidade?' in filtered_df.columns:
            associacao_counts = filtered_df['É associado a alguma entidade?'].value_counts()
            fig = px.pie(
                associacao_counts,
                names=associacao_counts.index,
                title='Associação a Entidades'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de associação não disponíveis")

with tab2:
    st.subheader("Práticas de Cultivo")
    
    # Bubble Chart - Relação entre área de mandioca e macaxeira
    if 'Area_Mandioca_ha' in filtered_df.columns and 'Area_Macaxeira_ha' in filtered_df.columns:
        # Criar coluna de área total para o tamanho das bolhas
        filtered_df['Area_Total_ha'] = filtered_df['Area_Mandioca_ha'] + filtered_df['Area_Macaxeira_ha']
        
        fig = px.scatter(
            filtered_df,
            x='Area_Mandioca_ha',
            y='Area_Macaxeira_ha',
            size='Area_Total_ha',
            color='Comunidade',
            hover_name='Nome produtor (entrevistado)',
            title='Relação entre Área de Mandioca e Macaxeira',
            labels={
                'Area_Mandioca_ha': 'Área de Mandioca (ha)',
                'Area_Macaxeira_ha': 'Área de Macaxeira (ha)',
                'Area_Total_ha': 'Área Total (ha)'
            },
            size_max=50
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados de área plantada específica não disponíveis")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Variedades de mandioca
        if 'Variedades_Mandioca' in filtered_df.columns:
            try:
                mandioca_variedades = filtered_df['Variedades_Mandioca'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    mandioca_variedades.head(10),
                    title='Variedades de Mandioca Mais Cultivadas',
                    labels={'index': 'Variedade', 'value': 'Contagem'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar variedades de mandioca")
        else:
            st.warning("Dados de variedades de mandioca não disponíveis")
            
    with col2:
        # Variedades de macaxeira
        if 'Qual(s) variedade(s) de MACAXEIRA?' in filtered_df.columns:
            try:
                macaxeira_variedades = filtered_df['Qual(s) variedade(s) de MACAXEIRA?'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    macaxeira_variedades.head(10),
                    title='Variedades de Macaxeira Mais Cultivadas',
                    labels={'index': 'Variedade', 'value': 'Contagem'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar variedades de macaxeira")
        else:
            st.warning("Dados de variedades de macaxeira não disponíveis")
    
    # Gráficos de tempo de colheita
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        # Tempo de colheita da mandioca
        if 'Meses_Colheita_Mandioca' in filtered_df.columns:
            fig = px.box(
                filtered_df, 
                y='Meses_Colheita_Mandioca',
                title='Tempo de Colheita da Mandioca (meses)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de tempo de colheita da mandioca não disponíveis")
    
    with col4:
        # Tempo de colheita da macaxeira
        if 'Quantos meses colhe a MACAXEIRA?' in filtered_df.columns:
            fig = px.box(
                filtered_df, 
                y='Quantos meses colhe a MACAXEIRA?',
                title='Tempo de Colheita da Macaxeira (meses)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dados de tempo de colheita da macaxeira não disponíveis")

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
                    labels={'index': 'Produto', 'value': 'Contagem'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar produtos comercializados")
        else:
            st.warning("Dados de produtos comercializados não disponíveis")
        
        st.subheader("Principais Compradores")
        if 'Com quem comercializa os produtos ?' in filtered_df.columns and not filtered_df['Com quem comercializa os produtos ?'].dropna().empty:
            compradores = filtered_df['Com quem comercializa os produtos ?'].dropna().str.split(',').explode().str.strip().str.title().value_counts()
            fig_compradores = px.pie(compradores, names=compradores.index, values=compradores.values, title="Para Quem os Produtores Vendem?", hole=0.4)
            st.plotly_chart(fig_compradores, use_container_width=True)
        else:
            st.warning("Dados de locais de comercialização não disponíveis")
    
    with col2:
        if 'Preco_Farinha' in filtered_df.columns:
            try:
                # Converter para numérico e remover não numéricos
                precos = filtered_df['Preco_Farinha'].apply(lambda x: pd.to_numeric(str(x).replace(',', '.'), errors='coerce'))
                precos = precos.dropna()
                
                if not precos.empty:
                    fig = px.histogram(
                        precos,
                        title='Distribuição de Preços da Farinha (R$/kg)',
                        labels={'value': 'Preço (R$/kg)'}
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
                    labels={'index': 'Dificuldade', 'value': 'Contagem'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar dificuldades de comercialização")
        else:
            st.warning("Dados de dificuldades na comercialização não disponíveis")

with tab4:
    st.subheader("Dificuldades no Cultivo")
    
    # Dificuldades no cultivo
    if 'Dificuldades_Cultivo' in filtered_df.columns:
        try:
            cultivo_dificuldades = filtered_df['Dificuldades_Cultivo'].str.split(',', expand=True).stack().value_counts()
            fig = px.bar(
                cultivo_dificuldades,
                title='Dificuldades no Cultivo',
                labels={'index': 'Dificuldade', 'value': 'Contagem'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Erro ao processar dificuldades no cultivo")
    else:
        st.warning("Dados de dificuldades no cultivo não disponíveis")
    

    

        # Assistência Técnica
    if 'Assistencia_Tecnica' in filtered_df.columns:
            assistencia = filtered_df['Assistencia_Tecnica'].value_counts()
            fig = px.pie(
                assistencia,
                names=assistencia.index,
                title='Acesso à Assistência Técnica'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
            st.warning("Dados de assistência técnica não disponíveis")
    

        # Dificuldades no processamento
    if 'Dificuldades_Processamento' in filtered_df.columns:
            try:
                processamento_dificuldades = filtered_df['Dificuldades_Processamento'].str.split(',', expand=True).stack().value_counts()
                fig = px.bar(
                    processamento_dificuldades,
                    title='Dificuldades no Processamento',
                    labels={'index': 'Dificuldade', 'value': 'Contagem'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Erro ao processar dificuldades no processamento")
    else:
            st.warning("Dados de dificuldades no processamento não disponíveis")

with tab5:
    st.subheader("Dados Completos")
    st.dataframe(filtered_df, height=600)
    
    # Botão para download
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar dados filtrados (CSV)",
        data=csv,
        file_name='dados_mandioca_filtrados.csv',
        mime='text/csv'
    )

# Rodapé
st.markdown("---")
st.caption("Dashboard de Produção de Mandioca e Macaxeira em Juruti - Dados coletados em 2025 | Maniva Tapajós | LABCRIA")