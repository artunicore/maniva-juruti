

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components



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

                            // Atualiza a posição dos elementos a cada tick da simulação
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

                            // Funções para arrastar os nós
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

# Legenda


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