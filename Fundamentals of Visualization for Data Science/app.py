import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import base64
import io

# Carregar os dados
path = 'data/Cleaned_Dataset.csv'
data = pd.read_csv(path)
data_corr = data.dropna(subset=['Salary', 'City', 'Country'])

# Configuração da página
st.set_page_config(page_title="Opportunities for Data Scientists in the UK", layout="wide")

# URL da imagem e link do Kaggle
source = "https://th.bing.com/th/id/OIP.zWLJ_uRbBRHelY8p_ZkWJQHaBH?rs=1&pid=ImgDetMain"
kaggle_link = "https://www.kaggle.com/datasets/emreksz/data-scientist-job-roles-in-uk"

# Adicionar ícone do GitHub no canto superior direito
st.markdown(
    """
    <style>
    .github {
        position: absolute;
        top: 50px;
        right: 10px;
        z-index: 1000;  
    </style>
    <a href="https://github.com/willianpina" target="_blank" class="github">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="50" height="50">
    </a>
    """,
    unsafe_allow_html=True
)

# Cabeçalho e informações do projeto
st.markdown(
    """
    <div style="background-color: white; padding: 10px 0; text-align: center;">
        <img src="{}" style="width: 600px;">
    </div>
    """.format(source),
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>Opportunities for Data Scientists in the UK: Trends and Salaries</h1>",
            unsafe_allow_html=True)
st.markdown(
    """
    This project aims to explore the job market for data scientists in the UK, with a focus on salary trends, required skills, and job locations.
    By analyzing this data, we can gain valuable insights into the current demand for data scientists and the qualifications needed to succeed in this field.

    The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/emreksz/data-scientist-job-roles-in-uk),
    containing detailed information about job roles for data scientists in the UK, such as company names, job titles, salary ranges, and required skills.

    This Streamlit application was developed as part of the final project for the Fundamentals of Data Visualization course in the Data Science Master's program at the University of Colorado Boulder.
    """
)

# Seção de análise de distribuição salarial
st.header('1. Salary Distribution Analysis')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Salary Distribution Across Cities in the UK')
    fig_city = px.box(data_corr, x='City', y='Salary')
    st.plotly_chart(fig_city, use_container_width=True)

with col2:
    st.subheader('Salary Distribution Across Countries')
    fig_country = px.box(data_corr, x='Country', y='Salary')
    st.plotly_chart(fig_country, use_container_width=True)

# Filtros de País e Cidade
st.header('2. Most In-Demand Skills for Data Scientists')

# Adicionar opção "All" para seleção de País e Cidade
selected_country = st.selectbox('Filter by Country:', options=['All'] + list(data_corr['Country'].unique()), index=0)
if selected_country != 'All':
    cities = ['All'] + list(data_corr[data_corr['Country'] == selected_country]['City'].unique())
else:
    cities = ['All']

selected_city = st.selectbox('Filter by City:', options=cities, index=0)

# Gerar gráfico de dispersão baseado nos filtros
filtered_data = data.copy()

if selected_country != 'All':
    filtered_data = filtered_data[filtered_data['Country'] == selected_country]

if selected_city != 'All':
    filtered_data = filtered_data[filtered_data['City'] == selected_city]

st.header('3. Correlation between Company Score and Salary')
scatter_fig = px.scatter(filtered_data, x='Company Score', y='Salary', color='Remote', symbol='Remote')
st.plotly_chart(scatter_fig, use_container_width=True)

# Nuvem de Palavras (WordCloud)
st.header('4. Most In-Demand Skills')
skills_list = data_corr['Skills'].dropna().str.split(', ')
all_skills = [skill for sublist in skills_list for skill in sublist]
skill_counts = Counter(all_skills)

if skill_counts:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_image = base64.b64encode(img.getvalue()).decode()
    st.markdown(
        f"<div style='display: flex; justify-content: center;'><img src='data:image/png;base64,{wordcloud_image}'></div>",
        unsafe_allow_html=True
    )
else:
    st.text("No skills data available.")


