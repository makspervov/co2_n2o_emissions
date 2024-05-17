import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pymysql
from simple_analytics import *

# Set up database connection
connection = pymysql.connect(host='db',  # Host name
                             user='root',  # MySQL username
                             password='my-secret-pw',  # MySQL user password
                             db='emissions')  # Database name

st.set_page_config(
    page_title="CO\u2082 & N\u2082O Emissions EDA",
    page_icon="🏭",
    initial_sidebar_state="auto",
)

st.title("CO\u2082 & N\u2082O Emissions EDA ")

data_type = st.sidebar.radio('Data',
                             ('CO\u2082 emisiions per capita','N\u2082O emisiions per capita'))

if data_type == 'CO\u2082 emisiions per capita':
    query = 'SELECT * FROM co_emissions_per_capita'
else: 
    query = 'SELECT * FROM per_capita_nitrous_oxide'

df = pd.read_sql(query, connection)

option = st.sidebar.selectbox(
    'Select the tab',
    ('Dataset Overview', 'Dataset Information', 'Graphs', 'ML Modeling'))

if option == 'Dataset Overview':
    st.title("Dataset Overview")
    
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

elif option == 'Dataset Information':
    st.title("Dataset Information")
    
    st.subheader("Data types of the columns:")
    st.dataframe(df.dtypes)

    st.subheader("Basic statistical information:")
    st.dataframe(df.describe())

elif option == 'Graphs':
    st.title("Graphs")
    
    graph_option = st.sidebar.radio(
        'Data type',
        ('Emission Worldwide', 'Annual Emission by Country'))
    
    if graph_option == 'Emission Worldwide':

        # Grouping the data by year and summing the values
        total_emissions_year = df.groupby('Year')['Value'].sum()

        fig = px.line(x=total_emissions_year.index, y=total_emissions_year.values)
        fig.update_layout(
            title='Annual Emissions by Year',
            xaxis_title='Year',
            yaxis_title='Total Emissions (mt)'
        )
        st.plotly_chart(fig)

        # Grouping data by country and calculating total emissions
        top_countries_emissions = df.groupby('Entity')['Value'].sum().nlargest(10)

        fig = px.bar(x=top_countries_emissions.values, y=top_countries_emissions.index, orientation='h')
        fig.update_layout(
            title='Top 10 Countries with Highest Emissions',
            xaxis_title='Annual Emissions (mt)',
            yaxis_title='Country',
        )
        st.plotly_chart(fig)

    elif graph_option == 'Annual Emission by Country':
        # Filtering data by selected country
        countries = df['Entity'].unique()

        selected_country = st.selectbox('Select a country', countries)
        country_data = df[df['Entity'] == selected_country]
        fig = px.line(country_data, x='Year', y='Value')
        fig.update_layout(
            title='Emissions by Country',
            xaxis_title='Year',
            yaxis_title='Total Emissions (mt)'
        )
        st.plotly_chart(fig)

elif option == 'ML Modeling':
    st.title("ML Modeling")

    # Creating an instance of EmissionDataModeling class
    modeling = EmissionDataModeling(data_type)

    # Training of models for all countries
    modeling.fit_models()

    # Country selection and number of forecast steps
    selected_country = st.selectbox('Select a country', modeling.countries)
    n_steps = st.slider('Number of steps to forecast', min_value=1, max_value=100, value=1)

    # Plotting the forecast graph for the selected country
    fig = modeling.plot_forecast(selected_country, n_steps)
    st.plotly_chart(fig)