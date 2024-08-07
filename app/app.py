import streamlit as st
import plotly.express as px
import pandas as pd
import pymysql
from lstm_model import EmissionModeling
from arima_forecast import EmissionDataModeling

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
        total_emissions_year = df.groupby('year_data')['value_mt'].sum()

        fig = px.line(x=total_emissions_year.index, y=total_emissions_year.values)
        fig.update_layout(
            title='Annual Emissions by Year',
            xaxis_title='Year',
            yaxis_title='Total Emissions (mt)'
        )
        st.plotly_chart(fig)

        # Grouping data by country and calculating total emissions
        top_countries_emissions = df.groupby('entity')['value_mt'].sum().nlargest(10)

        fig = px.bar(x=top_countries_emissions.values, y=top_countries_emissions.index, orientation='h')
        fig.update_layout(
            title='Top 10 Countries with Highest Emissions',
            xaxis_title='Annual Emissions (mt)',
            yaxis_title='Country',
        )
        st.plotly_chart(fig)

    elif graph_option == 'Annual Emission by Country':
        # Filtering data by selected country
        countries = df['entity'].unique()

        selected_country = st.selectbox('Select a country', countries)
        country_data = df[df['entity'] == selected_country]
        fig = px.line(country_data, x='year_data', y='value_mt')
        fig.update_layout(
            title='Emissions by Country',
            xaxis_title='Year',
            yaxis_title='Total Emissions (mt)'
        )
        st.plotly_chart(fig)

elif option == 'ML Modeling':
    st.title("ML Modeling")

    # Select the model type
    model_type = st.sidebar.selectbox('Select model type', ('ARIMA', 'LSTM'))

    # Creating an instance of EmissionModeling class
    if model_type == 'LSTM':
        modeling = EmissionModeling(data_type="sql")
        modeling.load_data(query, connection)
        modeling.load_models()

        countries = df['entity'].unique()
        selected_country = st.selectbox('Select a country', countries)
        n_steps = st.slider('Number of steps to forecast', min_value=1, max_value=100, value=1)

        if st.button('Show Forecast'):
            fig = modeling.train_and_plot_if_needed(selected_country, n_steps)
            st.plotly_chart(fig)

    elif model_type == 'ARIMA':
        modeling = EmissionDataModeling(data_type)
        modeling.fit_models()

        countries = df['entity'].unique()
        selected_country = st.selectbox('Select a country', countries)
        n_steps = st.slider('Number of steps to forecast', min_value=1, max_value=100, value=1)

        if st.button('Show Forecast'):
            fig = modeling.plot_forecast(selected_country, n_steps)
            st.plotly_chart(fig)

