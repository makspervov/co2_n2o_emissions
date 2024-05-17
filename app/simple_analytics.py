import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pymysql
from statsmodels.tsa.arima.model import ARIMA

# Set up database connection
connection = pymysql.connect(host='db',  # Host name
                             user='root',  # MySQL username
                             password='my-secret-pw',  # MySQL user password
                             db='emissions')  # Database name

# Auxiliary functions
def create_model(emission_series):
    if len(emission_series) == 0:
        return None

    # Convert index to datetime format
    emission_series.index = pd.to_datetime(emission_series.index, format='%Y')

    t = emission_series.index.map(lambda x: x.timestamp()).to_numpy()
    t -= t[0]
    y = emission_series.to_numpy()

    model = ARIMA(y, order=(3, 1, 1))
    model_fit = model.fit()

    return model_fit


def plot_time_series(emission_series, forecast, title):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=emission_series.index, y=emission_series, name="Observed", mode="lines+markers"
        )
    )
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast, name="Forecast", mode="markers")
    )
    fig.update_layout(title=title)
    return fig

class EmissionDataModeling:

    def __init__(self, emission_type):

        # Choose the data set based on the emission type
        if emission_type == "CO\u2082 emisiions per capita":
            query = 'SELECT * FROM co_emissions_per_capita'
        elif emission_type == "N\u2082O emisiions per capita":
            query = 'SELECT * FROM per_capita_nitrous_oxide'
        else:
            raise ValueError(f"Unknown emission type: {emission_type}")

        # Load the data set
        self.df = pd.read_sql(query, connection)

        # Get all countries from the model - the id is the pair of Entity and country_code
        countries = self.df["Entity"].unique()

        # Average every year for each country
        emissions_averaged = {}

        for i in range(len(countries)):
            emissions_averaged[countries[i]] = (
                self.df[self.df["Entity"] == countries[i]]
                .groupby("Year")["Value"]
                .mean()
            )

        self.emissions_averaged = emissions_averaged
        self.countries = countries

    def fit_models(self):
        self.models = {}

        for country in self.countries:
            model = create_model(self.emissions_averaged[country])
            if model is not None:
                self.models[country] = model

    def predict(self, Entity, n_steps):
        model = self.models[Entity]

        forecast = model.forecast(steps=n_steps)

        new_index = pd.date_range(
            start=self.emissions_averaged[Entity].index[-1], periods=n_steps, freq="Y"
        )

        forecast = pd.Series(forecast, index=new_index)

        return forecast
    
    def plot_forecast(self, Entity, n_steps):
        # Get a forecast
        forecast = self.predict(Entity, n_steps)

        # Get historical data
        emission_series = self.emissions_averaged[Entity]

        # Plot the historical data and the forecast
        fig = plot_time_series(emission_series, forecast, f'Emissions Forecast for {Entity}')

        return fig