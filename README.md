# CO₂ & N₂O Emissions Exploratory Data Analysis and Forecasting

This project aims to analyze and forecast CO₂ and N₂O emissions data using machine learning techniques. The project is divided into several parts:

## Dataset Overview

The datasets used in this project are `co-emissions-per-capita.csv` and `per-capita-nitrous-oxide.csv`. These datasets contain annual emissions data for different countries.

## Dataset Information

The datasets include the following columns:

- `Year`: The year of the data point.
- `Entity`: The country for which the data point is reported.
- `Value`: The amount of emissions (in metric tons) for that year and country.

## Graphs

Several graphs are generated to visualize the data:

- **Annual Emissions by Year**: This graph shows the total annual emissions for all countries.
- **Top 10 Countries with Highest Emissions**: This graph shows the 10 countries with the highest total emissions.
- **Emissions by Country**: This graph shows the annual emissions for a selected country.

## Machine Learning Modeling

A machine learning model is trained to forecast future emissions. The model is an instance of the `EmissionDataModeling` class, which uses the ARIMA model for time series forecasting.

The `EmissionDataModeling` class includes the following methods:

- `fit_models()`: Trains a model for each country.
- `predict(country_name, n_steps)`: Forecasts emissions for the next `n_steps` years for a given country.
- `plot_forecast(country_name, n_steps)`: Plots the forecasted emissions for a given country.

The project is deployed as a Streamlit web application and containerized using Docker for easy deployment and scaling.

## Deployment

The Streamlit application is containerized using Docker, and the Docker image is deployed on a Kubernetes cluster. This allows for easy scaling of the application to handle multiple users.

# Troubleshooting Kubernetes Cluster Issues

While working with a Kubernetes cluster, I have encountered various problems. One of them was the ImagePullBackOff and ErrImagePull error. I didn't fully understand the essence of this error, because earlier this project was running in K8s, but when I checked the project again, it started to generate these errors. 

## ImagePullBackOff or ErrImagePull Errors

These errors occur when Kubernetes cannot pull the Docker image. This might be because the image does not exist, or due to network issues, or authentication problems with the Docker registry.

## Conclusion

This project provides valuable insights into the trends and patterns of CO₂ and N₂O emissions. The forecasting model can be used to predict future emissions, which can inform policy decisions and climate change mitigation strategies.
