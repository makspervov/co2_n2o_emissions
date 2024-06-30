# CO₂ & N₂O Emissions Exploratory Data Analysis and Forecasting

This project aims to analyze and forecast CO₂ and N₂O emissions data using machine learning techniques. The project is divided into two main parts: the Streamlit application and the MySQL database.

## Streamlit Application

The Streamlit application is located in the `app` directory. It includes a machine learning model for forecasting and provides several visualizations of the emissions data.

### Dockerfile

The `Dockerfile` in the `app` directory is used to build a Docker image of the Streamlit application. It sets up the necessary Python environment, installs the required packages from the `requirements.txt` file, and runs the Streamlit application on port 8501.

## MySQL Database

The MySQL database is set up in the `db` directory. It contains the emissions data that the Streamlit application uses for analysis and forecasting.

### Dockerfile

The `Dockerfile` in the `db` directory is used to build a Docker image of the MySQL database. It sets up a MySQL server and initializes the database using the `init.sql` script.

## Dataset Information

The datasets include the following columns:

- `year_data`: The year of the data point.
- `entity`: The country for which the data point is reported.
- `value_mt`: The amount of emissions (in metric tons) for that year and country.

## Graphs

Several graphs are generated to visualize the data:

- **Annual Emissions by Year**: This graph shows the total annual emissions for all countries.
- **Top 10 Countries with Highest Emissions**: This graph shows the 10 countries with the highest total emissions.
- **Emissions by Country**: This graph shows the annual emissions for a selected country.

## Machine Learning Modeling

A machine learning model is trained to forecast future emissions. The models are instances of the `EmissionModeling` (for LSTM) and `EmissionDataModeling` (for ARIMA) classes.

### EmissionModeling Class (LSTM)

The `EmissionModeling` class includes the following methods:

- `load_data(query, connection)`: Loads data from the database using a SQL query and connection.
- `preprocess_data(country_data)`: Preprocesses the data for the LSTM model.
- `create_dataset(dataset, look_back=1)`: Creates a dataset for the LSTM model.
- `train_lstm(trainX, trainY)`: Trains the LSTM model.
- `train_models(countries)`: Trains models for each country.
- `save_models(path='models/')`: Saves the trained models.
- `load_models(path='models/')`: Loads the trained models.
- `plot_forecast(country, n_steps)`: Plots the forecasted emissions for a given country using Plotly.

### EmissionDataModeling Class (ARIMA)

The `EmissionDataModeling` class includes the following methods:

- `fit_models()`: Trains a model for each country.
- `predict(country_name, n_steps)`: Forecasts emissions for the next `n_steps` years for a given country.
- `plot_forecast(country_name, n_steps)`: Plots the forecasted emissions for a given country using Matplotlib.

## Deployment

The Streamlit application and the MySQL database are both containerized using Docker. These containers are defined and run together using Docker Compose, which simplifies the deployment process and ensures that the application and database work together seamlessly.

### Docker Compose

Docker Compose is used to run the Streamlit application and MySQL database together. The `docker-compose.yml` file in the root directory of the project defines the services that make up the application. It includes the build context and Dockerfile location for each service, and sets up the necessary environment variables and ports.

To deploy the application, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/emissions-forecasting.git
    ```
2. Navigate to the directory containing the `docker-compose.yml` file:
    ```sh
    cd emissions-forecasting
    ```
3. Build and start the services:
    ```sh
    docker-compose up --build
    ```
This will build the Docker images (if they don't already exist) and start the services defined in `docker-compose.yml`.

## Conclusion

This project provides valuable insights into the trends and patterns of CO₂ and N₂O emissions. The forecasting model can be used to predict future emissions, which can inform policy decisions and climate change mitigation strategies. The use of Docker and Docker Compose makes it easy to deploy and scale the application.
