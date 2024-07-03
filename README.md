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

## EmissionModeling Class (LSTM)

The `EmissionModeling` class facilitates time series forecasting of emissions using LSTM models. It includes methods for data loading, preprocessing, model training, saving/loading models, and plotting forecasts.

### Methods
- `__init__(self, data_type)`: Initializes the `EmissionModeling` object with the specified `data_type` and initializes empty dictionaries (`models`, `scalers`) to store trained models and scalers for each country.
- `load_data(query, connection)`: Loads emissions data from a SQL database using the provided query and connection parameters. The loaded data is stored in a Pandas DataFrame (`self.df`).
- `preprocess_data(country_data)`: Preprocesses the emissions data specific to a country. It scales the emissions values using `MinMaxScaler`, splits the data into training and testing sets, and reshapes the data into the required format for LSTM input.
- `create_dataset(dataset, look_back=1)`: Creates sequences of input features (`dataX`) and corresponding output labels (`dataY`) from the given dataset using a specified look-back period (`look_back`).
- `train_lstm(trainX, trainY)`: Defines and trains an LSTM model using the provided training data (`trainX`, `trainY`). The model consists of an LSTM layer with 4 units followed by a Dense layer. It uses mean squared error as the loss function and Adam optimizer.
- `train_models(countries)`: Iteratively trains LSTM models for each country specified in the `countries` list. For each country, it preprocesses the data, trains an LSTM model, and stores the trained model and scaler.
- `save_models(path='models/')`: Saves the trained LSTM models (`models`) and corresponding scalers (`scalers`) to the specified path using HDF5 format for models and pickle format for scalers.
- `load_models(path='models/')`: Loads previously trained LSTM models (`models`) and scalers (`scalers`) from the specified path. Models are loaded from HDF5 files and scalers are loaded from pickle files.
- `plot_forecast(country, n_steps)`: Generates a forecast plot for emissions of the specified `country` for `n_steps` into the future using the trained LSTM model. It retrieves the data for the country, scales it, makes predictions using the LSTM model, and plots the original data alongside the forecasted values.
- `train_and_plot_if_needed(country, n_steps)`: Checks if a trained model exists for the specified `country`. If not, it trains a new LSTM model using historical data for the country and then generates and returns a forecast plot for `n_steps` into the future. If a trained model already exists, it directly generates and returns the forecast plot. If no trained model exists for a country specified in `train_and_plot_if_needed`, it will raise an exception indicating that the model or scaler is not found.

### EmissionDataModeling Class (ARIMA)

The `EmissionDataModeling` class includes the following methods:

- `fit_models()`: Trains a model for each country.
- `predict(country_name, n_steps)`: Forecasts emissions for the next `n_steps` years for a given country.
- `plot_forecast(country_name, n_steps)`: Plots the forecasted emissions for a given country using Matplotlib.

## Deployment

The Streamlit application and the MySQL database are both containerized using Docker. These containers are defined and run together using Docker Compose, which simplifies the deployment process and ensures that the application and database work together seamlessly.

### Docker Compose

Docker Compose is used to run the Streamlit application and MySQL database together. The `docker-compose.yml` file in the root directory of the project defines the services that make up the application. It includes the build context and Dockerfile location for each service, and sets up the necessary environment variables and ports. This Docker Compose was run on a host machine running Windows 11. Unfortunately, I can't predict it will work on Linux and mac OS.

To deploy the application, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/makspervov/co2_n2o_emissions.git
    ```
2. Navigate to the directory containing the `docker-compose.yml` file:
    ```sh
    cd co2_n2o_emissions
    ```
3. Build the services:
    ```sh
    docker compose build
    ```
4. Start the services:
    ```sh
    docker compose up --build
    ```
This will build the Docker images (if they don't already exist) and start the services defined in `docker-compose.yml`.

## Conclusion

This project provides valuable insights into the trends and patterns of CO₂ and N₂O emissions. The forecasting model can be used to predict future emissions, which can inform policy decisions and climate change mitigation strategies. The use of Docker and Docker Compose makes it easy to deploy and scale the application.
