import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pickle
import plotly.graph_objs as go

class EmissionModeling:
    def __init__(self, data_type):
        self.data_type = data_type
        self.models = {}
        self.scalers = {}
        self.look_back = 1
    
    def load_data(self, query, connection):
        self.df = pd.read_sql(query, connection)
    
    def preprocess_data(self, country_data):
        data = country_data['value_mt'].values.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.8)
        test_size = len(scaled_data) - train_size
        train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

        trainX, trainY = self.create_dataset(train, self.look_back)
        testX, testY = self.create_dataset(test, self.look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return trainX, trainY, testX, testY, scaler

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def train_lstm(self, trainX, trainY):
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        return model
    
    def train_models(self, countries):
        for country in countries:
            country_data = self.df[self.df['entity'] == country].copy()
            country_data.set_index('year_data', inplace=True)
            trainX, trainY, testX, testY, scaler = self.preprocess_data(country_data)

            model = self.train_lstm(trainX, trainY)
            self.models[country] = model
            self.scalers[country] = scaler

            testPredict = model.predict(testX)
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            testScore = sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            print(f'{country} Test Score: {testScore:.2f} RMSE')

    def save_models(self, path='models/'):
        for country, model in self.models.items():
            model.save(f'{path}{country}_model.h5')
            with open(f'{path}{country}_scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers[country], f)

    def load_models(self, path='models/'):
        for country in self.df['entity'].unique():
            try:
                self.models[country] = tf.keras.models.load_model(f'{path}{country}_model.h5')
                with open(f'{path}{country}_scaler.pkl', 'rb') as f:
                    self.scalers[country] = pickle.load(f)
            except:
                pass

    def plot_forecast(self, country, n_steps):
        model = self.models.get(country)
        scaler = self.scalers.get(country)
        if model and scaler:
            country_data = self.df[self.df['entity'] == country].copy()
            data = country_data['value_mt'].values.reshape(-1, 1)
            scaled_data = scaler.transform(data)
            
            last_data = scaled_data[-self.look_back:].reshape(1, 1, self.look_back)
            predictions = []
            for _ in range(n_steps):
                pred = model.predict(last_data)
                predictions.append(pred[0, 0])
                last_data = np.roll(last_data, -1)
                last_data[0, 0, -1] = pred
            
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=country_data.index, y=country_data['value_mt'], mode='lines', name='Original Data'))
            forecast_index = range(country_data.index[-1] + 1, country_data.index[-1] + 1 + n_steps)
            fig.add_trace(go.Scatter(x=list(forecast_index), y=predictions.flatten(), mode='lines', name='Forecast'))
            fig.update_layout(title=f'Forecast for {country}', xaxis_title='Year', yaxis_title='Emissions')
            return fig
