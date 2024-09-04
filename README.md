# Stock Prediction Using LSTM

This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. The model is trained using past stock prices and is capable of predicting future trends.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
Predicting stock prices is a challenging task due to the inherent complexity and volatility of the financial markets. LSTM networks are especially suited for time series prediction tasks such as stock price forecasting, thanks to their ability to remember long-term dependencies. This project trains an LSTM model to predict stock prices.

## Features
- LSTM architecture for time-series stock price prediction.
- Pre-trained model (`lstm_stock_model.h5`) for testing.
- Jupyter Notebook (`Stock_Prediction_LSTM.ipynb`) for step-by-step guidance.

## Technologies Used
- Python
- Jupyter Notebook
- Keras with TensorFlow backend
- NumPy
- Pandas
- Matplotlib for visualization
- Scikit-learn for data preprocessing

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/sainruthik/Stock_Prediction_LSTM.git

## Dataset
The dataset used for training the LSTM model consists of historical stock prices. You can replace the dataset with any stock price dataset that follows a similar structure.

Ensure your dataset contains the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Usage
1. Open the Stock_Prediction_LSTM.ipynb file in Jupyter Notebook.
2. Follow the instructions in the notebook to:
    - Preprocess the data
    - Train the model
    - Test the model predictions
3. Alternatively, you can use the pre-trained model (lstm_stock_model.h5) for predictions.

## Model Overview
The LSTM model consists of:
- Sequential LSTM layers that capture long-term dependencies in stock price data.
- Dropout layers to prevent overfitting.
- The output layer with a linear activation function to predict stock prices for the next time step.

## Result
The model predicts future stock prices based on historical data. You can visualize the results and accuracy using the graphs provided in the Jupyter Notebook.
