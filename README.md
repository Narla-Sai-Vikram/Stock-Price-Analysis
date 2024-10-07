Stock Predictions Model

This repository contains a stock predictions model using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data from Google (GOOG) and uses a combination of technical indicators and machine learning algorithms to predict future stock prices.

Model Architecture

The model consists of four LSTM layers with 50, 60, 80, and 120 units, respectively. Each LSTM layer is followed by a dropout layer with a dropout rate of 0.2, 0.3, 0.4, and 0.5, respectively. The output of the final LSTM layer is fed into a dense layer with a single unit, which outputs the predicted stock price.

Training

The model is trained on a dataset of historical stock prices from Google (GOOG) from 2014-01-01 to 2024-07-31. The dataset is split into training and testing sets, with 80% of the data used for training and 20% used for testing. The model is trained using the Adam optimizer and mean squared error as the loss function.

Evaluation

The model is evaluated using the mean squared error between the predicted and actual stock prices. The model achieves a mean squared error of 0.0021 on the testing set.

Usage

To use the model, simply load the pre-trained model using Keras and provide it with a sequence of 100 previous stock prices as input. The model will output the predicted stock price for the next time step.

Dependencies

Python 3.x
Keras 2.x
TensorFlow 2.x
NumPy 1.x
Pandas 1.x
Matplotlib 3.x
yfinance 0.x

Acknowledgments
This repository is inspired by various online resources on stock price prediction using LSTM neural networks
