import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Layer:
    def __init__(self, input_num, output_num, bias):
        self.outputs = None
        self.nets = None
        self.errors = None
        self.bias = bias
        self.weights = np.random.randn(input_num + self.bias, output_num)

class mlp:
    def __init__(self,X_train ,y , hidden_layers, hidden_neurons, learning_rate, epochs, bias, activation_function):
        self.X_train = X_train.copy()
        self.y = y
        output_size = len(self.y.columns)
        self.output_size = output_size
        input_size = len(X_train.columns)
        self.bias = bias
        self.epochs = epochs
        self.layers = []
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.layer_num = hidden_layers + 1
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function

        for i in range(hidden_layers + 1):
            if i == 0:
                lyr = Layer(input_size, hidden_neurons[i], bias)
                self.layers.append(lyr)
            elif i == hidden_layers:
                lyr = Layer(hidden_neurons[i-1], output_size, bias)
                self.layers.append(lyr)
            else:
                lyr = Layer(hidden_neurons[i-1], hidden_neurons[i], bias)
                self.layers.append(lyr)

    def activationFn(self, actFn, net, derv=0):
        if (actFn+1) == 1:
            if derv == 1:
                return net*(1-net)
            else:
                y = 1 / (1 + np.exp(-net))
        if (actFn+1) == 2:
            if derv == 1:
                return 1 - (net ** 2)
            else:
                y = np.tanh(net)
        return y

    def forward_pass(self, x):
        for i in range(len(self.layers)):
            if i == 0:
                input = np.array(x).reshape(1, -1)
            else:
                input = self.layers[i-1].outputs
            if self.bias:
                input = np.hstack((input, np.ones((input.shape[0], 1))))
            net = np.dot(input, self.layers[i].weights)
            self.layers[i].nets = net
            y = self.activationFn(0, net)
            self.layers[i].outputs = y

    def backpropagation(self, Y_t):
        Y_t = np.array(Y_t).reshape(1, -1)
        LY = len(self.layers)-1
        Ou_lY = self.layers[LY]
        error = Y_t - Ou_lY.outputs
        delta = error * self.activationFn(self.activation_function, Ou_lY.outputs, derv=1)
        self.layers[LY].errors = delta
        for i in reversed(range(LY)):
            weight = self.layers[i+1].weights
            if self.bias:
                weight = weight[:-1,:]
            self.layers[i].errors = self.activationFn(self.activation_function, self.layers[i].outputs, derv=1) * (np.dot(self.layers[i+1].errors, weight.T))

    def update_weights(self, x):
        for i in range(len(self.layers)):
            if i == 0:
                input = np.array(x)
            else:
                input = self.layers[i-1].outputs
            if input.ndim == 1:
                input = input.reshape(1, -1)
            if self.bias:
                input = np.hstack((input, np.ones((input.shape[0], 1))))
            self.layers[i].weights = self.layers[i].weights + self.learning_rate * np.dot(input.T, self.layers[i].errors)

    def train(self):
        for epo in range(self.epochs):
            for i in range(len(self.X_train)):
                self.forward_pass(self.X_train.iloc[i,:])
                self.backpropagation(self.y.iloc[i,:])
                self.update_weights(self.X_train.iloc[i,:])

        # Evaluate on training set after all epochs
        y_true, y_pred = [], []
        for i in range(len(self.X_train)):
            self.forward_pass(self.X_train.iloc[i,:])
            actual    = np.array(self.y.iloc[i,:]).flatten()
            predicted = self.layers[-1].outputs.flatten()
            y_true.append(int(np.argmax(actual)))
            y_pred.append(int(np.argmax(predicted)))

        train_accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) * 100
        train_cm       = confusion_matrix(y_true, y_pred).tolist()
        return train_accuracy, train_cm

    def test(self, X_test, y_test):
        y_true, y_pred = [], []
        total_loss = 0.0
        for row in range(len(X_test)):
            self.forward_pass(X_test.iloc[row,:])
            actual    = np.array(y_test.iloc[row,:])
            predicted = self.layers[-1].outputs.flatten()
            y_true.append(int(np.argmax(actual)))
            y_pred.append(int(np.argmax(predicted)))
            total_loss += float(np.mean((actual - predicted) ** 2))
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) * 100
        avg_loss = total_loss / len(X_test)
        cm       = confusion_matrix(y_true, y_pred).tolist()
        return accuracy, avg_loss, cm
