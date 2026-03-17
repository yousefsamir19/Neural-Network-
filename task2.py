import numpy as np

class Layer:
    def __init__(self, input_num, output_num, bias):
        self.weights = np.random.randn(input_num, output_num) 
        self.bias = bias
        if self.bias == 0:
            self.biases = np.zeros((1, output_num))
        else:
            self.biases = np.random.randn(1, output_num)

class NeuralNetwork:
    def __init__(self, hidden_layers, hidden_neurons, learning_rate, epochs, bias, activation_function, input_size = 5, output_size = 3):
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.bias = bias
        self.activation_function = activation_function

        for i in range(hidden_layers + 1):
            if i == 0:
                lyr =Layer(input_size, hidden_neurons, bias)
                self.layers.append(lyr)
                print(f'Layer : {i}')
                print(lyr.weights)
                print(lyr.biases)
            elif i == hidden_layers:
                lyr =Layer(hidden_neurons, output_size, bias)
                print('Output Layer')
                print(lyr.weights)
                print(lyr.biases)
                self.layers.append(lyr)
            else:
                print(f'Layer : {i}')
                lyr =Layer(hidden_neurons, hidden_neurons, bias)
                print(lyr.weights)
                print(lyr.biases)
                self.layers.append(lyr)
obj = NeuralNetwork(hidden_layers=2, hidden_neurons=4, learning_rate=0.0341, epochs=104345, bias=True, activation_function='blahblah')
