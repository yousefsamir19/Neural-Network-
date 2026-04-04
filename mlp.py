import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder

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
        self.y = y # label column
        
        output_size = len(self.y.columns)  # get the output layer neuron counts
        self.output_size = output_size # Assign output_size to self
        input_size =len(X_train.columns) # get the number of the input features
        
        self.bias = bias 
        self.epochs = epochs
        self.layers = [] # list of layers of the class store each layer nets and y and weights and errors
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers # number hidden layers
        self.layer_num = hidden_layers + 1 # number of layers (hidden + output)
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function # 1 for sigmoid 2 for tanh

        for i in range(hidden_layers + 1):
            if i == 0: # make the intialization of the input layer
                lyr =Layer(input_size, hidden_neurons[i], bias)
                self.layers.append(lyr)
                
            elif i == hidden_layers: # make the intialization of the output layer
                lyr =Layer(hidden_neurons[i-1], output_size, bias)
                self.layers.append(lyr)

            else: # make the intialization of hidden layers
                lyr =Layer(hidden_neurons[i-1], hidden_neurons[i], bias)
                self.layers.append(lyr)


    def activationFn(self, actFn, net,derv=0):
          if (actFn+1) == 1: # Sigmoid Function
              if(derv)==1:
                  return net*(1-net) # the derivative of Sigmoid for error calculation
              else:
                   y = 1 / (1 + np.exp(-net))
          if (actFn+1) == 2: # Tanh Func
             if(derv)==1: 
                  return 1 - (net ** 2) # the derivative of Tanh for error calculation 
             else:
                y = np.tanh(net)
          return y


    def forward_pass(self ,x):
        for i in range(len(self.layers)): # itterate on each layer
            if i == 0: # take the input layer and convert it numpy array to be transposable
                input = np.array(x).reshape(1, -1)
            else: # take the previous output as input for the current layer
                input = self.layers[i-1].outputs

            if self.bias:# if there is a bias add column of ones to the input to match the weights matrix shape
                input = np.hstack((input, np.ones((input.shape[0], 1))))

            net = np.dot(input,self.layers[i].weights)
            self.layers[i].nets = net # store the net matix of the current layer

            y = self.activationFn(0,net)
            self.layers[i].outputs = y # store the output matix of the current layer


    def backpropagation(self,Y_t):
        Y_t = np.array(Y_t).reshape(1, -1)
        LY=len(self.layers)-1 
        Ou_lY=self.layers[LY] # get the output layer
        
        error=Y_t-Ou_lY.outputs # calculate the error 
        delta=error*self.activationFn(self.activation_function,Ou_lY.outputs,derv=1) # calculate the error signal for the output layer
        self.layers[LY].errors=delta
        
        for i in reversed(range(LY)): # calculate the error signals for the hidden layers
            weight=self.layers[i+1].weights
            if self.bias:
                weight=weight[:-1,:]
            self.layers[i].errors=self.activationFn(self.activation_function,self.layers[i].outputs,derv=1)*(np.dot(self.layers[i+1].errors,weight.T))


    def update_weights(self,x):
                for i in range(len(self.layers)):
                    if i == 0: 
                            input = np.array(x)
                    else: 
                            input = self.layers[i-1].outputs
                    if self.bias:
                            if input.ndim == 1:
                                input = input.reshape(1, -1)
                            input = np.hstack((input, np.ones((input.shape[0], 1)))) 
                    self.layers[i].weights=self.layers[i].weights + self.learning_rate*np.dot(input.T,self.layers[i].errors)


    def train(self):
        for epo in range(self.epochs):
            for i in range(len(self.X_train)):
                self.forward_pass(self.X_train.iloc[i,:])
                self.backpropagation(self.y.iloc[i,:])
                self.update_weights(self.X_train.iloc[i,:])
    
    def test(self,X_test,y_test):
        for x in X_test:
            self.forward_pass(x)
        
        acutal = y_test
        output_layer = len(self.layers) - 1
        predicted = self.layer_num[output_layer].outputs
        print(predicted)
        # for i in range(len(self.X_test)):
        #     if acutal == predicted:
        #         count+=1
        # accuracy = (count / len(X_test)) * 100