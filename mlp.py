import numpy as np
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
        output_size=len(y.value_counts()) # get the output layer neuron counts 
        input_size =len(X_train.columns) # get the number of the input features
        self.y = y # label column
        self.bias = bias
        self.epochs = epochs
        self.layers = [] # list of layers of the class store each layer nets and y and weights and errors
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers # number hidden layers 
        self.layer_num = hidden_layers + 1 # number of layers (hidden + output)
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function # 1 for sigmoid 2 for tanh
        
        if bias:
            self.X_train["x0"] = 1 # make a column of ones when bias = 1 to include bias in calculations in the input layer
        
        print(self.X_train.head())
        for i in range(hidden_layers + 1):
            if i == 0: # make the intialization of the input layer
                lyr =Layer(input_size, hidden_neurons, bias)
                self.layers.append(lyr)
                
                # --------------prints for test-------------- 
                # print(f'Layer : {i}')
                # print(lyr.weights)
                
            elif i == hidden_layers: # make the intialization of the output layer
                lyr =Layer(hidden_neurons, output_size, bias)
                self.layers.append(lyr)
                
                # --------------prints for test-------------- 
                # print('Output Layer')
                # print(lyr.weights)
                
            else: # make the intialization of hidden layers
                lyr =Layer(hidden_neurons, hidden_neurons, bias)
                self.layers.append(lyr)
                
                # --------------prints for test-------------- 
                # print(f'Layer : {i}')
                # print(lyr.weights)

    def activationFn(self, actFn, net):
          if (actFn+1) == 1: # Sigmoid Function
              y = 1 / (1 + np.exp(-net))
               
          if (actFn+1) == 2: # Tanh Func
              y = np.tanh(net)
             #y = ((e**net)-(e**(-net))) / ((e**net)+(e**(-net)))
          
          return y
      
    def forward_pass(self ,x):
        for i in range(len(self.layers)): # itterate on each layer 
            if i == 0: # take the input layer and convert it numpy array to be transposable 
                input = np.array(x)
            else: # take the previous output as input for the current layer
                input = self.layers[i-1].outputs
                if self.bias:# if there is a bias add column of ones to the input to match the weights matrix shape 
                    input = np.hstack((input, np.ones((input.shape[0], 1)))) 
            
            net = np.dot(input,self.layers[i].weights)
            self.layers[i].nets = net # store the net matix of the current layer
            
            y = self.activationFn(0,net)
            self.layers[i].outputs = y # store the output matix of the current layer
            
            
            # --------------prints for test-------------- 
            # print(type(self.layers[i].nets))
            # print("layer nets ",i," = ",self.layers[i].nets,"\n")
            # print("layer ys ",i," = ",self.layers[i].outputs)
    
    def backpropagation(self):
        pass  # raneem fn

        
        