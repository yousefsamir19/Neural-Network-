import numpy as np
import pandas as pd
class neuron:
    def __init__(self,y , error, net):
        self.y = y
        self.error = error
        self.net = net

class mlp:
    def __init__(self, layers, hidden_n, eta, epochs, bias, actv_fn):
        self.layers = layers
        self.hidden_n = hidden_n
        self.eta = eta
        self.epochs = epochs
        self.bias = bias
        self.actv_fn = actv_fn

    def initialization(self, X_train, y):
            outputs = len(y.value_counts())
            inputs = len(X_train.columns)
            print(outputs, inputs) 
            biaslist = []
            neuronBig = []
            for i in range(self.layers):
                neuronList = []
                if i == self.layers:
                     for j in outputs:
                          neuronList.append(neuron(0,0,0))
                else:
                     for j in range(self.hidden_n):
                          neuronList.append(neuron(0,0,0))
                neuronBig.append(neuronList)

            for i in range(self.layers):
                weightsbig = []
                if i == 0:
                    weightslist = []
                    weightslist.append(np.random.rand(self.hidden_n,inputs))
                elif i == self.layers:
                     weightslist.append(np.random.rand(outputs,self.hidden_n))
                else:
                     weightslist.append(np.random.rand(self.hidden_n,self.hidden_n))
                weightsbig.append(weightslist)


df = pd.read_csv("penguins.csv")
x_train = df.iloc[:,1:]
print(x_train)
y = df["Species"]
obj = mlp(2,3,1,1,1,1)
obj.initialization(x_train,y)
            