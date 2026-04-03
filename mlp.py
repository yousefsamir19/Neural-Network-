import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col.lower() == "species":
            rename_map[col] = "Species"
        elif col.lower() == "originlocation":
            rename_map[col] = "OriginLocation"
    return df.rename(columns=rename_map)


def get_preprocessed_df() -> pd.DataFrame:
    df = pd.read_csv("penguins.csv")
    df = normalize_columns(df)
    null_columns = df.columns[df.isnull().any()]
    numeric_cols = df.select_dtypes(include="number").columns
    for col in null_columns:
        if col in numeric_cols:
            df[col] = df.groupby("Species")[col].transform(lambda x: x.fillna(x.mean()))
    le = LabelEncoder()
    df["OriginLocation"] = le.fit_transform(df["OriginLocation"])
    return df




def split(data: pd.DataFrame):

    encoder = LabelEncoder()
    data["Species"] = encoder.fit_transform(data["Species"])

    train_data = data.iloc[np.r_[0:30, 50:80, 100:130]].copy()
    test_data  = data.iloc[np.r_[30:50, 80:100, 130:150]].copy()

    X_train = train_data.drop(columns=["Species"])
    y_train = train_data[["Species"]]

    X_test = test_data.drop(columns=["Species"])
    y_test = test_data[["Species"]]

    sc = MinMaxScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
    enc = OneHotEncoder(sparse_output=False)
    y_train_encoded = enc.fit_transform(y_train)
    y_test_encoded = enc.transform(y_test)

    return X_train, y_train_encoded, X_test, y_test_encoded, sc


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
        self.output_size = output_size # Assign output_size to self
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

        # if bias:
        #     self.X_train["x0"] = 1 # make a column of ones when bias = 1 to include bias in calculations in the input layer

        print(self.X_train.head())
        for i in range(hidden_layers + 1):
            if i == 0: # make the intialization of the input layer
                lyr =Layer(input_size, hidden_neurons[i], bias)
                self.layers.append(lyr)

                # --------------prints for test--------------
                # print(f'Layer : {i}')
                # print(lyr.weights)

            elif i == hidden_layers: # make the intialization of the output layer
                lyr =Layer(hidden_neurons[i-1], output_size, bias)
                self.layers.append(lyr)

                # --------------prints for test--------------
                # print('Output Layer')
                # print(lyr.weights)

            else: # make the intialization of hidden layers
                lyr =Layer(hidden_neurons[i-1], hidden_neurons[i], bias)
                self.layers.append(lyr)

                # --------------prints for test--------------
                # print(f'Layer : {i}')
                # print(lyr.weights)
    def activationFn(self, actFn, net,derv=0):


          if (actFn+1) == 1: # Sigmoid Function
              if(derv)==1:
                  return net*(1-net)
              else:
                   y = 1 / (1 + np.exp(-net))
          if (actFn+1) == 2: # Tanh Func
             if(derv)==1:
                  return 1 - (net ** 2)

             else:
                y = np.tanh(net)
             #y = ((e**net)-(e**(-net))) / ((e**net)+(e**(-net)))

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


            # --------------prints for test--------------
            # print(type(self.layers[i].nets))
            # print("layer nets ",i," = ",self.layers[i].nets,"\n")
            # print("layer ys ",i," = ",self.layers[i].outputs)


    def backpropagation(self,Y_t):
        Y_t = np.array(Y_t).reshape(1, -1)
        LY=len(self.layers)-1
        Ou_lY=self.layers[LY]
        error=Y_t-Ou_lY.outputs
        delta=error*self.activationFn(0,Ou_lY.outputs,derv=1)
        self.layers[LY].errors=delta
        for i in reversed(range(LY)):
            weight=self.layers[i+1].weights
            if self.bias:
                weight=weight[:-1,:]
            self.layers[i].errors=self.activationFn(0,self.layers[i].outputs,derv=1)*(np.dot(self.layers[i+1].errors,weight.T))

    # def updateweights(self, x_sample):
    #      input_data = np.array(x_sample)

    #      for i in range(len(self.layers)):
    #         if i == 0:
    #             input = input_data
    #         else:
    #            if self.bias:
    #                input = np.hstack((input_data, np.ones((input_data.shape[0], 1))))
    #            else:
    #                input = input_data

    #         DW = self.learning_rate * np.dot(input.T, self.layers[i].errors)
    #         self.layers[i].weights += DW

    #         input_data = self.layers[i].outputs

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
        
