import preprocessing as pre
import mlp 
import Visualization as vz


df = pre.get_preprocessed_df()
X_train, y_train, X_test, y_test,sc = pre.split(df)

model = mlp.mlp(X_train,y_train,1,[2],0.1,1,0,0)
model.train()
vz.visualize(model)
