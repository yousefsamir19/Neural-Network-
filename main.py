import preprocessing as pre
import mlp 

df = pre.preprocessing()
X_train, y_train, X_test, y_test,sc = pre.split(df)

model = mlp.mlp()

model = mlp(X_train,y_train,2,2,0.1,1,0,0)
sum =0
for i in range(len(X_train)):
    sum+=1
    model.forward_pass(X_train.iloc[i,:])
print("the sum = ",sum) 