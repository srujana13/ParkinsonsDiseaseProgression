import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import preprocessing

learning_rate=0.04
epochs=200
MAE_val=[]

def forward_propogation(X_input,weights,b):
    return (b+np.dot(weights,X_input))

def cost(z_predicted,y_actual):
    return ((float(1)/(float(2)*(y_actual.shape[1])))*np.sum(np.square(z_predicted-y_actual)))

def back_propogation(X,y,z):
    dz=(float(1)/float(y.shape[1]))*(z-y)
    return (np.dot(dz,X.T)), np.sum(dz)

def gradient_descent(w,b,dw,db,learning_rate):
    return (w-learning_rate*dw), (b-learning_rate*db)

def linear_regression(X_train,y_train,X_val,y_val,learning_rate,epochs):
    length_w=X_train.shape[0]
    w,b=np.random.randn(1, length_w),0
    rows_training,rows_validation=y_train.shape[1],y_val.shape[1]

    for i in range(epochs):
        z_train=forward_propogation(X_train, w, b)
        training_cost=cost(z_train, y_train)
        dw, db=back_propogation(X_train, y_train, z_train)
        w,b=gradient_descent(w,b,dw,db,learning_rate)
        training_MAE=(float(1)/float(rows_training))*np.sum(np.abs(z_train-y_train))

        z_val=forward_propogation(X_val, w, b)
        validation_cost=cost(z_val, y_val)  
        validation_MAE=(float(1)/float(rows_validation))*np.sum(np.abs(z_val-y_val))
        MAE_val.append(validation_MAE)

    return training_MAE, validation_MAE, np.sqrt(((z_val - y_val) ** 2).mean())

X_train, X_val, y_train, y_val = preprocessing.load_data('./parkinsons_updrs.csv')
X_train, X_val = preprocessing.random_forest_features(X_train, y_train, X_val)
X_train, X_val=X_train.T, X_val.T
y_train, y_val=np.array([y_train]), np.array([y_val])

training_MAE, validation_MAE, RMSE=linear_regression(X_train, y_train, X_val, y_val, learning_rate, epochs)

print ("learning_rate "+str(learning_rate)+" epochs "+str(epochs))
print("Training MAE")
print(training_MAE)
print("Validation MAE")
print(validation_MAE)
print ("RMSE")
print (RMSE)


linear_regression=linear_model.LinearRegression()
model=linear_regression.fit(X_train.T, y_train.T)
predictions=linear_regression.predict(X_val.T)

print ("MAE with the Sklearn library")
MAE_with_library=(1.0/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))
print(MAE_with_library)
print ("RMSE with the SKlearn library")
RMSE_with_library=(1.0/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))
print(np.sqrt(((predictions - y_val) ** 2).mean()))


plt.plot(MAE_val)
plt.xlabel("Iterations")
plt.ylabel("MAE values")
plt.show()
