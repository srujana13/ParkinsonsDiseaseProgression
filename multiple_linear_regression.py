import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import preprocessing

costs_train=[]
def initialize_parameters(lenw):
    w=np.random.randn(1, lenw)
    #random numbers through random normal distribution
    b=0
    return w,b

def forward_prop(X,w,b):
    z=np.dot(w,X)+b
    return z
    #w has dimension 1*n
    #x has dimension n*m where n is the number of features and m is number of training samples

def cost_function(z,y):
    m=y.shape[1]
    # print(float(1.0/2*m))
    #print(np.sum(np.square(z-y)))
    J=(float(1)/(float(2)*(m)))*np.sum(np.square(z-y))
    #print(J)
    return J

def back_prop(X,y,z):
    m=y.shape[1]
    dz=(float(1)/float(m))*(z-y)
    #print(dz)
    dw=np.dot(dz,X.T)
    #print(dw)
    db=np.sum(dz)
    #print(db)
    return dw, db

def gradient_descent_update(w,b,dw,db,learning_rate):
    w=w-learning_rate*dw
    #print(learning_rate*dw)
    b=b-learning_rate*db
    #print(learning_rate*db)
    return w, b

def linear_regression_model(X_train,y_train,X_val,y_val,learning_rate,epochs):
    lenw=X_train.shape[0]
    w,b=initialize_parameters(lenw)
    
    m_train=y_train.shape[1]
    m_val=y_val.shape[1]

    for i in range(1, epochs+1):
        z_train=forward_prop(X_train, w, b)
        #print(cost_function(z_train, y_train))
        cost_train=cost_function(z_train, y_train)
        dw, db=back_prop(X_train, y_train, z_train)
        w,b=gradient_descent_update(w,b,dw,db,learning_rate)
        if i%10==0:
            costs_train.append(cost_train)

        MAE_train=(float(1)/float(m_train))*np.sum(np.abs(z_train-y_train))

        z_val=forward_prop(X_val, w, b)
        cost_val=cost_function(z_val, y_val)  
        MAE_val=(float(1)/float(m_val))*np.sum(np.abs(z_val-y_val))

        print (" learning_rate "+str(learning_rate)+" epochs "+str(epochs))
        print("Training Cost")
        print(cost_train)
        print("Validation Cost")
        print(cost_val)


        print("Training MAE")
        print(MAE_train)
        print("Validation MAE")
        print(MAE_val)

   

X_train, X_val, y_train, y_val = preprocessing.load_data('./parkinsons_updrs.csv')
X_train, X_val = preprocessing.random_forest_features(X_train, y_train, X_val)
X_train=X_train.T
X_val=X_val.T
y_train=np.array([y_train])
y_val=np.array([y_val])
linear_regression_model(X_train, y_train, X_val, y_val,  0.0001 , 500)

linear_regression=linear_model.LinearRegression()
model=linear_regression.fit(X_train.T, y_train.T)
predictions=linear_regression.predict(X_val.T)
print ("with the library")
MAE_with_library=(1.0/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))
print(MAE_with_library)

plt.plot(costs_train)
plt.xlabel("Iterations")
plt.ylabel("Training Cost")
plt.show()
