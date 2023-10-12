# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:'

- Hardware – PCs
- Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

- Upload the file to your compiler.

- Type the required program.

- Print the program.

- End the program.


## Program:
```py
Program to implement the linear regression using gradient descent.
Developed by: MIDHUN AR
RegisterNumber:212222240066  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))

*/
```

## Output:
### 1. Profit Prediction graph


![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/ade7d394-96f8-4eff-a7a8-d67d71b58b27)
### 2.Compute Cost Value


![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/5436abbd-7265-4e59-b403-d98ef139d902)
### 3.h(x) Value

![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/ee64d2eb-0b85-498e-87ef-53a0f4e5a0c0)
### 4.Cost function using Gradient Descent Graph

![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/0e522f54-44e9-473c-98f0-4869c4164cf4)
### 5.Profit Prediction Graph

![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/a0094920-3d6d-41f9-ad1d-1d37c661c59a)
###6.Profit for the Population 35,000

![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/30711e74-3b36-4f2d-b07d-63f0c1d2b8bf)
### 7.Profit for the Population 70,000
![image](https://github.com/MidhunArPrabhu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118054670/6495fb92-5c76-4fe3-bf31-2854e9f3dc48)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
