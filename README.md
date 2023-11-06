# EXPERIMENT-03

# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.
   
2.Type the required program.

3.Print the program.

4.End the program.
 

## Program:
```py
Program to implement the linear regression using gradient descent.
Developed by: MIDHUN AZHAHU RAJA P
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
 
```

## Output:
### profit prediction graph:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/977e8b21-77a6-48e6-8d83-522ea1492b20)

### compute cost value:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/ad6e5fac-22ba-45da-a249-cf1f7d9cbaf0)

### h(x) value:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/d6e61072-d515-4468-94c9-484bed892e8e)

### cost function using gradient descent graph:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/d1e69a37-7817-4eda-949e-095f69f9c4d3)

profit prediction graph:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/98519423-8459-4728-b7be-3728682807aa)

### profit for the population 35,000:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/c6a96918-f188-4d2f-ae2e-71a8064bc96d)

### profit for the population 70,000:   

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393818/ba7f6131-b2f8-412d-bbd0-936a9ddd49ff)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
