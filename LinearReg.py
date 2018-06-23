import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, SGDRegressor

#----------------------------------------LINEAR REGRESSION----------------------------------------------------------

#Lets generate some random linear variables to visualize linear regression
np.random.seed(42)
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

#Lets plot this
plt.plot(X, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0,2,0,15])
plt.show()


#Now lets find the theta that optimizes the MSE for our model, first add x0=1 to each instance (b=y-intercept)
X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #The Normal Equation
theta_best #Theres are very close to our 4 and 3 in our linear reg model!!!

#Now lets make predictions using our theta_best on a new instance
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new] #add x0=1 to each instance (same as we did before)
y_predict = X_new_b.dot(theta_best)
y_predict


#Plot this prediction
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])
plt.show()


#Let do the same thing but with SciKit-Learn
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_,lin_reg.coef_
lin_reg.predict(X_new) #Same as our np model!!


#--------------------------------------------BATCH GRADIENT DESCENT------------------------------------

eta = 0.1 #Learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) #random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) #Gradient vector of cost function (MSE)
    theta = theta - eta*gradients #Gradient descent step (determines size of 'next step')
theta #these are exactly what our orignal np model found AND our Normal Equation!!
X_new_b.dot(theta)


#What happens if we change the learning rate (eta)?
theta_path_bgd = []
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = 'b-' if iteration > 0 else 'r--'
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('x')
    plt.axis([0, 2, 0, 15])
np.random.seed(42)
theta = np.random.randn(2,1) #random initialization

plt.figure()
plt.subplot(131); plot_gradient_descent(theta, eta =0.02)
plt.ylabel('y', rotation=0)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta = 0.5)
plt.show() #Left:eta is too low, center:eta is good, right:eta is too high


#-------------------------------------STOCHASTIC GRADIENT DESCENT-------------------------------------------

theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1)  # random initialization
for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = 'b-' if i > 0 else 'r--'
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
plt.plot(X, y, 'b.')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.axis([0, 2, 0, 15])
plt.show()
theta  #With only 50 iterations in SGD vs. 1000 iterations in BGD, we get very similar coefficients (SGD better for large scale data)

#Lets use SGD for LinReg using SciKit-Learn. SciKit-Learn defaults to optimizing the squared error cost function
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X,y)
sgd_reg.intercept_
sgd_reg.coef_

#--------------------------------------MINI BATCH GRADIENT DESCENT------------------------------------------

#Similar to BGD and SGD, computes the gradients on 'batches' of instances instead of all at once(BGD) or only one(SGD)
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
theta #Very similar coefficients to our previous Gradient algos and Normal Equation

#Now lets plot all 3 Gradient algos to better visualize advantages/disadvantages
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure()
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], 'r-s', linewidth=1, label='Stochastic')
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='Mini-Batch')
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], 'b-o', linewidth=3, label='Batch')
plt.legend(loc="upper left")
plt.xlabel('theta_0')
plt.ylabel(r'theta_1', rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()

#-----------------------------------------------COMPARISON OF ALGORITHMS FOR LINEAR REGRESSION-------------------------------------------------
data = {'Algorithm': ['Normal Equation', 'Batch GD', 'Stochastic GD', 'Mini-Batch GD'], 'Large M': ['Fast','Slow','Fast','Fast'], 'Large N': ['Slow', 'Fast', 'Fast', 'Fast'], 'Scaling Required': ['No', 'Yes', 'Yes', 'Yes']}
comparison = pd.DataFrame(data=data)
comparison
