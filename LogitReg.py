import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.linear_model import LogisticRegression

#Logistic Regression is commonly used to estimate the probability that an instance belongs to a particular class

#Lets first graph the logistic function
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], 'k-')
plt.plot([-10, 10], [0.5, 0.5], 'k:')
plt.plot([-10, 10], [1, 1], 'k:')
plt.plot([0, 0], [-1.1, 1.1], 'k-')
plt.plot(t, sig, 'b-', linewidth=2, label=r'$\sigma(t) = \frac{1}{1 + e^{-t}}$')
plt.xlabel('t')
plt.legend(loc='upper left', fontsize=18)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()

#Now lets use Logistic regression on the iris dataset
from sklearn import datasets
iris = datasets.load_iris()
iris.keys()
iris.data
iris.target

#Lets build a classifier to detect the Iris-Virginica (2) type based only on the petal width feature
X = iris['data'][:,3:] #petal width
y = (iris['target']==2).astype(np.int)

#Train our model
log_reg = LogisticRegression()
log_reg.fit(X,y)
#Now lets look at the estimated probabilities for flowers with petal length (0,3)
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:,0], 'b--', label='Not Iris-Virginica')
plt.legend(loc='lower right')
plt.xlabel('Petal Width(cm)')
plt.ylabel('Probability')
plt.show()

#The point where the lines overlap is called the decision boundary, below that point(<~1.5) the model predicts not iris-virginica, and vice versa
log_reg.predict([[1.8],[1.3]]) #Using predict instead of predict_proba returns the class the model predicts is most likely, not the probability

#Now lets plot petal width vs petal length. The dotted line represents the decision boundary of our logistic boundary
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], 'bs')
plt.plot(X[y==1, 0], X[y==1, 1], 'g^')

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, 'k--', linewidth=3)
plt.text(3.5, 1.5, 'Not Iris-Virginica', fontsize=14, color='b', ha='center')
plt.text(6.5, 2.3, 'Iris-Virginica', fontsize=14, color='g', ha='center')
plt.xlabel('Petal length', fontsize=14)
plt.ylabel('Petal width', fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()
