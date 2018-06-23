import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
np.random.seed(42)
#What if data is not just a straight line, but more like a winding curve? By adding powers (X^n) of features as new features we can perform polynomial regression
#Lets plot some data based off of a simple quadratic equation

m = 100
X = 6*np.random.rand(m,1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)
plt.plot(X,y,'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-3,3,0,10])
plt.show()

#Lets use sklearn to transform our data and adding the square of each feature in dataset as a new feature
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
lin_reg.intercept_
lin_reg.coef_

#Lets plot our prediction line
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r-', linewidth=2, label='Predictions')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc='upper left')
plt.axis([-3, 3, 0, 10])
plt.show() #Fits much better than a simple line

#Lets plot learning curves for degrees=1,2,300
for style, width, degree in (('g-', 1, 300), ('b--', 2, 2), ('r-+', 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ('poly_features', polybig_features),
            ('std_scaler', std_scaler),
            ('lin_reg', lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, 'b.', linewidth=3)
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.axis([-3, 3, 0, 10])
plt.show() #Notice how degree=300 severely overfits the data

#Lets plot Learning Curves as a way to visualize if our model is overfitting/underfitting our data
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), 'r-+', label='train')
    plt.plot(np.sqrt(val_errors), 'b-', label='val')
    plt.legend(loc='upper right')
    plt.xlabel('Training_set_size')
    plt.ylabel('rmse')
plot_learning_curves(lin_reg, X, y)
plt.axis([0,80,0,3])
plt.show()

#Now lets plot the learning curves of a 10-degree polynomial model
polynomial_regression = Pipeline([
    ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
    ('lin_reg', LinearRegression())
])
plot_learning_curves(polynomial_regression, X, y)
plt.axis([0,80,0,3])
plt.show() #The gap between val and train signify our model may be overfitting, RMSE is lower than lin_reg because our data is quadratic
