from my_simple_linear_regressor import MySimpleLinearRegressor
from my_linear_regression_gd import MyLinearRegressionGD


import numpy as np
def main():
    # Starting with PA4, we will implement common ML algorithms
    # following the style of popular libraries (like scikit-learn or TensorFlow,etc.) API
    # API = Application Programming Interface
    # Reference: https://scikit-learn.org/1.3/tutorial/statistical_inference/supervised_learning.html

    # X: 2D feature matrix (rows = instances, columns = attributes)
    # y: 1D target vector (values we want to predict)
    # Note: y is stored separately from X

    # Typical workflow:
    # 1. Split X and y into training and testing sets
    # 2. Train the model using the training set (X_train, y_train)
    # 3. Evaluate the model using the testing set (X_test, y_test)
    # Note: X_train aligns with y_train, X_test aligns with y_test

    # Each algorithm is implemented as a class with two main methods:
    # - fit(X_train, y_train): trains the model on the training data
    # - predict(X_test): returns predictions (y_pred) for the test data
    #   y_pred is aligned with y_test

    # Model evaluation:
    # - Regression: e.g.,Mean Squared Error (MSE) Mean Absolute Error (MAE) — average of absolute differences
    # - Classification: e.g., Accuracy — proportion of correct predictions

    # my_simple_linear_regressor.py is our simple linear regression code
    # let's see the API in action!!
    # we need X_train and y_train data
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    my_lin=MySimpleLinearRegressor()
    my_lin.fit(X_train,y_train)
      
    y_pred=my_lin.predict([[200],[300]])
    print(y_pred)


    # Generate synthetic training data with two features 
    X_train = [[x, 100 - x] for x in range(100)]
    y_train = [3 * row[0] + 5 * row[1] + np.random.normal(0, 10) for row in X_train]
    print("X_train:", X_train[:5])
    print("y_train:", y_train[:5])

    mylin=MyLinearRegressionGD()
    mylin.fit(X_train,y_train)
    print(mylin.slopes)
    print(mylin.intercept)
    y_pred= mylin.predict([[4,50]])
    print(y_pred)




  

  


if __name__=='__main__':
    main()