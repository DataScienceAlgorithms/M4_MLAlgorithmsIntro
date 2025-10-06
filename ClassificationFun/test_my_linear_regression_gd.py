import numpy as np
from sklearn.linear_model import SGDRegressor
from my_linear_regression_gd import MyLinearRegressionGD

#test for linear regression gradient descent fit 
def test_mylinearregression_gd_fit():
    np.random.seed(0)
    X_train=[[x, 100-x] for x in range(100)]
    y_train=[2 * x[0] + 5 * x[1]+np.random.normal(0,10) for x in X_train]
    # Fit sklearn model as baseline
    sklearn_model = SGDRegressor(max_iter=1000, eta0=0.0001, random_state=0)
    sklearn_model.fit(X_train, y_train)
    
    my_model=MyLinearRegressionGD(iterations=1000,learning_rate=0.0001)
    my_model.fit(X_train,y_train)
   

    assert np.allclose(my_model.slopes, sklearn_model.coef_, atol=0.15)
    #TODO: Add intercept comparison
        
#test for linear regression gradient descent predict  
def test_mylinearregression_gd_predict():
    np.random.seed(0)
    X_train=[[x, 100-x] for x in range(100)]
    y_train=[2 * x[0] + 5 * x[1]+np.random.normal(0,10) for x in X_train]
    #TODO add test cases


    