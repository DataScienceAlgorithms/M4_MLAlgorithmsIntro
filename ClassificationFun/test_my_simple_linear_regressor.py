import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from my_simple_linear_regressor import MySimpleLinearRegressor


# test modules and test functions 
# with pytest start with test_

def test_mysimplelinearregressor_fit():
    # use assert statements to form test cases
    # start with simple/common test cases
    # then move on to complex/edge test cases
    
    # test cases for fit()
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    my_lin = MySimpleLinearRegressor()
    my_lin.fit(X_train, y_train) # "fits" slope (m) and intercept (b)
    
    # 2 main ways to write asserts
    # 1. assert against "desk calculation" or "desk check"
    # order: actual, expected (solution)
    assert np.isclose(my_lin.slope, 1.9249174584304438)
    assert np.isclose(my_lin.intercept, 5.211786196055158)

    # 2. assert against a known correct implementation
    # assert against slope and intercept values
    # from sci-kit learn's LinearRegression
    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    #[0] because only 1 coefficient because only 1 feature
    assert np.isclose(my_lin.slope, sk_lin_reg.coef_[0])
    assert np.isclose(my_lin.intercept, sk_lin_reg.intercept_)
    
    # TODO: should probably add more test cases (zero slope)
   
#test for mismatched lengths
def test_fit_mismatch_length():
    X_train1 = [[1], [2]]
    y_train1= [2]
    my_reg=MySimpleLinearRegressor()

    with pytest.raises(ValueError, match="X and y must have same length"):
        my_reg.fit(X_train1, y_train1)
#test for where all X values are the same   
def test_fit_all_x_same_raises():
    X = [[3], [3], [3],[3]]
    y = [2, 4, 6,8]
    #TODO if all X values are the same raises a ValueError.
    pass
    

  
# test for test_mysimplelinearregressor_predict()
def test_mysimplelinearregressor_predict():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    my_lin = MySimpleLinearRegressor()
    my_lin.fit(X_train, y_train)
    sklearnm=LinearRegression()
    sklearnm.fit(X_train,y_train)

    # Test prediction on a single point
    X_test = [[150]]
    y_pred = my_lin.predict(X_test)
    y_pred_m=sklearnm.predict(X_test)
    assert np.isclose(y_pred, y_pred_m)

    # TODO: Add more tests for multiple points, edge cases

# test driven development (TDD)
# write the unit tests before the units themselves
# often makes writing the units go smoother because you
# have a deep understanding of what makes them "correct"