"""
my_linear_regressor_gd.py
@author: Sophina Luitel

A linear regression implementation using gradient descent.
"""

class MyLinearRegressionGD:
    """
    A linear regression model that fits a line to data with two features
    using gradient descent optimization.
    
    Equation:
        y = slope_1 * x_1 + slope_2 * x_2 + intercept

    Attributes:
        slopes (list of numerica val): List containing slopes (coefficients) for each feature.
        intercept (numeric val): Intercept (bias) term.
        learning_rate (float): Step size for gradient descent updates.
        iterations (int): Number of gradient descent iterations during training.
    """
    def __init__(self, slopes=None, intercept=0.0, learning_rate=0.0001, iterations=100):
        if slopes is None:
            self.slopes = [0.0, 0.0]  # Two features
        else:
            self.slopes = slopes
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.iterations = iterations
           

    def fit(self, X_train, y_train):
        """
       Fit the linear regression model to the training data using gradient descent.

        Args:
            X_train (list of list of numeric vals): Training data samples, each with two features.
                Shape: (n_samples, 2)
                Example: [[1, 2], [3, 4], ...]

            y_train (list of numeric vals): Target values corresponding to each training sample.
                Shape: (n_samples,)
                Example: [5, 11, ...]

        Updates:
            self.slopes and self.intercept to minimize the mean squared error.
        """
        n = len(y_train)
        for _ in range(self.iterations):
            y_pred = [
                self.slopes[0] * x[0] + self.slopes[1] * x[1] + self.intercept
                for x in X_train
            ]
            error = [y_pred[i] - y_train[i] for i in range(n)]

            # Gradients
            db = sum(error) / n
            dm1 = 2*sum(error[i] * X_train[i][0] for i in range(n)) / n
            dm2 = 2*sum(error[i] * X_train[i][1] for i in range(n)) / n

            # Update slopes and intercept
            self.intercept -= self.learning_rate * db
            self.slopes[0] -= self.learning_rate * dm1
            self.slopes[1] -= self.learning_rate * dm2
    def predict(self, X_test):
        """
        Predict values for test data.
        Args:
            X_test (list of list of numeric vals): Test samples, each with two features.
                Shape: (n_samples, 2)
                Example: [[1, 2], [3, 4], ...]

        Returns:
            list of numeric vals: Predicted target values.

        Raises:
            ValueError: If the model has not been fitted (slopes or intercept are None).
        """
        if self.slopes is None or self.intercept is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")

        return [
            self.slopes[0] * x[0] + self.slopes[1] * x[1] + self.intercept
            for x in X_test
        ]
