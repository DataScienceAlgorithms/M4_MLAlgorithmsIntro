from my_simple_linear_regressor import MySimpleLinearRegressor
from my_linear_regression_gd import MyLinearRegressionGD
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier
import operator

import numpy as np

def test_compute_euclidean_distance():
    np.random.seed(0)
    v1=np.random.random(10)
    v2=np.random.random(10)
    dist=compute_euclidean_distance(v1,v2)
    sp_dist=euclidean(v1,v2)
    assert np.isclose(dist,sp_dist)


def compute_euclidean_distance(v1,v2):
    return np.sqrt(sum([(a-b)**2 for a,b in zip(v1,v2)]))


# Note: this function will be super handy for PA5
def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

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
      
    y_pred=my_lin.predict([[8]])
    print(y_pred)


    np.random.seed(0)
    X_train = np.random.rand(100, 2) * 100   # Two independent features
    y_train = 2 * X_train[:, 0] + 5 * X_train[:, 1] + np.random.normal(0, 10, 100)
    print("X_train:", X_train[:5])
    print("y_train:", y_train[:5])


    mylin=MyLinearRegressionGD()
    mylin.fit(X_train,y_train)
    print(mylin.slopes)
    print(mylin.intercept)
    y_pred= mylin.predict([[4,50]])
    print(y_pred)


 

    # PA4 kNN starter code
    header = ["att1", "att2"]
    X_train = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]
        ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    unseen_instance = [2, 3]
    

    # 1. normalize if needed (not needed for dataset, assumed [0, 10])
    # 2. calculate distances between each training set instance
    # and the unseen instance
    row_dist=[]
    k=3
    for i,row in enumerate(X_train):
        dist=compute_euclidean_distance(row,unseen_instance)
        row_dist.append((i,dist))
    
    #sort the list of (index, distance) pairs by distance value (second element)
    row_dist.sort(key=operator.itemgetter(1))
    top_k= row_dist[:k]
    print(top_k)
    
    # top k closes neighbors' y labels from y_train
    # then use majority voting to find the prediction for this test instance
    # can make use of get_frequencies()
    y_labels=[y_train[i] for i,_ in top_k]
    freq={}
    for i in y_labels:
        freq[i]=freq.get(i,0)+1
    y_pred=max(freq,key=freq.get)
    print(y_pred)
   
    

    # check work against sci-kit learn
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([unseen_instance])
    print(distances)
    print(indexes)

    np.random.seed(0)
    print("randomizing in place")
    randomize_in_place(X_train, y_train)
    for i in range(len(X_train)):
        # check they are still parallel
        print(X_train[i], y_train[i])


    
if __name__=='__main__':
    main()