README for Assignemtn3

This code was run and test under Python 2.7

run code: python script_classify.py

1.
a) If we include the columns of ones, and the program will run into error. Because if we add a column of ones, the mean for that column will be one, and lead to standard deviation of 0. When calculating the probability for Gaussian distribution, the formula will be dividing by 0, which will result in an error.

d)
If the dataset can be linearly separable, then the algorithms able to predict good result.

With naive bayes, it calculated the means and std for each clusters. Then to predict new y, simply calculate the probabilities of each features respect to different clusters. The clusters with the highest probabilities will become the new prediction.
The average error for naive Bayes is 25.7

For logistic regression, it gave a smaller error without regularization. With regularizer, the gradient will move slower than the gradient without the regularizer, and if the optimal weight is far away from our initial weights, then regression with regularizer give us a larger error. If the initial weight did not set to 0, then the algorithm might gave us a better error than before

The average error for logistic regression without regularization is 23.52

The average error logistic regression with l2 regularizer is 38.56

For neural network, the number of hidden units increase, the better it fit to the dataset. If increase the number of hidden layers, the weights will be overfit to the training dataset, and might gave a larger error on the test data. If decrease the number of hidden layers, it will not overfit to the training set, but it unable to minimize the error. Hidden layers = 300 seems to provide a good prediction while not overfitting to the training data.

The average error for neural network is 26.82 with 300 hidden layers


2.
a)The performance was slightly better than the classifier implemented in Q1 with k = 15. With other k values might resulted in greater error values. 

If enters is the training data, then the amount of computation is large. Selecting first k(15) data points as enters seem to provide a good predictions. 

b)
The performance was very good compare to the random predictor. Kernel Logistic Regression error values is less than half of the error produce from random predictor
The Kernel Logistic Regression with Hamming distance had an average error of 21.96.
Random predictor had an average error of 48.58



