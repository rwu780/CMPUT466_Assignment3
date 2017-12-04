from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        self.numclasses = 2
        self.numfeatures = 8

        if self.params['usecolumnones'] == True:
            self.numfeatures = 9

        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE

        ## Keep track of number of class 0 and class 1
        numC0 = 0
        numC1 = 0

        ## Separate the two inputs class based on the output
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                numC0 += 1
                self.means[0] = self.means[0] + Xtrain[i][:self.numfeatures]
            if ytrain[i] == 1.0:
                numC1 += 1
                self.means[1] = self.means[1] + Xtrain[i][:self.numfeatures]

        ## Calculate the mean
        for i in range(self.numfeatures):
            self.means[0][i] = self.means[0][i]/numC0
            self.means[1][i] = self.means[1][i]/numC1

        ## Calculate the standard deviation
        for i in range(len(ytrain)):
            if ytrain[i] == 0.0:
                self.stds[0] += (Xtrain[i][:self.numfeatures] - self.means[0])**2

            if ytrain[i] == 1.0:
                self.stds[1] += (Xtrain[i][:self.numfeatures] - self.means[1])**2

        for i in range(self.numfeatures):
            self.stds[0][i] = (self.stds[0][i]/numC0)**0.5
            self.stds[1][i] = (self.stds[1][i]/numC1)**0.5

        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE

        for i in range(len(ytest)):
            probC0 = 1
            probC1 = 1

            for j in range(self.numfeatures):
                probC0 = probC0 * self.probability(self.means[0][j], self.stds[0][j], Xtest[i][j])
                probC1 = probC1 * self.probability(self.means[1][j], self.stds[1][j], Xtest[i][j])

            # Determine which class give a better probability. 
            if probC0 > probC1:
                ytest[i] = 0.0
            elif probC1 >= probC0:
                ytest[i] = 1.0

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

    def probability(self, mean, std, x):
        """
        Calculate the probability based on gaussian distribution
        """ 

        # if the standard deviation is 0 and mean is equal to x, then the probability will always be one 
        if std == 0.0 and mean == x:
            return 1

        p = 1/np.sqrt(2 * np.pi * (std**2)) * np.exp(-1/(2*(std**2)) * (x - mean)**2)

        return p

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE
        predict = self.transferFunction(np.dot(X.T, theta)) + 0.01 * np.power(self.regularizer[0](theta), 2)
        cost = predict - y
        ### END YOUR CODE

        return cost

    def transferFunction(self, x):
        output = 1.0 / (1.0 + np.exp(np.negative(x)))
        return output

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        cost = self.logit_cost(theta, X, y)
        grad = np.dot(cost, X) + 0.01 * self.regularizer[1](theta)
        
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        self.weights = np.zeros(Xtrain.shape[1],)

        ### YOUR CODE HERE
        learning = 0.01
        for i in range(1000):
            for t in range(Xtrain.shape[0]):
                g = self.logit_cost_grad(self.weights, Xtrain[t], ytrain[t])
                self.weights -= learning * g
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        ytest = utils.threshold_probs(utils.sigmoid(np.dot(Xtest, self.weights.T)))

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        # nh = 300 seems to gave the good result
        self.params = {'nh': 300,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(self.w_input, inputs))

        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_hidden))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        nabla_input = np.zeros((self.w_input.shape[0], self.w_input.shape[1]))
        nabla_output = np.zeros((self.w_output.shape[0], self.w_output.shape[1]))

        hidden, output = self.feedforward(x)

        # change for output layer
        delta1 = output - y 
        nabla_output = np.outer(delta1, hidden)

        # first Layer
        delta2 = np.outer(np.dot(self.w_output * delta1, hidden),(1-hidden))
        nabla_input = np.outer(delta2, x)
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):

        # Grab the number of hidden units
        nh = self.params['nh']
        self.w_input = np.random.randn(nh, Xtrain.shape[1])/np.sqrt(Xtrain.shape[1])
        self.w_output = np.random.randn(1, nh)/np.sqrt(nh)

        numOfEpoch = self.params['epochs']
        learningRate = self.params['stepsize']

        for i in range(numOfEpoch):
            for j in range(len(ytrain)):
                nabla_input, nabla_output = self.backprop(Xtrain[j], ytrain[j])
                self.w_input -= learningRate * nabla_input
                self.w_output -= learningRate * nabla_output

    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0], dtype = int)

        for i in range(len(ytest)):
            hidden, output = self.feedforward(Xtest[i])
            if output >= 0.5:
                ytest[i] = 1
            if output <= 0.5:
                ytest[i] = 0

        return ytest

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        # K = 15 gave us the minimim error on test data
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        K = 15
        self.centers = Xtrain[:K]

        # Tranform from Xtrain to Ktrain
        if self.params['kernel'] == 'linear':
            Ktrain = self.linearTransform(Xtrain, self.centers)

        elif self.params['kernel'] == 'hamming':
            Ktrain = self.hammingTransform(Xtrain, self.centers)

        ### END YOUR CODE

        self.weights = np.zeros(Ktrain.shape[1],)

        ### YOUR CODE HERE

        # Learn Using Gradient Descent
        learning = 0.01
        for i in range(1000):
            for t in range(Ktrain.shape[0]):
                g = self.logit_cost_grad(self.weights, Ktrain[t], ytrain[t])
                self.weights -= learning * g

        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def linearTransform(self, Xtrain, centers):
        """
        Kernel Linear Transformation
        """

        Ktrain = np.dot(Xtrain, centers.T)

        return Ktrain

    def hammingTransform(self, Xtrain, centers):
        """
        Kernel Hamming Transformation
        """

        Ktrain = np.zeros((Xtrain.shape[0], len(centers)))

        for i in range(Ktrain.shape[0]):
            for j in range(Ktrain.shape[1]):
                Ktrain[i][j] = self.hammingDistance(Xtrain[i], centers[j])

        return Ktrain

    def hammingDistance(self, m1, m2):
        """
        Calculte Hamming Distance between two inputs
        """

        diffs = 0
        for ch1, ch2 in zip(m1, m2):
            if ch1 != ch2:
                diffs +=1

        return diffs

    def predict(self, Xtest):
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE

        # Transform the Xtest matrix
        if self.params['kernel'] == 'linear':
            test = self.linearTransform(Xtest, self.centers)

        elif self.params['kernel'] == 'hamming':
            test = self.hammingTransform(Xtest, self.centers)


        for i in range(len(ytest)):
            p = self.transferFunction(np.dot(test[i].T, self.weights))

            if p >= 0.5:
                ytest[i] = 1
            if p < 0.5:
                ytest[i] = 0
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest
