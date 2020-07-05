import copy
import numpy as np
import math


class Adaboost:
    '''Adaboost Classifier.

    Note that this class only support binary classification.
    '''

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in Adaboost.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        y_=np.where(y==1,1,-1)
        sample_weights = np.ones(X.shape[0]) / X.shape[0]
        for i in range(self.n_estimator):
            self._estimators[i].fit(X,y_,sample_weights)
            y_pred = self._estimators[i].predict(X)
            index = np.where(y_pred!=y_)
            e_i = np.sum(sample_weights[index])/np.sum(sample_weights)
            self._alphas[i]=math.log((1-e_i)/e_i)
            sample_weights[index]= (1-e_i)/e_i*sample_weights[index]
        # end answer
        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)
        # YOUR CODE HERE
        # begin answer
  
        for i in range(self.n_estimator):
            y= self._estimators[i].predict(X)
        
            y_pred+=np.array(self._alphas[i])*y
        y_pred=np.where(y_pred>0, 1, 0)
        # end answer
        
        return y_pred
