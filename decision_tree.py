import numpy as np
import math

class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        C=np.unique(y)
        if len(C)>1:#not pure
            S=[0 for i in range(len(C))]
            for i in range(len(C)):
                index = np.where(y==C[i])
                S[i]=np.sum(sample_weights[index])
            tot=np.sum(S)
            S=S/tot
            for i in range(len(S)):
                entropy = entropy - S[i]*math.log(S[i],2)
        # end answer
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        tot = np.sum(sample_weights)
        x=X[:,index]
        values=np.unique(x)
        if len(values)>1:
            entropy=self.entropy(y,sample_weights)
            S=[0 for i in range(len(values))]
            for i in range(len(values)):
                index = np.where(x==values[i])
                y_ = y[index]
                weights = sample_weights[index]
                Sv=np.sum(weights)
                S[i]=(Sv/tot)*self.entropy(y_, weights)
            info_gain=entropy-np.sum(S)
        # end answer
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0
        split_information = 0.0
        # YOUR CODE HERE
        # begin answer
        x=X[:,index]
        values=np.unique(x)
        tot=np.sum(sample_weights)
        if len(values)>1:
            IG=self._information_gain(X,y, index, sample_weights)
            S=[0 for i in range(len(values))]
            for i in range(len(values)):
                index = np.where(x==values[i])
                weights = sample_weights[index]
                Sv= np.sum(weights)
                S[i]=(-Sv/tot)*math.log(Sv/tot,2)
            IV=np.sum(S)
            info_gain_ratio=IG/IV
        # end answer
        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        tot = np.sum(sample_weights)
        C=np.unique(y)
        if len(C)>1:
            for i in range(len(C)):
                index = np.where(y==C[i])
                p=np.sum(sample_weights[index])/tot
                gini-=p**2
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        x=X[:,index]
        values=np.unique(x)
        tot=np.sum(sample_weights)
        S=[0 for i in range(len(values))]
        for i in range(len(values)):
            index = np.where(x==values[i])
            y_ = y[index]
            weights = sample_weights[index]
            Sv=np.sum(weights)
            S[i]=Sv*self.gini_impurity(y_,weights)/tot
        new_impurity=np.sum(S)
        # end answer
        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        x=X[:,index]
        pos = np.where(x==value)
        sub_X=X[pos]
        sub_y = y[pos]
        sub_sample_weights=sample_weights[pos]
        sub_X=np.delete(sub_X,index, axis = 1)

        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        
        N,D = X.shape
        if self.sample_feature:
            numOfSample = int(math.sqrt(D))
            sam_f=np.random.choice(D, numOfSample, False)
            X_s = X[:,sam_f]
            S=[self.criterion(X_s,y,i,sample_weights) for i in range(numOfSample)]

        else:
            S=[self.criterion(X,y,i,sample_weights) for i in range(D)]
        if self.criterion==self._gini_purification:
            best_feature_idx=S.index(min(S))
        else:
            best_feature_idx=S.index(max(S))#not suitable for gini

        best_feature_idx= sam_f[best_feature_idx] if self.sample_feature else best_feature_idx
        # end answer
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        C=np.unique(y)
        if len(C)>1:
            S=[0 for i in range(len(C))]
            for i in range(len(C)):
                index = np.where(y==C[i])
                S[i]=np.sum(sample_weights[index])
            index = S.index(max(S))
            majority_label=C[index]
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, 
                the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf,
                 we use the majority_label by calling majority_vote.
                mytree = {
                    'title': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful.
        # begin answer
        if depth == self.max_depth+1 or X.shape[0]<=self.min_samples_leaf or len(np.unique(y))<=1:#requirement
            return self.majority_vote(y,sample_weights)
        else:
            if X.shape[1]==0:
                return self.majority_vote(y,sample_weights)
            index= self._choose_best_feature(X,y,sample_weights)
            key=feature_names[index]
            x=X[:,index]
            values = np.unique(x)
            f_n=[feature_names[i] for i in range(len(feature_names)) if i!=index]
            value_dic = dict()
            for value in values:
                sub_X, sub_y, sub_sample_weights=self._split_dataset(X,y,index,value,sample_weights)
                value_dic[value]=self._build_tree(sub_X, sub_y,f_n,depth+1,sub_sample_weights)
            mytree[key]=value_dic
        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            dic = tree
            while isinstance(dic, dict):
                keys = list(dic.keys())
                lable = keys[0]
                dic = dic[lable]
                values = list(dic.keys())
                noValue = True
                for i in range(len(values)):
                    if values[i]==x[lable]:
                        noValue=False
                        dic = dic[values[i]]
                if noValue:
                    value = np.random.choice(values)
                    dic=dic[value]
            return dic
            # end answer

        # YOUR CODE HERE
        # begin answer
        y = [_classify(self._tree,X.iloc[i]) for i in range(X.shape[0])]
        return np.array(y)
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
