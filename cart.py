import numpy as np
import math
class CartTree:
    def __init__(self,
                 max_depth,
                 min_samples_leaf):
                 
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
        """
        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1)
        return self

    def _best_thresh(self, X, y, index):

        # YOUR CODE HERE
        # begin answer
        x=X[:,index]
        values=np.unique(x)

        min_loss = np.inf
        t = values[0]
        for i in range(len(values)):
            L = np.where(x<=values[i])
            H = np.where(x>values[i])
            y_L = y[L]
            y_H = y[H]
            loss1 = self.cal_loss(y_L)
            loss2 = self.cal_loss(y_H) if len(y_H)>0 else 0
            loss = loss1+loss2
            if loss<min_loss:
                min_loss = loss
                t = values[i]
        # end answer
        return min_loss, t
    
    def cal_loss(self, y):
        m = np.mean(y)
        y_ = y-m
        y_=y_*y_
        return np.sum(y_)
    
    def _choose_best_feature(self, X, y):

        best_feature_idx = 0
        min_loss = np.inf
        N,D = X.shape
        t=None
        for i in range(D):
            loss, t = self._best_thresh(X,y,i)
            if loss<min_loss:
                min_loss=loss
                thresh = t
                best_feature_idx = i
        # end answer
        return best_feature_idx, thresh
    

    def _split_dataset(self, X, y, index, thresh):
        
        x=X[:,index]
        L = np.where(x<=thresh)
        R = np.where(x>thresh)
        X_L=X[L]
        y_L = y[L]
        X_r = X[R]
        y_r = y[R]

        X_L=np.delete(X_L,index, axis = 1)
        X_r=np.delete(X_r,index, axis = 1)
        return X_L, y_L, X_r, y_r

    def _build_tree(self, X, y, feature_names, depth):

        mytree = dict()
 
        if depth == self.max_depth+1 or X.shape[0]<=self.min_samples_leaf or len(np.unique(y))<=1:#requirement
            return np.mean(y)
        else:
            if X.shape[1]==0:
                return np.mean(y)
            index, thresh= self._choose_best_feature(X,y)
            key=feature_names[index]
            x=X[:,index]
            f_n=[feature_names[i] for i in range(len(feature_names)) if i!=index]
            value_dic = dict()
            X_L, y_L, X_r, y_r = self._split_dataset(X,y,index,thresh)
            value_dic[thresh]=self._build_tree(X_L, y_L,f_n,depth+1)
            value_dic[thresh+1]=self._build_tree(X_r, y_r,f_n,depth+1)
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
                if x[lable]<=values[0]:
                    dic = dic[values[0]]
                else:
                    dic = dic[values[1]]
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