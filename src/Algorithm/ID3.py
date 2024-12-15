import numpy as np
from collections import Counter
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index/name used for splitting
        self.threshold = threshold  # Threshold value for numerical features
        self.left = left          # Left child node
        self.right = right        # Right child node
        self.value = value        # Predicted class (for leaf nodes)

class ID3:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        # Split the data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        if feature_idx is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Create child nodes
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature_idx, threshold, left, right)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        X = np.array(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)