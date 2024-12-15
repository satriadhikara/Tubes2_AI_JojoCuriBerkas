import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):
        if k < 1 or not isinstance(k, int):
            raise ValueError("Invalid k. k must be an integer greater than 0.")
        if metric not in ['manhattan', 'euclidean', 'minkowski'] or not isinstance(metric, str):
            raise ValueError("Invalid metric. Valid metrics are 'manhattan', 'euclidean', and 'minkowski'.")
        if p < 1 or not isinstance(p, (int, float)):
            raise ValueError("Invalid p. p must be a number greater than 0.")
        if weights not in ['uniform', 'distance']:
            raise ValueError("Invalid weights. Valid values are 'uniform' and 'distance'.")
        if not isinstance(weights, str):
            raise ValueError("Invalid weights. weights must be a string.")
        if n_jobs < 1 and n_jobs != -1 or not isinstance(n_jobs, int):
            raise ValueError("Invalid n_jobs. Must be an integer greater than 0, or -1 to use all available cores.")
        if not isinstance(verbose, bool):
            raise ValueError("Invalid verbose. verbose must be a boolean.")
        
        self.k = k
        self.verbose = verbose
        self.metric = metric
        self.weights = weights
        self.p = p if metric == 'minkowski' else (1 if metric == 'manhattan' else 2)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def _compute_distances(self, test):
        distances = np.linalg.norm(self.X_train - test, ord=self.p, axis=1)
        return distances
    
    def _get_nearest_neighbours(self, test):
        distances = self._compute_distances(test)
        indices = np.argsort(distances)[:self.k]
        
        if self.weights == 'distance':
            distances = distances[indices]
            weights = 1 / (distances + 1e-10)  # Avoid division by zero
            weights /= np.sum(weights)
        else:
            weights = np.ones_like(indices, dtype=float) / self.k  # Uniform weights
        
        return indices, weights
    
    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values.astype(float)
        else:
            self.X_train = np.array(X_train).astype(float)
        self.y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    
    def _predict_instance(self, row):
        indices, weights = self._get_nearest_neighbours(row)
        labels = self.y_train.iloc[indices].values
        
        if self.weights == 'uniform':
            labels = [tuple(label) if isinstance(label, np.ndarray) else label for label in labels]  # Make labels hashable
            prediction = max(set(labels), key=labels.count)
        else:  # Weighted voting
            label_weight_dict = {}
            for i, label in enumerate(labels):
                label = tuple(label) if isinstance(label, np.ndarray) else label  # Make labels hashable
                label_weight_dict[label] = label_weight_dict.get(label, 0) + weights[i]
            prediction = max(label_weight_dict, key=label_weight_dict.get)
        
        return prediction
    
    def predict(self, X_test):
        if self.verbose:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")
        
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values.astype(float)
        else:
            X_test = np.array(X_test).astype(float)
        
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._predict_instance, X_test), total=len(X_test))) if self.verbose else list(executor.map(self._predict_instance, X_test))
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")
        
        return np.array(results)
    
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
