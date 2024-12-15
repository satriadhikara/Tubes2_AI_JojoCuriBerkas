import numpy as np
from collections import defaultdict
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Tuple, Union, Optional

class NBayes:
    def __init__(self):
        self.class_probabilities: Dict = {}
        self.mean: Dict = {}
        self.variance: Dict = {}
        self._epsilon = 1e-10
        
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or infinity values")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_input(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        # Vectorized class probability calculation
        class_counts = np.bincount(y)
        self.class_probabilities = dict(zip(self.classes, class_counts / n_samples))
        
        # Vectorized mean and variance calculation
        self.mean = {}
        self.variance = {}
        for label in self.classes:
            mask = (y == label)
            class_samples = X[mask]
            self.mean[label] = np.mean(class_samples, axis=0)
            self.variance[label] = np.var(class_samples, axis=0) + self._epsilon

    @lru_cache(maxsize=1024)
    def _gaussian_probability(self, x: float, mean: float, variance: float) -> float:
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent

    def _calculate_class_probability(self, features: np.ndarray, label: int) -> float:
        prob = np.log(self.class_probabilities[label])
        mean = self.mean[label]
        variance = self.variance[label]
        
        # Vectorized probability calculation
        log_probs = np.sum(
            np.log(
                (1 / np.sqrt(2 * np.pi * variance)) * 
                np.exp(-((features - mean) ** 2) / (2 * variance))
            )
        )
        return prob + log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        predictions = np.zeros(X.shape[0], dtype=int)
        
        # Parallel prediction using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, sample in enumerate(X):
                future = executor.submit(self._predict_single, sample)
                futures.append((i, future))
            
            for i, future in futures:
                predictions[i] = future.result()
        
        return predictions