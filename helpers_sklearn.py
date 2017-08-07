import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split


class HelperClass:
    
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42, stratify=self.y)
    
    def score(self, est, params, cv=5):
        return  GridSearchCV(est, param_grid=params, cv=cv, n_jobs=-1).fit(self.X_train, self.y_train).score(self.X_test, self.y_test)
    
    def train_model(self, est, params, cv=5):
        return GridSearchCV(est, param_grid=params, cv=cv, n_jobs=-1).fit(self.X, self.y).best_estimator_


def nested_cross_val_score(X, y, est, params, cv_inner, cv_outer):
    grid = GridSearchCV(est, param_grid=params, cv=cv_inner, n_jobs=-1)
    return cross_val_score(grid, X, y, cv=cv_outer)

