import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

# KNN mean imputation

def knn_imputation(df, col):
    
    # Extract features and target variable
    
    y = df[col].values
    X = df.drop([col], axis=1).values
    missings_indices = np.isnan(y)
    train_indices = np.logical_not(missings_indices)
    X_train, y_train, X_missings = X[train_indices, :], y[train_indices], X[missings_indices, :]
    
    
    # KNN regression set up

    pipe_knn = make_pipeline(StandardScaler(), PCA(random_state=42), KNeighborsRegressor())


    params_grid = [{'pca__n_components' : np.arange(1, X.shape[1] + 1),
                    'kneighborsregressor__n_neighbors' : np.arange(1,21)}]


    # Predict and return
    
    y[missings_indices] = GridSearchCV(pipe_knn, param_grid=params_grid, cv=10, n_jobs=-1).fit(X_train, y_train).predict(X_missings)
    
    return pd.Series(y, index=df.index)