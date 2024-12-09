import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!
 
        w_opt = None
        # ====== YOUR CODE: ======
        reg_lambda = self.reg_lambda

        # Closed form solution
        N, d = X.shape

        # Regularization term: Identity matrix of shape (d, d)
        I = np.eye(d)

        # Compute the optimal weights using the closed-form solution
        reg_mat = reg_lambda * N * I
        reg_mat[0, 0] = 0  # no regularization to the bias term
        
        w_opt = np.linalg.inv(X.T @ X + reg_mat) @ X.T @ y # (X^T * X + lambda * N * I)^-1 * X^T*Y
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    X = df.drop(target_name, axis=1) if feature_names is None else df[feature_names]
    y = df[target_name]
    y_pred = model.fit_predict(X.values, y)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        bias_column = np.ones((X.shape[0], 1))  # Create a column of ones
        xb = np.hstack((bias_column, X))       # Horizontally stack the bias column and X
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # Define indices for specific features in the Boston dataset
        self.bias_offset = 1  # Adjust for the added bias column
        self.crim_index = 0 + self.bias_offset  # Crime rate feature
        self.chas_index = 3 + self.bias_offset  # Charles River feature
        self.lstat_index = 12 + self.bias_offset  # Lower status population percentage
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        X_transformed = np.array(X, copy=True)

        # Apply log transformation to specified features
        X_transformed[:, self.crim_index] = np.log(X_transformed[:, self.crim_index])
        X_transformed[:, self.lstat_index] = np.log(X_transformed[:, self.lstat_index])

        # Remove the 'chas' feature 
        X_transformed = np.delete(X_transformed, self.chas_index, axis=1)

        # Generate polynomial features
        poly = sklearn.preprocessing.PolynomialFeatures(degree=self.degree, include_bias=False)
        X_transformed = poly.fit_transform(X_transformed)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    
    # Calculate correlations with the target feature
    corr= df.corr()[target_feature].drop(target_feature)

    # Sort correlations by absolute value in descending order
    sorted_corr = corr.abs().sort_values(ascending=False)

    # Select the top n features
    top_n_features = sorted_corr.head(n).index.tolist()
    top_n_corr = corr.loc[top_n_features].tolist()

    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    # Mean Squared Error 
    mse = np.square(np.subtract(y,y_pred)).mean() 
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    sum_se = np.square(np.subtract(y,y_pred)).sum()
    ms_diff_mean = np.square(y-y.mean()).sum() 
    r2 = 1 - sum_se/ms_diff_mean
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'bostonfeaturestransformer__degree': degree_range,  # Polynomial degree
        'linearregressor__reg_lambda': lambda_range,             # Regularization parameter
    }

    # Define the scoring metric (Mean Squared Error in this case)
    scorer = sklearn.metrics.make_scorer(mse_score, greater_is_better=False)

    # Initialize GridSearchCV
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=k_folds,
        n_jobs=-1  # Use all available cores for parallel processing
    )

    # Perform the grid search
    grid_search.fit(X, y)

    # Extract the best hyperparameters from the search
    best_params = grid_search.best_params_

    # ========================

    return best_params
