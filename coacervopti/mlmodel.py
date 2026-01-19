#!

import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, List, Dict, Union, Protocol, Optional, Any

# Generic MLModel class

class MLModel(Protocol):
    def train(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def __repr__(self) -> str: ...

# - GAUSSIAN PROCESS REGRESSOR WITH INTEGRATED GRIDSEARCH

class GPR:
    """
    Gaussian process regressors with hyperparameter tuning using GridSearchCV
    """
    def __init__(self, 
                 log_transform: bool = False, 
                 use_gridsearch: bool = True,
                 param_grid: Optional[Dict[str, Any]] = None,
                 cv: int = 5,
                 scoring: str = 'neg_mean_squared_error',
                 n_jobs: int = -1,
                 verbose: int = 0,
                 **kwargs) -> None:
        
        self.log_transform = log_transform
        self.eps = 1e-8
        self.use_gridsearch = use_gridsearch
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Handle different scoring options
        if scoring == 'rmse' or scoring == 'neg_rmse':
            # Create custom RMSE scorer (negated for GridSearchCV maximization)
            self.scoring_func = make_scorer(
                lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), 
                greater_is_better=False
            )
        elif scoring == 'neg_root_mean_squared_error':
            # Use sklearn's built-in RMSE scorer (available in sklearn >= 0.22)
            self.scoring_func = 'neg_root_mean_squared_error'
        else:
            # Use provided scoring metric
            self.scoring_func = scoring
        
        # Default parameter grid if none provided
        if param_grid is None:
            self.param_grid = {
                'kernel' : [
                    ConstantKernel(1.0) * RBF(lc) + WhiteKernel(noise_level=noise)
                    for lc in [0.01, 0.1, 1.0, 10]
                    for noise in [.75, .5, .25, 0.1]
                ],
                'alpha': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
                'normalize_y': [True, False],
                'n_restarts_optimizer': [50]
            }
        else:
            self.param_grid = param_grid
        
        # Initialize base model
        self.base_model = GaussianProcessRegressor(**kwargs)
        
        # Initialize GridSearchCV if requested
        if self.use_gridsearch:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring_func,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            self.model = self.base_model
    
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the model with optional hyperparameter tuning"""
        if self.log_transform:
            y = np.log10(y + self.eps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            # warnings.filterwarnings("ignore", category=Warning)
            self.model.fit(x, y)
        
        # Store best parameters if using gridsearch
        if self.use_gridsearch:
            self.best_params_ = self.model.best_params_
            self.best_score_ = self.model.best_score_
            self.cv_results_ = self.model.cv_results_
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        # Get the best estimator if using gridsearch
        if self.use_gridsearch:
            predictor = self.model.best_estimator_
        else:
            predictor = self.model
        
        y_hat_mean, y_hat_uncertainty = predictor.predict(x, return_std=True)
        
        if self.log_transform:
            # Store the log-scale predictions
            y_hat_log = y_hat_mean
            # Convert mean back to original scale
            y_hat_mean = 10 ** y_hat_log
            # Apply delta method using log-scale values
            # Derivative of 10^x is 10^x * ln(10)
            derivative = y_hat_mean * np.log(10)  # This is 10^y_hat_log * ln(10)
            variance_original = (derivative ** 2) * (y_hat_uncertainty ** 2)
            y_hat_uncertainty = np.sqrt(variance_original)
        
        # Dummy variable as they are identical
        # in Ensemble methods y_hat is the set of predictions
        y_hat = y_hat_mean
        return y_hat, y_hat_mean, y_hat_uncertainty
    
    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found during gridsearch"""
        if self.use_gridsearch and hasattr(self, 'best_params_'):
            return self.best_params_
        else:
            return {}
    
    def get_best_score(self) -> float:
        """Return the best cross-validation score"""
        if self.use_gridsearch and hasattr(self, 'best_score_'):
            return self.best_score_
        else:
            return None
    
    def get_cv_results(self) -> Dict[str, Any]:
        """Return detailed cross-validation results"""
        if self.use_gridsearch and hasattr(self, 'cv_results_'):
            return self.cv_results_
        else:
            return {}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test set"""
        _, y_pred, _ = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def __repr__(self) -> str:
        try:
            if self.use_gridsearch:
                best_kernel = self.model.best_estimator_.kernel_
                return f"GPR(log_transform={self.log_transform}, gridsearch=True, best_kernel={best_kernel}, trained=True)"
            else:
                return f"GPR(log_transform={self.log_transform}, gridsearch=False, kernel={self.model.kernel_}, trained=True)"
        except AttributeError:
            return f"GPR(log_transform={self.log_transform}, gridsearch={self.use_gridsearch}, trained=False)"


class KernelFactory:
    '''
    KernelFactory usage.

    Example:
    kernel_recipe = ['+', {'type': 'C', 'constant_value': 2.0}, ['*', 'RBF', {'type': 'W', 'noise_level': 1.0}]]
    kernel_factory = KernelFactory(kernel_recipe)
    kernel = kernel_factory.get_kernel()
    -> kernel = 1.41**2 + RBF(length_scale=1) * WhiteKernel(noise_level=1)

    '''
    def __init__(self, kernel_recipe: List[Union[str,Dict]]):
        self.kernel_recipe = kernel_recipe

    def get_kernel(self):
        kernel_map = {
            'RBF': RBF,
            'Matern': Matern,
            'RationalQuadratic': RationalQuadratic,
            'C': ConstantKernel,
            'W': WhiteKernel,
        }
        
        return self._parse_kernel(self.kernel_recipe, kernel_map)
    
    def _parse_kernel(self, kernel_recipe, kernel_map):
        # Simple string case
        if isinstance(kernel_recipe, str):
            return kernel_map[kernel_recipe]()

        # Dictionary with parameters
        elif isinstance(kernel_recipe, dict):
            kernel_type = kernel_recipe.pop('type')
            return kernel_map[kernel_type](**kernel_recipe)

        # Composite kernel
        elif isinstance(kernel_recipe, list):
            operator = kernel_recipe[0]
            first_kernel = self._parse_kernel(kernel_recipe[1], kernel_map)
            second_kernel = self._parse_kernel(kernel_recipe[2], kernel_map)
            
            if operator == '*':
                return first_kernel * second_kernel
            elif operator == '+':
                return first_kernel + second_kernel
            else:
                raise ValueError(f"Unknown operator: {operator}")

        else:
            raise ValueError(f"Invalid kernel definition: {kernel_recipe}")
        
# --------------------------------------------------------------