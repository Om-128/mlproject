import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evalute_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            ''' Create a dictionary of models to train '''
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            '''     Hyperparameter tuning for the models    '''
            params = {
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        "splitter": ["best", "random"],
                        },
                    "Random Forest": {
                        "n_estimators": [10, 50, 100],
                        "criterion": ["squared_error", "absolute_error"],
                    },
                    "Gradient Boosting": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "Linear Regression": {
                        "fit_intercept": [True, False],
                        "positive": [True, False]
                    },
                    "XGBoost": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "CatBoost": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100]
                    },
                    "K-Neighbors": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "p": [1, 2]
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0]
                    }
                }


            model_report: dict = evalute_model(x_train=x_train, 
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            models=models,
            param=params
            )

            ''' Get the best model based on R2 score '''
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found for both training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
