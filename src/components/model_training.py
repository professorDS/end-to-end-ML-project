import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grids = {
                            "Random Forest": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                            },

                            "Decision Tree": {
                            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                            "splitter": ["best", "random"],
                            "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                            },

                            "Gradient Boosting": {
                            "n_estimators": [100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "subsample": [0.8, 1.0],
                            "max_depth": [3, 5, 7]
                            },

                            "Linear Regression": {
                            # Linear Regression has almost no hyperparameters to tune,
                            # but we can check for normalization
                            "fit_intercept": [True, False],
                            "positive": [False, True]
                            },

                            "XGBRegressor": {
                            "n_estimators": [100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "subsample": [0.8, 1.0],
                            "colsample_bytree": [0.8, 1.0]
                            },

                            "CatBoosting Regressor": {
                            "depth": [4, 6, 8],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "iterations": [100, 200]
                            },

                            "AdaBoost Regressor": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1, 1.0],
                            "loss": ["linear", "square", "exponential"]
                            }
                        }


            model_report={}

            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=param_grids)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
    
        except Exception as e:
            raise CustomException(e,sys)