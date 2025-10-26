import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path="artifacts/preprocess.pkl"


class DataTransformation:
    def __init__(self):
        self.config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            # Example: Suppose you have these features
            num_features = ['math_score', 'reading_score', 'writing_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch',
                            'test_preparation_course']
            
            # Create pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first'))
                ]
            )

            logging.info("Numerical columns scaling sucessfully")
            logging.info("categorical columns encoding sucessfully")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )
            
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor=self.get_data_transformer_object()

            #target column
            target_column='math_score'

            #training data
            X_train=train_df.drop(columns=target_column,axis=1)
            y_train=test_df[target_column]

            #testing data
            X_test=test_df.drop(columns=target_column,axis=1)
            y_test=test_df[target_column]

            logging.info("Applying preprocessing object on traning dataframe and testing dataframe")

            #preprocessing
            X_train_pre=preprocessor.fit_transform(X_train)
            X_test_pre=preprocessor.transform(X_test)


            #combining arrays side by side X_train_pre to y_train
            x_train_arr=np.c_(X_train_pre,np.array(y_train))

            #combining arrays side by side X_test_pre to y_test
            x_test_arr=np.c_(X_test_pre,np.array(y_test))

            return x_train_arr,x_test_arr
        
        
        except Exception as e:
            raise CustomException(e,sys)








