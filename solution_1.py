import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def prepare_data(df):

    categorical_cols = [cols for cols in df.columns if df[cols].dtype == 'object']
    numerical_cols = [cols for cols in df.columns if df[cols].dtype in ['int', 'float']]
        
    impute_transformer = SimpleImputer(strategy = 'constant')

    #Preprocessing categorical data.
    categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('ordinal_enc', OrdinalEncoder(handle_unknown= 'error')),
            #('scale', MinMaxScaler(feature_range = (0, 1)))
        ]
    )

    #Preprocessor for numerical and categorical data.
    preprocessor = ColumnTransformer(
        transformers = [
            ('imp', impute_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return pd.DataFrame(preprocessor.fit_transform(df), columns = df.columns)