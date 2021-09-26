import typing
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class BaselineModel:

    def __init__(self, features: typing.List[str], model):
        self.model = model
        self.features = features

        self.preprocessor = ColumnTransformer(transformers=[
            ('vectorizer', CountVectorizer(ngram_range=(2, 2)), self.features),  # строим биграммы
            ('tf-idf', TfidfTransformer())
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):

        logging.info('Fit model')
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame):
        predictions = self.pipeline.predict(X)

        return predictions
