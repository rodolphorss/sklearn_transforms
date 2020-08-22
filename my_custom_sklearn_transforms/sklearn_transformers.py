from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import SimpleImputer

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data_2 =  data.drop(labels=self.columns, axis='columns')

        si = SimpleImputer(
            missing_values=np.nan,
            strategy='median'
        )
        si.fit(data_2.select_dtypes(exclude=['object']))

        # Reconstrução de um novo DataFrame Pandas com o conjunto imputado
        data3 = data_2.select_dtypes(exclude=['object'])
        data3 = si.transform(data_2.select_dtypes(exclude=['object']))
        data4 = pd.DataFrame.from_records(data3, columns=data_2.columns[data_2.columns != 'PERFIL'])
        data4['PERFIL'] = X['PERFIL']
        return data4

