import numpy as np
import re
from joblib import load


userInputFeatures = [
    'Manufacturer', 'Model', 'Color', 'Prod. year', 
    'Leather interior', 'Mileage']


def getExtraFeatures(df, generalDF):
    df = df.copy()
    extraFeatures = list(set(generalDF.keys()).difference(set(userInputFeatures)))
    extraFeatures.remove('ID'); extraFeatures.remove('Price'); extraFeatures.remove('Levy')
    # Hay que arreglar esto; data no es un dataframe, es  un diccionario. Foook
    auxModel = generalDF.loc[generalDF['Model'] == df['Model'][0]].iloc[0]
    for feature in extraFeatures:
        newFeature = auxModel[feature]
        df.loc[0, feature] = newFeature
    df = clean(df)
    print(df.head())
    return df

def clean(df):
    df = df.copy()
    df["Engine volume"] = df["Engine volume"].apply(lambda x: float(x.split(" ")[0]))
    df["Doors"] = df["Doors"].apply(lambda x: np.int64(re.findall(r'\d+', x)[0]))
    df.drop(["ID", "Price", "Levy"], axis=1, inplace=True)
    return df

class DataManipulation:
    def __init__(self, home):
        self.home = home
        self.prepare_models()
        
    def prepare_models(self):
        # load models from the file
        self.preparation_pipeline = load(f'{self.home}models/preparation_pipeline.joblib')
        self.pca = load(f'{self.home}models/pca.joblib')
        self.final_model = load(f'{self.home}models/final_model.joblib')

    def predict(self, df):
        X = self.preparation_pipeline.transform(df)
        X = self.pca.transform(X.toarray())
        Y = self.final_model.predict(X)
        return Y[0]