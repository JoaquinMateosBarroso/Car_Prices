# %% [markdown]
# # Prediction of car prices based on their condition
# 
# 
# ![image](https://storage.googleapis.com/kaggle-datasets-images/1479517/2444963/1aaa3760e7dd34a87af175482c1514ae/dataset-cover.jpg?t=2021-07-21-09-56-46)
# 
# Data extracted from kaggle. Click [here](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset) to see it.

# %% [markdown]
# ## First view at the data <a id="section1"></a>

# %%
import os
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("Running on Kaggle")
    data_dir = "../input/car-prices-dataset"
else:
    print("Not running on Kaggle")
    data_dir = "archive"

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# We already have the data partitioned
df = pd.read_csv(f"{data_dir}/train.csv")
df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)


print(f"Shape: {df_train.shape}")
df_train.head()

# %%
df_train.isna().sum() # Below we can see the data is extremely clean; this is not common at all

# %%
sorted(df_train["Prod. year"].unique())

# %%
import re

def clean(df):
    df = df.copy()
    df["Leather interior"] = df["Leather interior"].apply(lambda x: 1 if x=="Yes" else 0) # Numeric atributes are much easier to manage
    df.drop(["ID", "Levy"], axis=1, inplace=True)
    df["Engine volume"] = df["Engine volume"].apply(lambda x: float(x.split(" ")[0]))
    df["Mileage"] = df["Mileage"].apply(lambda x: float(x.split(" ")[0]))
    df["Doors"] = df["Doors"].apply(lambda x: np.int64(re.findall(r'\d+', x)[0]))
    return df

print("We have a {:.2%} of null levies".format(df_train['Levy'].apply(lambda x: x=='-').sum() / df_train.shape[0]))
print("It is very high, so we won't be using that feature")

df_train = clean(df_train)
df_train.info()

# %%
df_train.describe()

# %%
import matplotlib.pyplot as plt

Y = df_train.select_dtypes(include = ["float64", "int64"])

# %%
fig, axs = plt.subplots(3, 3)

fig.suptitle("Data before dropping outliers")
fig.set_dpi(200)
fig.set_size_inches(20, 10)
for index, column in enumerate(Y.columns):
    axs[int(index/3)][index%3].boxplot(Y[column])
    axs[int(index/3)][index%3].set_title(column)

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# We can appreciate various outliers in *Price*, *Engine Volume* and a few more. Let's drop them

# %%
for column in ["Price", "Prod. year", "Engine volume", "Mileage"]:
    Q1 = df_train[column].quantile(0.25)
    Q3 = df_train[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df_train[column]>=upper)[0]
    lower_array = np.where(df_train[column]<=lower)[0]
    
    # Removing the outliers
    df_train.drop(index=upper_array, inplace=True)
    df_train.drop(index=lower_array, inplace=True)
    df_train.reset_index(inplace=True, drop=True)

Y = df_train.select_dtypes(include = ["float64", "int64"])

# %%
fig, axs = plt.subplots(3, 3)

fig.suptitle("Data after dropping outliers")
fig.set_dpi(200)
fig.set_size_inches(20, 10)

for index, column in enumerate(Y.columns):
    axs[int(index/3)][index%3].boxplot(Y[column])
    axs[int(index/3)][index%3].set_title(column)

# %% [markdown]
# This already looks like a much cleaner dataset. We could drop a lot of data that seems to be outlier, but we'll use all the data for the moment and worry later

# %%
fig, ax = plt.subplots(figsize = (Y.shape[1], Y.shape[1]))
fig = plt.imshow(Y.corr(), cmap = "YlOrRd")
ax.set_xticks(range(Y.shape[1]))
ax.set_xticklabels(Y.axes[1])
ax.set_yticks(range(Y.shape[1]))
ax.set_yticklabels(Y.axes[1])
for (j,i),label in np.ndenumerate((Y.corr()*100).round(2)):
    ax.text(i,j,label,ha='center',va='center')
plt.show()

# %%
fig, axs = plt.subplots(3, 3)

fig.suptitle("Cars price in function of features")
fig.set_dpi(200)
fig.set_size_inches(20, 10)

for index, column in enumerate(Y.columns):
    axs[int(index/3)][index%3].scatter(Y[column], Y["Price"])
    axs[int(index/3)][index%3].set_title(column)

# %% [markdown]
# ## Get extra features

# %% [markdown]
# This project is aimed at predicting car prices based on a few features. We can get extra features that could help us in the prediction. We'll use the jerarquical KMeans algorithm to get the clusters.

# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

class GetExtraFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dataset = None
        self.knownFeatures = []
        self.unknownFeatures = []
        
    def fit(self, X, y=None):
        self.dataset = X
        self.__assignFeatures__(X.columns)
        return self
    
    def __assignFeatures__(self, columns):
        self.knownFeatures = ["Manufacturer", "Model", "Color", "Prod. year", "Mileage", "Leather interior"]
        self.unknownFeatures = list((set(columns) - set(self.knownFeatures)) - {"Price"})
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def transformRow(row):
            carModelInstance = self.dataset[
                (self.dataset["Manufacturer"]==row["Manufacturer"]) & 
                (self.dataset["Model"]==row["Model"])].iloc[0, :]
            newRow = pd.Series(index=self.knownFeatures+self.unknownFeatures)
            newRow[self.knownFeatures] = row[self.knownFeatures]
            newRow[self.unknownFeatures] = carModelInstance[self.unknownFeatures]
            
            return newRow
        
        X = X.apply(transformRow, axis=1)

        return X

# %%
getter = GetExtraFeatures()
getter.fit(df_train)



knownFeatures = ["Manufacturer", "Model", "Color", "Prod. year", "Mileage", "Leather interior"]
X = df_train[knownFeatures]

# getter.transform(X)

# %% [markdown]
# ## Select and Train a Model

# %% [markdown]
# We will use a Random Forest, that should work relatively well

# %%
df_train.dtypes[(df_train.dtypes!="float64") & (df_train.dtypes!="int64")].index

# %%
from catboost import CatBoostRegressor

categoricalFeatures = list(df_train.dtypes[(df_train.dtypes!="float64") & (df_train.dtypes!="int64")].index)
tree = CatBoostRegressor(num_trees=100, max_depth=16, random_seed=42)
tree.fit(df_train.drop(columns=["Price"]), df_train["Price"], categoricalFeatures)

# %%
df_test = clean(df_test)
predictedPrice = tree.predict(df_test.drop(columns=["Price"]))

# %%
from sklearn.metrics import mean_squared_error

mean_squared_error(df_test["Price"], predictedPrice)**(0.5)

# %%
df_test

# %%
from sklearn.metrics import classification_report

classification_report(df_test["Price"], predictedPrice)

# %% [markdown]
# ## Save models

# %%
tree.save_model("tree.model")

import pickle
pickle.dump(getter, open("extraFeaturesGetter.pkl", 'wb'))

# %% [markdown]
# # License
# 
# This Jupyter Notebook and its contents are licensed under the terms of the GNU General Public License Version 2 as published by the Free Software Foundation. The full text of the license can be found at: https://www.gnu.org/licenses/gpl-2.0.html
# 
# Copyright (c) 2024, Joaquín Mateos Barroso
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/ for a list of additional licenses.


