import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


housing = pd.read_csv("C:/pythonprogram/ML Project-1/data.csv")

# print(housing.head())

# print(housing.describe())

# print(housing['CHAS'].value_counts())

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# print(strat_train_set['CHAS'].value_counts())
# print(strat_test_set['CHAS'].value_counts())
housing = strat_train_set.copy()

# print(housing.shape)

# handling missing values in the dataset
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
# print(imputer.statistics_)

# correlation of MEDV with the other fields
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)
# print(corr_matrix['MEDV'])

# attributes = ["MEDV", "RM", "ZN", "LSTAT"]
# pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()

# creating a pipeline
my_pipeline = Pipeline([ 
    ('imputer', SimpleImputer(strategy = "median")),
    ('std_scaler', StandardScaler())
])

housing_num_tr = my_pipeline.fit_transform(housing)
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)


# print("Predicted Data : ", model.predict(prepared_data))
# print("Actual Data\n", some_labels)

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# 1 2 3 4 5 6 7 8 9 10
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores) # as scores are negative and sqrt takes positive input for real nos.
# print(rmse_scores)

def print_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard Deviation : ", scores.std())

# print_scores(rmse_scores)

X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)  

print(final_rmse)

print(final_predictions)
print(list(Y_test))