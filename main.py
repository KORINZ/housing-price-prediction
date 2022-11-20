import tarfile
from pathlib import Path
import urllib.request
from figure_management import FigureManagement
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ClusterSimilarity import ClusterSimilarity
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()

figure_1 = housing.hist(bins=50, figsize=(12, 8))
FigureManagement.save_fig("attribute_histogram_plots")

# generate descriptive statistics of the data
# print(housing.describe().to_string())

# stratify sampling sets
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# create stratified training and test sets
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],
                                                   random_state=42)

# dropping income_cat column since it won't be used again
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# exploring training set
housing = strat_train_set.copy()
figure_2 = housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100,
                        label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True,
                        sharex=False, figsize=(10, 7))  # geographical scatter plot
FigureManagement.save_fig("median_house_value_scatter_plot")

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
figure_3 = scatter_matrix(housing[attributes], figsize=(12, 8))  # most promising attribute is median_house_value vs. median_income

# revert to a clean training set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
imputer = SimpleImputer(strategy="median")  # perform imputation based on median value of each attribute
housing_num = housing.select_dtypes(include=[np.number])  # exclude non-numerical attributes
imputer.fit(housing_num)
# print(housing_num.median().values)  # see the imputed median value of each attribute

# transform the training set with median values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Revert the categorical attribute to numerical
housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Scaling data
# Gaussian RBF similarity to the nearest cluster center
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)
housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

figure_4 = housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")
FigureManagement().save_fig("district_cluster_plot")

# training the model
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_prepared = preprocessing.fit_transform(housing)

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                              squared=False)
print(lin_rmse)

plt.show()
