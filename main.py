import tarfile
from pathlib import Path
import urllib.request
from figure_management import FigureManagement
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
figure_2 = plt.figure(2)
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income Category")
plt.ylabel("Number of Districts")

# create training and test sets
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],
                                                   random_state=42)

# dropping income_cat column since it won't be used again
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# exploring training set
housing = strat_train_set.copy()
figure_3 = housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100,
                        label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True,
                        sharex=False, figsize=(10, 7))  # geographical scatter plot
FigureManagement.save_fig("median_house_value_scatter_plot")

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))  # most promising attribute is median_house_value vs. median_income

plt.show()
