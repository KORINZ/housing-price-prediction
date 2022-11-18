import numpy as np
import pandas as pd
import tarfile
from pathlib import Path
import urllib.request
from figure_management import FigureManagement
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)


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

housing.hist(bins=50, figsize=(12, 8))
FigureManagement.save_fig("attribute_histogram_plots")

# generate descriptive statistics of the data
# print(housing.describe().to_string())

# create training and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

plt.show()
