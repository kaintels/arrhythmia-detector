import pickle
import pandas as pd
from scipy.io import loadmat, savemat

with open("../dataset/test/inference.pkl", "rb") as f:
    data = pickle.load(f)

data = loadmat("../dataset/test/inference.mat", squeeze_me=True)["ECG"]

mats = {"ECG": data, "label": "ML2"}
savemat("inference.mat", mats)
csvs = pd.DataFrame(data)

csvs.to_csv("inference.csv")