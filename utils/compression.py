import pickle
import wfdb

record_data_dic = wfdb.rdrecord(
    "./mit-bih-arrhythmia-database-1.0.0/119", channels=[0]
)  #

ecg_data = record_data_dic.__dict__

data = ecg_data["p_signal"].squeeze()

with open('119.pkl', 'wb') as f:
    pickle.dump(data, f)
