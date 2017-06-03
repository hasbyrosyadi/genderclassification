import numpy as np

from util import load_dataset, save_model, load_model, normalize_rgb, extracting_name_feature
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from numpy import array


# Goal: diberikan labeled training data, kita ingin bangun model klasifikasi

# kita load training data untuk membentu model classification
# dataArray dalam bentuk numpy array 2D (num_sample x num_feature)

dataArray = normalize_rgb('extract_twitter_training')

data_tambahan = load_dataset('train.gender.data.new')

i = 0

new_dataArray = []

for line in data_tambahan:
	new_dataArray.append(np.insert(dataArray[i], 2, line[1]))
	i += 1

# for line in new_dataArray:
# 	print line

data = extracting_name_feature(array(new_dataArray))

# kita pisah antara Fitur dan Class
# X merupakan kumpulan fitur untuk setiap instance, kolom indeks 0 - 3
# Y merupakan label dari setiap instance, kolom indeks 4
# Y adalah vector of integers or strings.
X_train = data[:,2:13]
Y_train = data[:,13]

# Kita gunakan Algoritma Machine Learning Logistic Regression untuk membangun model

model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)

# model = xgboost.XGBClassifier()
# model = KMeans(n_clusters=3, random_state=0)
# model = xgboost.XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=8, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.6, objective= 'binary:logistic', nthread=2, scale_pos_weight=1, seed=69)

# Training ! proses ini akan membutuhkan waktu, untuk estimasi parameter

model.fit(X_train, Y_train)


# Kita simpan model yang sudah dilatih ke sebuah file
save_model(model, 'Model.model')



