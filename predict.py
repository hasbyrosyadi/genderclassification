import numpy as np

from sklearn import preprocessing
from util import load_dataset, save_model, load_model, normalize_rgb, extracting_name_feature
from numpy import array


# Goal: Kita ingin menggunakan model yang sudah dibuat untuk prediksi sebuah instance

# Kita muat kembali model yang sudah kita latih sebelumnya
loaded_model = load_model('Model.model')

# kita coba prediksi dua buah instance baru, yang belum diketahui labelnya
dataArray = normalize_rgb('extract_twitter_predict')

data_tambahan = load_dataset('test.gender.nolabel.data')

i = 0

new_dataArray = []

for line in data_tambahan:
	new_dataArray.append(np.insert(dataArray[i], 2, line[1]))
	i += 1

data = extracting_name_feature(array(new_dataArray))

# for line in data:
# 	print line

# hasil prediksi ada di variable predicted_class
predicted_class = loaded_model.predict(data[:,2:])

# cetak label hasil prediksi ke layar
for cetak in predicted_class:
	print cetak