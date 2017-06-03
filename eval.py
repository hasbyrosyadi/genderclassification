import numpy as np

from numpy import array
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from util import load_dataset, save_model, load_model, normalize_rgb, extracting_name_feature
from sklearn import preprocessing


# Goal: Kita ingin evaluasi seberapa baik performa klasifikasi model kita? 

# load testing data
dataArray = normalize_rgb('extract_twitter_testing')

data_tambahan = load_dataset('test.gender.data.new')

i = 0

new_dataArray = []

for line in data_tambahan:
	new_dataArray.append(np.insert(dataArray[i], 2, line[1]))
	i += 1

data = extracting_name_feature(array(new_dataArray))

X_test = data[:,2:13]
Y_test = data[:,13] #label yang sebenarnya 

# Kita muat kembali model yang sudah kita latih sebelumnya
loaded_model = load_model('Model.model')

# beri label testing data terlebih dahulu
Y_predicted = loaded_model.predict(X_test)

##### metrics evaluasi #####
# hitung score, bandingkan hasil label prediksi, dengan label sesungguhnya di testing data
# --> tampilkan nilai accuracy
accuracy = accuracy_score(Y_test, Y_predicted)
print 'accuracy : ', accuracy

# --> metric lebih detail: precision, recall, f1
report = classification_report(Y_test, Y_predicted)
print '\nprecision, recall, f1:'
print report

# --> confusion matrix
print '\nconfusion matrix:'
print confusion_matrix(Y_test, Y_predicted)

