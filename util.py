import pandas as pd
import numpy as np
import pickle
import colorsys
import re

from sklearn import preprocessing
from numpy import array

# fungsi untuk load dataset
def load_dataset(filename):
    # kita load training data untuk membentu model classification
    # header = -1 jika tidak ada header, 0 jika ada header
    dataFrame = pd.read_csv(filename, header = -1, error_bad_lines=False)

    # ubah format dataFrame panda ke bentuk numpy array
    # library sklearn hanya memproses dataset dalam bentuk numpy array
    dataArray = dataFrame.values
    return dataArray

# fungsi untuk save learned model ke sebuah file
def save_model(model, filename):
    # Kita simpan model yang sudah dilatih ke sebuah file, dengan library Pickle
    # library untuk serialisasi di python
    pickle.dump(model, open(filename, 'wb'))

# fungsi untuk load kembali model yang sudah di-save pada suatu file
def load_model(filename):
    # Kita muat kembali model yang sudah kita latih sebelumnya
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# fungsi untuk melakukan normalisasi pada nilai RGB profil Twitter
def normalize_rgb(filename):
    # load data setnya menggunakan method load_dataset
    dataArray = load_dataset(filename)

    # processing normalisasinya
    rgb_normalized = preprocessing.normalize(dataArray[:, 4:10], norm='l2')

    # print(rgb_normalized[0])
    
    # proses replacement data normalized ke data array utama
    i = 0
    for line in rgb_normalized:

        buffer_data = []
        
        for element in line:
            buffer_data.append(element)
        
        color1 = colorsys.rgb_to_hsv(buffer_data[0], buffer_data[1], buffer_data[2])
        color2 = colorsys.rgb_to_hsv(buffer_data[3], buffer_data[4], buffer_data[5])

        j = 4
        for color in color1:
            dataArray[i][j] = color
            j += 1

        for color in color2:
            dataArray[i][j] = color
            j += 1
        
        i += 1

    return dataArray

def extracting_name_feature(dataArray):

    # Loading the dictionary
    # Dictionary is gotten from https://github.com/organisciak/names

    name_dict = load_dataset('../names-master/data/us-likelihood-of-gender-by-name-in-2014.csv')
    man_name_dict = []
    woman_name_dict = []

    for line in name_dict:
        
        if line[0] == 'M' :
            man_name_dict.append(line[1])
        
        else :
            woman_name_dict.append(line[1])


    # Calculating the likelihood of girl and boy of every name

    likelihood = []

    for username in dataArray[:, 1] :

        boyness = 0
        girlness = 0

        for name in man_name_dict :
            # print(username)
            if name in str(username) :
                boyness += 1 

        for name in woman_name_dict :

            if name in str(username) :
                girlness += 1

        # s = u(username)

        # count = len(re.findall(ru'[\U0001f600-\U0001f650]', username))

        # if count > 0 :
        #     girlness += 1

        likelihood.append([boyness, girlness])


    likelihood = preprocessing.normalize(likelihood)

    i = 0


    # Adding features to the dataset

    data = []

    for line in likelihood :

        step_1 = np.insert(dataArray[i], 2, line[0])
        new_instance = np.insert(step_1, 3, line[1])

        data.append(new_instance)

        i += 1
        

    return array(data)

