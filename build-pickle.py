# -*- coding: utf-8 -*-
from unidecode import unidecode
import os
import csv
import gzip
import xml.etree.ElementTree as ET
import random

import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer

def remove_non_ascii(text):
    return unidecode(text)

file_reader = open('dataset.csv', 'rb')
genre_reader = csv.reader(file_reader, delimiter=',', quotechar='|')
genre_list = { "Action": 0, "Adult": 1, "Adventure": 2, "Animation": 3, "Biography": 4, 
                "Comedy": 5, "Crime": 6, "Documentary": 7, "Drama": 8, "Family": 9, 
                "Fantasy": 10, "Film-Noir": 11, "Game-Show": 12, "History": 13, 
                "Horror": 14, "Music": 15, "Musical": 16, "Mystery": 17, "News": 18, 
                "Reality-TV": 19, "Romance": 20, "Sci-Fi": 21, "Short": 22, "Sport": 23, 
                "Talk-Show": 24, "Thriller": 25, "War": 26, "Western": 27 }

genre_counts = [0] * 28

film_list = {}
subtitle_list = {}

imdb_checked_list = []
osub_checked_list = []
genre_set = set()

for g in genre_reader:
    if g[1] not in imdb_checked_list:
        l = []
        for i in range(2, len(g)):
            if (g[i] not in genre_list):
                genre_set.add(g[i])
            if (g[i] in genre_list) and (g[0]!='') and g[1]!='Duplicate':
                genre_counts[genre_list[g[i]]] += 1
                l.append(genre_list[g[i]])
        film_list[g[0]] = l
        osub_checked_list.append(g[0])
        imdb_checked_list.append(g[1])

print(genre_counts)

file_reader.close()
print('%s Film loaded' % len(film_list))

load_max = 1000000

subtitle_count = 0
for root, dirs, files in os.walk("raw"):
    if subtitle_count >= load_max:
        break

    for file in files:
        if file.find('.xml.gz') > 0:
            with gzip.open(root + '/' + file, 'rb') as f:
                os_id = file.replace('.xml.gz', '')
                if os_id in osub_checked_list:
                    try:
                        file_content = f.read()
                        file_text = ""
                        xml_root = ET.fromstring(file_content)
                        for element in xml_root.findall("s"):
                            file_text += remove_non_ascii("".join(element.itertext()).replace('\n', ' ').strip() + '. ')

                        subtitle_list[os_id] = file_text
                        subtitle_count += 1
                        if subtitle_count>=load_max:
                            break
                        if subtitle_count % 1000 == 0:
                            print("%s Loaded" % subtitle_count)
                    except:
                        print('hata', os_id)
                        pass
                else:
                    pass

print('%s Subtitle loaded' % len(subtitle_list))
print('Learning will start with %s films' % len(subtitle_list))

max_words = 100000 # 569136

test_split = 0.2
train_len = (1-test_split) * len(subtitle_list)

x_train = []
y_train = []
x_test = []
y_test = []

x_films = [[],[],[],[],[]]
y_films = [[],[],[],[],[]]
key_films = [[],[],[],[],[]]

print('Splitting data...')
key_list = list(subtitle_list.keys())
random.shuffle(key_list)

num_classes = 28 # np.max(y_train) + 1
print(num_classes, 'classes')

def tokenize_reshape(data):
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words, filters='\'!\?"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([ v for v in subtitle_list.values() ])
print(len(tokenizer.word_index), "Different Words")

words_file = open("words.txt","w")
words_file.write(str(tokenizer.word_index))

li = 0
print('%s film will split into 5 list', len(key_list))

i = 0
for film_id in key_list:
    if i < (li+1)*len(key_list)/5:
        pass
    else:
        li += 1

    x_films[li].append(subtitle_list[film_id])
    y_films[li].append(film_list[film_id])
    key_films[li].append(film_id)
    i+=1

x_dataset = [[],[],[],[],[]]
x_results = [[],[],[],[],[]]

for i in range(0, 5):
    x_dataset[i] = tokenizer.texts_to_matrix(x_films[i], mode='count')
    x_dataset[i] = tokenize_reshape(x_dataset[i])
    x_results[i] = y_films[i]


with open('dataset.pickle', 'wb') as output:
    pickle.dump(x_dataset, output)

with open('results.pickle', 'wb') as output:
    pickle.dump(y_films, output)

with open('keys.pickle', 'wb') as output:
    pickle.dump(key_films, output)
