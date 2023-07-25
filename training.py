import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer =  WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '.', '_', ',']

for intent in intents['intents']:
    #print(intent)
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#print(len(train_x))
#print(len(train_y))

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
#model.add(Dropout(Dense(64, activation='relu')))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#sgd = tf.keras.optimizers.SGD(lr=0.01, decay = 1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
               metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
          epochs=700,
          batch_size=32,
          #validation_split = 0.2,
          verbose=1)
model.save('catbot_model.h5', hist)
print('Done!')
