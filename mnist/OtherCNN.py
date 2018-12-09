'''Trains a simple convnet on the MNIST dataset.
Gets to 91.63% test accuracy after 12 epochs

This model includes 3 layers
'''
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
import numpy as np
import pandas as pd

# np.random.seed(133723)  # for reproducibility

trainFileName = "./data/train.csv"
nb_classes = 10
nb_epoch = 20
batch_size = 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# rawData = np.genfromtxt(trainFileName, delimiter=',')
rawData = np.array(pd.read_csv(trainFileName))
print("raw data size", rawData.shape)
''' data '''
trainData = rawData[0:int(rawData.shape[0] * 0.7)]
testData = rawData[int(rawData.shape[0] * 0.7):]


''' label '''
trainLabel = trainData[:, 0]
testLabel = testData[:, 0]
trainData = trainData[:, 1:]
testData = testData[:, 1:]
input_shape = trainData.shape[1]

y_train = np_utils.to_categorical(trainLabel, nb_classes)
y_test = np_utils.to_categorical(testLabel, nb_classes)
print("convert train label to ", y_train.shape[1], "classes")
print("convert test label to ", y_test.shape[1], "classes")

''' keras model'''
print("trainData shape:", trainData.shape, "y_train shape", y_train.shape)
print("testData shape:", testData.shape, "y_test shape", y_test.shape)

model = Sequential()

model.add(Dense(output_dim=128, input_dim=input_shape, init='normal', activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=128, init='normal', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=nb_classes, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(trainData, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(testData,y_test))
score = model.evaluate(testData, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
W, b = model.layers[0].get_weights()
# print('Weights=', W, '\n biases=', b)
y_test_pred = model.predict(testData)
print(np_utils.probas_to_classes(y_test)[:10],np_utils.probas_to_classes(y_test_pred)[:10])

'''test '''
predictFileName = "./data/test.csv"
x_pre = np.array(pd.read_csv(predictFileName))
print("x_pre shape", x_pre.shape)
y_pred = model.predict(x_pre)
pre_class = np_utils.probas_to_classes(y_pred)
print("predict class :", pre_class)
index = np.linspace(1,len(pre_class),len(pre_class))
print("index",index)

np.savetxt("./data/pre.csv", list(zip(index,pre_class)), delimiter=',',fmt='%10.5f')