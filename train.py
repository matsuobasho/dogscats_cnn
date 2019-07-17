import os
import numpy as np
import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from datetime import datetime
from PIL import Image
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


def main():
#def main(num_images):

    cat=os.listdir("train/cats")#[0:num_images]
    dog=os.listdir("train/dogs")#[0:num_images]
    filepath="train/cats/"
    filepath2="train/dogs/"

    print("[INFO] Loading images of cats and dogs each...", datetime.now().time())
    #print("[INFO] Loading {} images of cats and dogs each...".format(num_images), datetime.now().time())
    images=[]
    label = []
    for i in cat:
        image = Image.open(filepath+i)
        image_resized = image.resize((300,300))
        images.append(image_resized)
        label.append(0) #for cat images

    for i in dog:
        image = Image.open(filepath2+i)
        image_resized = image.resize((300,300))
        images.append(image_resized)
        label.append(1) #for dog images

    images_full = np.array([np.array(x) for x in images])

    label = np.array(label)
    label = to_categorical(label)

    images_full, label = shuffle(images_full, label)

    print("[INFO] Splitting into train and test", datetime.now().time())
    (trainX, testX, trainY, testY) = train_test_split(images_full, label, test_size=0.25)


    filters = 10
    filtersize = (5, 5)

    epochs = 5
    batchsize = 32

    input_shape=(300,300,3)
    #input_shape = (30, 30, 3)

    print("[INFO] Designing model architecture...", datetime.now().time())
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.convolutional.Conv2D(filters, filtersize, strides=(2, 2), padding='same',
                                                data_format="channels_last", activation='relu'))
    model.add(keras.layers.convolutional.Conv2D(32, (3,3), strides=(2, 2), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.convolutional.Conv2D(32, filtersize, strides=(2,2), padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))

    model.add(keras.layers.Dense(units=2, activation='softmax'))
    #model.add(keras.layers.Dense(units=2, input_dim=5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("[INFO] Fitting model...", datetime.now().time())
    model.fit(trainX, trainY, epochs=epochs, batch_size=batchsize, validation_split=0.3)

    model.summary()

    print("[INFO] Evaluating on test set...", datetime.now().time())
    eval_res = model.evaluate(testX, testY)
    print(eval_res)

if __name__== "__main__":
    main()