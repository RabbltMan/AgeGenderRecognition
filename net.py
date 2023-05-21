from keras.layers import *
from keras.models import Sequential

class Net(object):
    def __init__(self, tag=0):
        model = Sequential()
        model.add(Input(shape=(50, 50, 3)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(96, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(120, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(240, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())

        if (tag == 0):
            model.add(Dropout(0.3))
            model.add(Dense(192, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(116))
            self.model = model
        else:
            model.add(Dense(60, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(1))
            self.model = model

if __name__ == '__main__':
    model = Net()
    model.model.summary()