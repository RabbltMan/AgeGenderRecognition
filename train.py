from net import *
from data import *
import tensorflow as tf
from keras import optimizers, losses
import pickle

class Train(object):
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005

    def __init__(self):
        self.ageNet, self.genderNet = Net(0), Net(1)
        self.data = Data()

    def train(self):
        if (not os.path.exists("./AgeCheckpoints/")):
            os.mkdir('./AgeCheckpoints/')
        if (not os.path.exists("./GenderCheckpoints/")):
            os.mkdir('./GenderCheckpoints/')
        GenderCheckpoint = 'GenderCheckpoints\\model.ckpt'
        AgeCheckpoint = 'AgeCheckpoints\\model.ckpt'
        saveAgeModelCallback = tf.keras.callbacks.ModelCheckpoint(AgeCheckpoint,
                                                                  save_weights_only=True,
                                                                  save_best_only=True,
                                                                  verbose=0,
                                                                  save_freq='epoch')
        saveGenderModelCallback = tf.keras.callbacks.ModelCheckpoint(GenderCheckpoint,
                                                                     save_weights_only=True,
                                                                     save_best_only=True,
                                                                     verbose=0,
                                                                     save_freq='epoch')


        opt = optimizers.Adam(learning_rate=self.LEARNING_RATE)
        self.ageNet.model.compile(optimizer=opt,
                                  loss='mse',
                                  metrics=['mae'])
        ageHistory = self.ageNet.model.fit(self.data.imageAgeTrain, self.data.labelAgeTrain,
                              batch_size=self.BATCH_SIZE,
                              epochs=100,
                              callbacks=[saveAgeModelCallback],
                              validation_data=(self.data.imageAgeVal, self.data.labelAgeVal))

        opt2 = optimizers.Adam(learning_rate=self.LEARNING_RATE)
        self.genderNet.model.compile(optimizer=opt2,
                                     loss=losses.binary_crossentropy,
                                     metrics=['accuracy'])

        genderHistory = self.genderNet.model.fit(self.data.imageGenderTrain, self.data.labelGenderTrain,
                                 batch_size=self.BATCH_SIZE,
                                 epochs=50,
                                 callbacks=[saveGenderModelCallback],
                                 validation_data=(self.data.imageGenderVal, self.data.labelGenderVal))

        with open("AgeTrainHistoryDict.txt", 'wb') as ageTrain:
            pickle.dump(ageHistory.history, ageTrain)
        with open("GenderTrainHistoryDict.txt", 'wb') as genderTrain:
            pickle.dump(genderHistory.history, genderTrain)

if __name__ == '__main__':
    newTrain = Train()
    newTrain.train()