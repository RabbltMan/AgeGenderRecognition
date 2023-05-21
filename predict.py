import tensorflow as tf
import numpy as np
from keras import losses
from PIL import Image
import matplotlib.pyplot as plt
from net import *
from data import *


class Predict(object):
    def __init__(self):
        self.data = Data()
        AgeCheckpoint = tf.train.latest_checkpoint('./AgeCheckpoints')
        GenderCheckpoint = tf.train.latest_checkpoint('./GenderCheckpoints')
        self.ageNet, self.genderNet = Net(0), Net(1)
        self.ageNet.model.load_weights(AgeCheckpoint)
        self.genderNet.model.load_weights(GenderCheckpoint)

    def runTestCases(self):
        self.ageNet.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.ageNet.model.evaluate(self.data.imageAgeTest, self.data.LabelAgeTest)
        self.genderNet.model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=['accuracy'])
        self.genderNet.model.evaluate(self.data.imageGenderTest, self.data.LabelGenderTest)

    def predict(self, ImagePath: str) -> None:
        # Converting "handtaken" photo into a RGB copy
        img = Image.open(ImagePath).convert('RGB')
        # Reshape to fit model input
        img = img.resize((50, 50))
        inputArray = np.reshape(img, (1, 50, 50, 3)) / 255
        outputArray1 = self.ageNet.model.predict(inputArray, verbose=1)
        outputArray2 = self.genderNet.model.predict(inputArray, verbose=1)
        if (outputArray2[0] > 0.5): gender = 'Female'
        else: gender =  'Male'
        print(f'Age: {int(outputArray1[0][0])}')
        print(f'Gender: {gender}')

        plt.figure('AgeGenderRecognition')
        plt.imshow(img)
        plt.axis('off')
        plt.text(0, 10, f'Age: {int(outputArray1[0][0])}\nSex: {gender}', fontsize=20, color='red')
        plt.show()




if __name__ == '__main__':
    newPredict = Predict()
    newPredict.runTestCases()
    newPredict.predict('me.jpg')