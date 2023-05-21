from PIL import Image, ImageEnhance
import os
import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self):
        self.AUGMENT = 3
        images, ages, genders = self.DatasetInitialize()
        # split sets for age
        imageSamples, self.imageAgeTest, labelSamples, self.LabelAgeTest = train_test_split(images, ages, test_size=0.15)
        self.imageAgeTrain, self.imageAgeVal, self.labelAgeTrain, self.labelAgeVal = train_test_split(imageSamples, labelSamples, test_size=0.1)
        # split sets for gender
        imageSamples, self.imageGenderTest, labelSamples, self.LabelGenderTest = train_test_split(images, genders, test_size=0.15)
        self.imageGenderTrain, self.imageGenderVal, self.labelGenderTrain, self.labelGenderVal = train_test_split(imageSamples, labelSamples, test_size=0.1)

    def DatasetInitialize(self):
        DatasetImages = os.listdir('./UTKFace')
        if (not os.path.exists('./Aug/')):
            os.mkdir('./Aug/')
            images, ages, genders = list(), list(), list()
            iterCnt = 0
            for image in DatasetImages:
                labels = [int(label) for label in image.split("_", 3)[:2]]
                # data augment
                for _ in range(self.AUGMENT):
                    ages.append(labels[0])
                    genders.append(labels[1])
                # original image
                img0 = Image.open(f'./UTKFace/{image}').convert("RGB")
                # transferred images
                img1 = self.randomChange(img0, f'{iterCnt+1}-1')
                img2 = self.randomChange(img0, f'{iterCnt+1}-2')
                # append images
                images.append(np.array(img0.resize((50, 50)))/255)
                images.append(np.array(img1.resize((50, 50)))/255)
                images.append(np.array(img2.resize((50, 50)))/255)
                iterCnt += 1
                print(f'{iterCnt}/{len(DatasetImages)} Done.', end='\r')
            # convert lists into Numpy arrays
            images = np.array(images)
            np.save('images', images)
            ages = np.array(ages, dtype=np.uint8)
            np.save('ages', ages)
            genders = np.array(genders, dtype=np.int8)
            np.save('genders', genders)
            return images, ages, genders
        
        elif (os.path.exists('./Aug/') and os.path.exists('./genders.npy')):
            images = np.load('images.npy')
            ages = np.load('ages.npy')
            genders = np.load('genders.npy')
            return images, ages, genders
        
        else:
            files = os.listdir('./Aug/')
            for file in files:
                os.remove(f'./Aug/{file}')
            self.DatasetInitialize()
            

    def randomChange(self, img, name):
        # random brightness
        brightness = np.random.ranf() * 0.5 + 0.75
        brightEnhancer = ImageEnhance.Brightness(img)
        img = brightEnhancer.enhance(brightness)
        # random contrast
        contrast = np.random.ranf() * 0.5 + 0.75
        contrastEnhancer = ImageEnhance.Contrast(img)
        img = contrastEnhancer.enhance(contrast)
        # random color balance
        balance = np.random.ranf() * 0.5 + 0.75
        colorEnhancer = ImageEnhance.Color(img)
        img = colorEnhancer.enhance(balance)
        # random flipping and rotation
        if (np.random.ranf() > 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if (np.random.ranf() > 0.5):
            img = img.rotate(np.random.randint(-10, 11))
        # save changed image
        img.save(f'./Aug/{name}.jpg')

        return img
    
if __name__ == '__main__':
    data = Data()
