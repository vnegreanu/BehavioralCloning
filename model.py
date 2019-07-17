###Imports###

import numpy as np
import cv2
import csv

from sklearn.utils import shuffle        
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten 
from keras.layers import Lambda 
from keras.layers import Cropping2D
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.models import load_model

from pathlib import Path


###Functions###

def check_image(image):
    
    #check for None type or empty image 
    if image is None or image.size==0:
        return False
    return True

def generator(samples, batch_size=32): # Using images from center, left, and right cameras
    
    num_samples = len(samples)
    correction = 0.2
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #get the center camera image
                name = data_path+image_path+batch_sample[0].split('/')[-1]

                #read the center camera image
                center_image = cv2.imread(name)

                #check if image is None type or empty as size
                if check_image(center_image) == False:
                    continue
                else:
                    #correcting image color space representation	
                    center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB) 
                    #get the center steering angle
                    center_angle = float(batch_sample[3])
                    #append image and angle to the global list
                    images.append(center_image)
                    angles.append(center_angle)


                #get the left camera image
                name = data_path+image_path+batch_sample[1].split('/')[-1]
                #read the left camera image
                left_image = cv2.imread(name)
                
                if check_image(left_image) == False:
                    continue
                else:
                    #correcting image color space representation
                    left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                    #get the left steering anngle
                    left_angle = center_angle + correction
                    #append image and angle to the global list
                    images.append(left_image)
                    angles.append(left_angle)


                #get the right camera image
                name = data_path+image_path+batch_sample[2].split('/')[-1]
                #read the right camera image
                right_image = cv2.imread(name)
                
                if check_image(right_image) == False:
                    continue
                else:
                    #correcting image color space representation
                    right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                    #get the right steering anngle
                    right_angle = center_angle - correction
                    #append image and angle to the global list
                    images.append(right_image)
                    angles.append(right_angle)

                
            #set up the training set
            X_train = np.array(images)
            y_train = np.array(angles)
            #yield generator batch
            yield shuffle(X_train, y_train)
            
### Model architecture ####
def get_model():

    input_shape = (160, 320, 3)
 
    model = Sequential()
     
    model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    
    model.add(Conv2D(16, kernel_size=(5,5), activation='relu', padding='valid', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))
    
    model.add(Conv2D(24, kernel_size=(5,5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))

    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(500))
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Dropout(0.50))
    model.add(Dense(1))	
    
    return model

###  Data loading ###
samples = []

#init paths 
data_path = './data/'
image_path = 'IMG/'

#get the steering data, throtlle and brake from csv file
with open(data_path+'driving_log.csv', 'r', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        samples.append(line)
        
#shuffle samples
samples = shuffle(samples)

#split the train and the validation sets - 80/20 percentage
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#set up variables for generators to be used later for the fit_generator function
train_generator = generator(train_samples, batch_size=32)

validation_generator = generator(validation_samples, batch_size=32)

### retreive model ####
my_model = Path('./model.h5')

if my_model.is_file():
    
    model = load_model('model.h5')
    
else:

    model = get_model()

    model.summary()

    model.compile(loss='mse', optimizer='adam')

#### Model training and validation ####

    model.fit_generator(train_generator, steps_per_epoch = 3*len(train_samples)/32, epochs=3, 
                        validation_data=validation_generator, validation_steps = 3*len(validation_samples)/32)

    model.save('model.h5')