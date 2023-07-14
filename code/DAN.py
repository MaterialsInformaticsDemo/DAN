# Learning Transferable Features with Deep Adaptation Networks
# based on tensorflow 2, keras
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication

import os
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from MK_MMD import MK_MMD
# threshold parameter specifies the element threshold for array display
np.set_printoptions(threshold=np.inf)

"""
NOTE:
In order to facilitate data loading and streamline the experimentation process, 
I have chosen to utilize the test data from the cafar dataset as the target domain training data for my implement. 
This decision stems from our primary focus on investigating the network's structure and implementation. 
Additionally, the availability of suitable and sufficiently large datasets within the material field posed significant limitations
"""
cifar10 = tf.keras.datasets.cifar10
(source_x_train, source_y_train), (target_x_train, _) = cifar10.load_data()
source_x_train, target_x_train = source_x_train[:10000,:,:,:] / 255.0, target_x_train / 255.0
source_y_train = source_y_train[:10000]

class DAN(Model):
    def __init__(self):
        super(DAN, self).__init__()
        # def the structure of ALEXnet
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')
                         
        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')
                         
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()

        self.f1_source = Dense(2048, activation='relu')
        self.d1_source = Dropout(0.5)
        self.f2_source = Dense(2048, activation='relu')
        self.d2_source = Dropout(0.5)
        self.f3_source = Dense(10, activation='softmax')

        self.f1_target = Dense(2048, activation='relu')
        self.d1_target = Dropout(0.5)
        self.f2_target = Dense(2048, activation='relu')
        self.d2_target = Dropout(0.5)
        self.f3_target = Dense(10, activation='softmax')

    def call(self, inputs):
        source_x, target_x  = inputs

        # def the forward propagation of source domain training data
        source_x = self.c1(source_x, training=False)
        source_x = self.b1(source_x)
        source_x = self.a1(source_x)
        source_x = self.p1(source_x)

        source_x = self.c2(source_x, training=False)
        source_x = self.b2(source_x)
        source_x = self.a2(source_x)
        source_x = self.p2(source_x)

        source_x = self.c3(source_x, training=False)

        # fine-tune
        source_x = self.c4(source_x)
        source_x = self.c5(source_x)
        source_x = self.p3(source_x)

        source_x = self.flatten(source_x)
        source_y1 = self.f1_source(source_x)
        source_x = self.d1_source(source_x)
        source_y2 = self.f2_source(source_x)
        source_x = self.d2_source(source_x)
        y_source = self.f3_source(source_x)

        # def the forward propagation of traget domain training data
        target_x = self.c1(target_x, training=False)
        target_x = self.b1(target_x)
        target_x = self.a1(target_x)
        target_x = self.p1(target_x)

        target_x = self.c2(target_x, training=False)
        target_x = self.b2(target_x)
        target_x = self.a2(target_x)
        target_x = self.p2(target_x)

        target_x = self.c3(target_x, training=False)

        # fine-tune
        target_x = self.c4(target_x)
        target_x = self.c5(target_x)
        target_x = self.p3(target_x)

        target_x = self.flatten(target_x)
        target_y1 = self.f1_target(target_x)
        target_x = self.d1_target(target_x)
        target_y2 = self.f2_target(target_x)
        target_x = self.d2_target(target_x)
        y_target = self.f3_target(target_x)

        # define the transfer regularization loss function in DAN 
        MMDloss1 = MK_MMD(source_y1, target_y1)
        MMDloss2 = MK_MMD(source_y2, target_y2)
        MMDloss3 = MK_MMD(y_source, y_target)

        self.add_loss(1/3 * MMDloss1 + 1/6 * MMDloss2 + 1/6* MMDloss3)
    
        return y_source


model = DAN()

# comolie the model 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


# save the model, and retrain the model
checkpoint_save_path = "./checkpoint/DAN.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


source_x = source_x_train
target_x = target_x_train 

# recore the training precess
csv_logger = CSVLogger('training_log.csv')
history = model.fit([source_x, target_x,], source_y_train, batch_size=32, epochs=1, validation_data=([source_x, target_x,], source_y_train), callbacks=[cp_callback])

# save the training process 
with open('history.txt', 'w') as file:
    for key, value in history.history.items():
        file.write(f'{key}: {value}\n')
