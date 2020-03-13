import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Reshape, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from logging import log, info, debug 
import os
from utils import ModelsSufix

class CNNSoundTransferModel:
    def __init__(self, width, height, channels, ouputmodelpath):
        self.disc_output_content_model_path = ouputmodelpath + ModelsSufix.DICSR_CONT
        self.disc_output_style_model_path = ouputmodelpath + ModelsSufix.DICSR_STYLE
        self.gen_output_model_path = ouputmodelpath + ModelsSufix.GEN
        self.width = width
        self.height = height
        self.channels = channels

        # Generator model initialization
        self.content_optimizer = Adam(lr=0.002, beta_1=0.5, decay=8e-8)
        self.style_optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.g_model = self.__generator(self.gen_output_model_path)
        self.g_model.compile(loss='binary_crossentropy', optimizer=self.content_optimizer)

        # Style content model initialization
        self.d_content_optimizer = Adam(lr=0.02)
        self.d_content_model = self.__discriminator(self.disc_output_content_model_path)
        self.d_content_model.compile(loss='binary_crossentropy', optimizer=self.d_content_optimizer,metrics=['accuracy'])
        
        # Style dyscriminator model initialization
        self.d_style_optimizer = Adam(lr=0.02)
        self.d_style_model = self.__discriminator(self.disc_output_style_model_path)
        self.d_style_model.compile(loss='binary_crossentropy', optimizer=self.d_style_optimizer,metrics=['accuracy'])

        # Stack content model
        self.stack_content_model = Sequential()
        self.stack_content_model.add(self.g_model)
        self.stack_content_model.add(self.d_content_model)
        self.stack_content_model.compile(loss='binary_crossentropy', optimizer=self.content_optimizer)

        # Stack style model
        self.stack_style_model = Sequential()
        self.stack_style_model.add(self.g_model)
        self.stack_style_model.add(self.d_style_model)
        self.stack_style_model.compile(loss='binary_crossentropy', optimizer=self.style_optimizer)

    def __discriminator(self, path):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(self.width, self.height, self.channels)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        if os.path.exists(path):
            info("Loading discriminator weights.")
            model.load_weights(path)
        return model

    def __generator(self, path):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.width, self.height, self.channels)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))
        if os.path.exists(path):
            info("Loading generator weights.")
            model.load_weights(path)
        return model

    def train(self, X_content_train, X_style_train, epochs=20000, batch = 1, save_interval = 100, save_callback= None):
        for cnt in range(epochs):
            random_index = 0

            ## Prepare training datasets
            if len(X_content_train) != 1:
                random_index = np.random.randint(0, len(X_content_train) - np.int64(batch))
          
            legit_content_data = X_content_train[random_index : random_index + np.int64(batch)].reshape(np.int64(batch), self.width, self.height, self.channels)
            legit_style_data = X_style_train[random_index : random_index + np.int64(batch)].reshape(np.int64(batch), self.width, self.height, self.channels)
       
            
            # Generating fake data
            # Learn model to generate from content data
            gen_noise = legit_content_data
            fake_data = self.g_model.predict(gen_noise)

            # Creating combined dataset for content
            x_combined_content_batch = np.concatenate((legit_content_data, fake_data))
             
            # Creating combined dataset for style
            x_combined_style_batch = np.concatenate((legit_style_data, fake_data))

            # Create combined y values
            y_combined_batch = np.concatenate((np.ones((np.int64(batch), 1)), np.zeros((np.int64(batch), 1))))
           
            d_loss_content = self.d_content_model.train_on_batch(x_combined_content_batch, y_combined_batch)
            d_loss_style = self.d_style_model.train_on_batch(x_combined_style_batch, y_combined_batch)

            # train generator
            y_mislabled = np.ones((batch, 1))
            g_content_loss = self.stack_content_model.train_on_batch(gen_noise, y_mislabled)
            g_style_loss = self.stack_style_model.train_on_batch(gen_noise, y_mislabled)
            g_total_loss = g_content_loss + g_style_loss
            info('epoch: %d, [Discriminator :: content d_loss: %f style d_loss %f], [Generator :: total loss: %f content loss: %f style loss: %f]' % (cnt, d_loss_content[0], d_loss_style[0], g_total_loss, g_content_loss, g_style_loss))

            if (cnt+1) % save_interval == 0:
                info('Saving model for epoch: %d, [Discriminator :: content d_loss: %f style d_loss %f], [Generator :: total loss: %f content loss: %f style loss: %f]' % (cnt, d_loss_content[0], d_loss_style[0], g_total_loss, g_content_loss, g_style_loss))
                self.__run_save_model(save_callback)
                
        info('Training completed, exporting models')
        self.__run_save_model(save_callback)

    def __run_save_model(self, callback):
        self.d_content_model(self.disc_output_content_model_path)
        self.d_style_model(self.disc_output_style_model_path)
        self.g_model.save(self.gen_output_model_path)
        info('Models saved locally....')
        if callback != None:
            info('Start uploading models to cloud...')
            callback(self.gen_output_model_path, self.disc_output_model_path)
        