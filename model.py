from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

def build_generator(args,qos_shape):

    model = Sequential()

    model.add(Dense(256, input_dim=args.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(np.prod(qos_shape), activation='relu'))
    model.add(Reshape(qos_shape))

    model.summary()

    noise = Input(shape=(args.latent_dim,))
    qos = model(noise)

    return model

def build_discriminator(qos_shape):

    model = Sequential()

    model.add(Flatten(input_shape=qos_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    qos = Input(shape=qos_shape)
    validity = model(qos)

    return Model(qos, validity)

def build_autoencoder(args):
    
    model = Sequential()
    model.add(Dense(1024, input_shape=(args.cols,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(100,    activation='sigmoid', name="bottleneck"))
    
    return model



    