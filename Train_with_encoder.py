import cfg
import os
import pandas as pd
import numpy as np
from model import build_generator, build_discriminator, build_autoencoder
from tester import test_autoencoder, mean_every, mean_all
from tensorflow.keras.optimizers import Adam 
from tqdm import tqdm
from keras.layers import Input
from keras.models import Model
import tensorflow as tf




# load model
print('==> loading models')

def main():
    
    resultat = pd.DataFrame(columns = ['EPOCHS','RMSE_AUTOENCODERS'])
    
    args = cfg.parse_args()
    
    log_path = os.path.join(args.save_path+'/log')
    model_path = os.path.join(args.save_path+'/models')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    #SHAPE 
    qos_shape = (1, args.cols)
    # optimizer
    optimizer = Adam(0.001, 0.5)
    # Build and compile the discriminator
    discriminator = build_discriminator(qos_shape)
    discriminator.compile(loss=args.loss, optimizer=optimizer,
                               metrics=['accuracy'])
    # Build the encoder
    encoder = build_autoencoder(args)
    encoder.compile(loss=args.loss, optimizer=optimizer,
                               metrics=['accuracy'])
    # Build the generator
    generator = build_generator(args, qos_shape)
    
    # The generator takes noise as input and generates vectors
    z = Input(shape=(args.latent_dim,))
    qos = generator(z)
    
    # For the combined model we will only train the generator
    discriminator.trainable = False
    
    # The discriminator takes generated vectors as input and determines validity
    validity = discriminator(qos)
    
    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, validity)
    combined.compile(loss=args.loss, optimizer=optimizer)
    
    # Load the dataset
    rtMatrix = pd.read_csv('./dataset/train.txt', delimiter="\t", header=None)
        
	# Rescale -1 to 1 
    X_train = np.expand_dims(rtMatrix, axis=1)
    X_train = X_train / 20.
        

	# Adversarial ground truths
    valid = np.ones((args.batch_size, 1))
    fake = np.zeros((args.batch_size, 1))

    for epoch in tqdm(range(args.max_epoch)):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of vectors
        idx = np.random.randint(0, X_train.shape[0], args.batch_size)
        v_qos = X_train[idx]

        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        
        lay1= generator.layers[0].get_weights()[0].T
        lay3= generator.layers[3].get_weights()[0].T
        lay6= generator.layers[6].get_weights()[0].T
        lay9= generator.layers[9].get_weights()[0].T

        K = tf.keras.backend
        K.set_value(encoder.layers[1].weights[0], lay9)
        K.set_value(encoder.layers[4].weights[0], lay6)
        K.set_value(encoder.layers[7].weights[0], lay3)
        K.set_value(encoder.layers[10].weights[0], lay1)
        
        encoder = encoder.predict(v_qos)

        # Generate a batch of new vectors
        gen_v_qos = generator.predict(encoder)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(v_qos, valid)
        d_loss_fake = discriminator.train_on_batch(gen_v_qos, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)
            
        if epoch % args.sample_interval == 0:
            # three different test methods
            if args.test_method == 1:
                function_test = mean_all(args, epoch, generator,d_loss[0], 100*d_loss[1], g_loss)
            elif args.test_method == 2:
                function_test = mean_every(args, epoch, generator)
            elif args.test_method == 3:
                function_test=test_autoencoder(args, epoch, generator)

            resultat = resultat.append({'EPOCHS' : epoch,
                                        'RMSE_AUTOENCODERS' : function_test}, ignore_index=True)
        if epoch % args.save_freq == 0:
        	#save the model
            return 
        if epoch == args.max_epoch-1:
            #Final Result  
            print(resultat)
            resultat.to_csv(log_path + '/resultat.txt', index=False,sep='\t',
                            float_format='%.3f')
if __name__ == '__main__':

	main()