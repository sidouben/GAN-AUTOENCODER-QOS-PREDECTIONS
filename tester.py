import pandas as pd
from model import build_autoencoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np

def test_autoencoder(args, epoch, generator):
    mean_final=0
    def get_invalid_indices(vecteur):
        indice_1 = np.argwhere(vecteur > 19.).flatten()  # index of invalid values (no reponse after invocation)
        indice_2 = np.argwhere(vecteur <= 0.).flatten()  # index of invalid values (none invocation)
        indice_invalid_values_ds1 = np.append(indice_1, indice_2) # concatenate of all invalid valeus
        return indice_invalid_values_ds1
    
    #Test set
    rtMatrix = pd.read_csv('./dataset/rtMatrix.txt', delimiter="\t", header=None)
    rtMatrix = rtMatrix.drop(5825, axis=1)
    
    #get weight after training
    lay1= generator.layers[0].get_weights()[0].T
    lay3= generator.layers[3].get_weights()[0].T
    lay6= generator.layers[6].get_weights()[0].T
    lay9= generator.layers[9].get_weights()[0].T
        
    ba1= generator.layers[0].get_weights()[1]
    ba3= generator.layers[3].get_weights()[1]
    ba6= generator.layers[6].get_weights()[1]
    ba9= generator.layers[9].get_weights()[1]        
    
    autoencoder = build_autoencoder(args)
    
    K = tf.keras.backend
    K.set_value(autoencoder.layers[0].weights[0], lay9)
    K.set_value(autoencoder.layers[3].weights[0], lay6)
    K.set_value(autoencoder.layers[6].weights[0], lay3)
    K.set_value(autoencoder.layers[9].weights[0], lay1)
    
    K.set_value(autoencoder.layers[0].weights[1], ba6)
    K.set_value(autoencoder.layers[3].weights[1], ba3)
    K.set_value(autoencoder.layers[6].weights[1], ba1)
    #K.set_value(autoencoder.layers[9].weights[1], ba1)
    
    #339 users
    
    for i in range(args.rows):
        indice_invalid_values = get_invalid_indices(rtMatrix.iloc[i,:].to_numpy())
        vecteur_user_ds1 = np.delete(rtMatrix.iloc[i,:].to_numpy(), indice_invalid_values, axis=None)
        vecteur_user_ds1 = np.expand_dims(vecteur_user_ds1, axis=0)
        rtMatrixt = np.expand_dims(rtMatrix.iloc[i], axis=0)
        
        train_x = rtMatrixt / 20
        
        encoded_data = autoencoder.predict(train_x)
        
        gen_qos = generator.predict(encoded_data)
        
        gen_qos = 20 * gen_qos
        reduire = np.delete(gen_qos, indice_invalid_values, axis=None)
        reduire= np.expand_dims(reduire, axis=0)
        
        rms = mean_squared_error(vecteur_user_ds1, reduire, squared=False)

        mean_final = mean_final + rms
    result = mean_final / args.rows
    print('epoch', epoch,' Mean RMSE Error using the autoencoders in test set ',result,' \n')
    return result

def mean_every(args, epoch, generator):
    test = pd.read_csv('./dataset/test.txt', delimiter="\t", header=None) 
    t_mean=test.mean()
    mean_final=0
    
    #66 users
    for i in range(args.rows_test_split):
        rt=test.iloc[i]
        fin=np.expand_dims(rt, axis=0)
            
        noise = np.random.normal(0, 1, (1, args.latent_dim))
        gen_qos = generator.predict(noise)
        # Rescale 
        gen_qos = 20 * gen_qos
        reduire=np.squeeze(gen_qos, axis=1)
        df = pd.DataFrame(data=reduire)
        mean = df.mean()
        
        rms = mean_squared_error(fin, reduire, squared=False)
        mean_final = mean_final + rms
    result = mean_final/args.rows_test_split

    print('epoch', epoch ,
          ' Mean of all RMSE Error between original vector and generated vector' ,result,
          ' ' )
    return result

def mean_all(args, epoch, generator,d_loss, a_loss, g_loss):
    test = pd.read_csv('./dataset/test.txt', delimiter="\t", header=None)
    t_mean=test.mean()
    noise = np.random.normal(0, 1, (args.rows_test_split, args.latent_dim))
    gen_qos = generator.predict(noise)
    # Rescale 
    gen_qos = 20 * gen_qos
    reduire=np.squeeze(gen_qos, axis=1)
    df = pd.DataFrame(data=reduire)
    mean = df.mean()
    rms = mean_squared_error(t_mean, mean, squared=False)
    
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] " % (epoch, d_loss, a_loss, g_loss))
    print('epoch',epoch ,
          ' RMSE Error between mean of all vector original and vector generated ', rms,
          '' )
    
    return  rms