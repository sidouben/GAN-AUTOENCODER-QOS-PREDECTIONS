import argparse


def parse_args():

    parser = argparse.ArgumentParser('imaginator training config')


    parser.add_argument('--max_epoch', type=int, default=20000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=273, help='train batch size')



    parser.add_argument('--dataset', type=str, default='mug', choices=['mug', 'uva'], help='dataset choice')

    
    parser.add_argument('--save_freq', type=int, default=100, help='model save frequence')
    parser.add_argument('--sample_interval', type=int, default=200, help='model test frequence')
    
    parser.add_argument('--save_path', type=str, default='./result', help='model and log save path')
    parser.add_argument('--data_path', type=str, default='', help='dataset path')

    parser.add_argument('--random_seed', type=int, default='12345')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent_dim')
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--cols', type=int, default='5825')
    parser.add_argument('--rows', type=int, default='339')
    parser.add_argument('--rows_test_split', type=int, default='66')
    parser.add_argument('--test_method', type=int, default=3, help='test_method')
    

    
    args = parser.parse_args()
    
    return args