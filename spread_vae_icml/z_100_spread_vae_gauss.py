vae_spread_opt= {
    'save': True,
    'save_epoch': [2,100,200,300,400,500,600,700,800,900,1000],
    'dataset': 'celebA',
    'spread': True,
    'image_dim': 64,
    'channels': 3,
    'all_pic_dim': 12288,
    'x_dim': 12288,
    'dis': 'gauss',
    'batch_size': 100,
    'training_epoch':1000,
    'noise_epoch': 1,
    'num_units': 1024,
    'net':'dc',
    'encoder_layers': 4,
    'decoder_layers': 4,
    'lr_decay': False,
    'decay_period': 30000,
    'decay_rate': 0.9,
    'optimizer':'adam',
    'kl_lr': 1e-4,
    'z_dim': 100,
    'kl_sgx': 0.5,
    'restore': False,
    'restore_variable':['encoder','decoder'],
    'path': '../result/spread_vae_gauss/',
    'work_dir': '../result/spread_vae_gauss/',
    'data_dir': '../../data/CelebA/images',
    'annealing': True,
    'start_annealing_epoch':500,
    'annealing_period':100,
    'annealing_end_std':0.1,
    'low_variance': True,
    'celebA_crop': 'closecrop',
    'input_normalize_sym': False
}



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import configs
from datahandler import DataHandler
from celeba_vae_injective import main
opt = vae_spread_opt
print(opt['data_dir'])
data = DataHandler(opt)
main(opt, data)
