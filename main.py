# -*- coding: utf-8 -*-
"""
INSERIRE COPYRIGHT
This is the code for MONet: Multi-Objecive CNN Based Algorithm for SAR Despecking
Take a noisy image with single look speckle in amplitude format and produce the filtered output

The code referes to the paper:
    S.Vitale, G.Ferraioli, V.Pascazio "Multi-Objecive CNN Based Algorithm for SAR Despecking", 2020, Accepted to TGRS

"""
import os
import argparse

parser = argparse.ArgumentParser( 
        description = 'MONet: Multi-Objective CNN based SAR Despeckling')
                        
parser.add_argument('-g', '--gpu', action='store_true',default=False,
                        help='the identifier of the used GPU.')

parser.add_argument('-a', '--area', type=str, default='storagetanks',
                        help='the identifer of the used sensor.')

config, _ = parser.parse_known_args()
area = config.area                        
if (config.gpu):
	os.environ["THEANO_FLAGS"]='device=cuda0,floatX=float32,init_gpu_device=cuda0'
else:
	os.environ["THEANO_FLAGS"] = "floatX=float32"

import theano
import numpy as np
import scipy.io as sio
from testing_model import Network, BN_convLayer
from test_ import DNN_test

#"Select the simulated image to despeckle: images are sample from UCMerced dataset "
#area= 'baseballdiamond'
#area = 'golfcourse'
#area = 'storagetanks'

#%% Loading Data
img_path = './imgs/'+area+'.mat'
model_path = './model/model.mat'

#data are with single look speckle in ampliyude format
I =sio.loadmat(img_path,squeeze_me=True)
I_ref = np.asarray(I['ref'],dtype=theano.config.floatX)
I_noisy = np.asarray(I['noisy'],dtype=theano.config.floatX)

DNN_model=sio.loadmat(model_path,squeeze_me=True)

#compute the scope of the network
net_scope=0
for i in range(len(DNN_model['layers'])-1):
    if len(DNN_model['layers'][i].shape) >2:
        net_scope+=DNN_model['layers'][i].shape[-1]-1
        
#%% building Network    
layer=[]
layer.append(BN_convLayer(weigth=np.expand_dims(DNN_model['layers'][0],axis=1),bias=DNN_model['layers'][1],BN=False))
for i in xrange(2,len(DNN_model['layers'])-2,5):
	layer.append(BN_convLayer(weigth=DNN_model['layers'][i],gamma= DNN_model['layers'][i+1],beta=DNN_model['layers'][i+2],mean=DNN_model['layers'][i+3], var=DNN_model['layers'][i+4]))
layer.append(BN_convLayer(weigth=np.expand_dims(DNN_model['layers'][-2],axis=0),bias=np.expand_dims(DNN_model['layers'][-1],axis=0),BN=False))
    
net=Network(layer)
    
#%% testing
"Despeckling"
I_out = DNN_test(I_noisy,net)
I_out = np.squeeze(I_out)
        
        
#%% Visualization:
import matplotlib.pyplot as plt    
plt.close('all')

plt.figure()    
plt.subplot(131),plt.imshow(I_noisy[net_scope/2:-net_scope/2,net_scope/2:-net_scope/2], cmap = 'gray',vmin=0,vmax=255),plt.title('noisy')
plt.subplot(132),plt.imshow(I_ref[net_scope/2:-net_scope/2,net_scope/2:-net_scope/2], cmap = 'gray',vmin=0,vmax=255),plt.title('ref')
plt.subplot(133),plt.imshow(I_out , cmap = 'gray',vmin=0,vmax=255),plt.title('image -filtered')



