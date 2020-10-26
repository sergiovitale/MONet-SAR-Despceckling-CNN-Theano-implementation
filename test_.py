# -*- coding: utf-8 -*-

import numpy as np
import time

def DNN_test(I,net):
    """
    Despeckling of single look image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    I:      single look noisy imge
    net:    pretrained network
    I_out:  filtered image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    I_in = I.astype('float32')   
    
    I_in = np.expand_dims(I_in,axis=0)
    I_in = np.expand_dims(I_in,axis=1)

    start = time.time()
    I_out=net.build(I_in)
    stop = time.time()
    print('Testing Time: %0.4f'%(stop-start))
    
    return I_out
    
    
