# -*- coding: utf-8 -*-
"""
Definition of network architecture
Defintion of  convolutional layers
Defintion of batch normalization
"""

import theano
import theano.tensor as T
import numpy as np

class Network:
     """ prende in ingresso una lista di ConvLayer e l'immagine
         mettere l'immagine in ingresso per ora per il test """
     def __init__ (self,layer):
        self.layers = layer
        self.x = T.ftensor4('x')
        self.mode = T.lscalar('mode')
        
        lay1 = self.x
        skip_step=3
        last = False
        for i in range(len(self.layers)):
            if i==len(self.layers) - 1:
                last=True
            lay = self.layers[i]
            if i%skip_step==0 and i>0:
                lay1 = lay.get_result(lay1+self.layers[i-3].output[:,:,skip_step-1:-(skip_step-1),skip_step-1:-(skip_step-1)],last, self.mode)
            else:
                lay1 = lay.get_result(lay1,last, self.mode)
        self.output = lay1
        
     def build(self,img):
         I_in=T.ftensor4('I_in')
         setnet= theano.function([I_in],self.output,givens={self.x :I_in},allow_input_downcast=True)
         return setnet(img)


class BatchNormalization:
    def __init__ (self,  input_shape=None, alpha=0.9, gamma=None, beta=None, mean=None, var=None, shape=None):
        
        self.alpha=alpha
        self.gamma=theano.shared(np.asarray((gamma),dtype=theano.config.floatX), name='gamma', borrow=True)
        self.beta=theano.shared(np.asarray((beta),dtype=theano.config.floatX), name='beta', borrow=True)
        self.mean=theano.shared(np.asarray((mean),dtype=theano.config.floatX), name='mean', borrow=True)
        self.var=theano.shared(np.asarray((var),dtype=theano.config.floatX), name='var', borrow=True)
        self.input_shape=tuple(shape)
        
    def get_result(self,input,mode):
            
        epsilon=1e-16
            
        now_mean1=T.switch(mode,T.mean(input,axis=(0,2,3)),self.mean)
        now_var1 =T.switch(mode, T.var(input,axis=(0,2,3)),self.var)
                        
        now_mean1 = now_mean1.dimshuffle('x',0,'x','x')
        now_var1 = now_var1.dimshuffle('x',0,'x','x')
        now_gamma = self.gamma.dimshuffle('x',0,'x','x')
        now_beta = self.beta.dimshuffle('x',0,'x','x')                        
            
        output = now_gamma * (input - now_mean1) / T.sqrt(now_var1+epsilon) + now_beta
        
        return output
        
class BN_convLayer:
    def __init__  (self, weigth=None, mean=None, var=None, gamma=None, beta=None, bias=None, BN=True ):
                
       self.BN = BN        
       self.filter_shape=weigth.shape
       if self.BN == True:
           self.w=theano.shared(np.asarray((weigth),dtype=theano.config.floatX),name='w',borrow=True)
           self.BNlayer=BatchNormalization(gamma=gamma,beta=beta,mean=mean,var=var,shape=self.w.eval().shape)
       else:
           self.w=theano.shared(np.asarray((weigth),dtype=theano.config.floatX),name='w',borrow=True)
           self.b=theano.shared(np.asarray((bias),dtype=theano.config.floatX),name='b',borrow=True)
           
            
    def get_result(self, inpt, last,mode):
        
        self.input=inpt
        conv_out=T.nnet.conv2d(self.input, self.w,
                        filter_shape=self.filter_shape, filter_flip=False)
            
        if self.BN==True:
            out=self.BNlayer.get_result(conv_out,0)
            self.output=T.nnet.relu(out)
        else:
            out = conv_out+self.b.dimshuffle('x',0,'x','x')
            if last:
                self.output = out
            else:
                self.output=T.nnet.relu(out)
                            
        return self.output          
        
