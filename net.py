#! /usr/bin/python
# Author: Jasjeet Dhaliwal

import os, time, pickle
import numpy as np
from softmax import *
from gradient_checker import *
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
NUM_CLASSES = 10



def get_params(inp_size, num_classes, hidden_size):
  """Define the network as a graph

  Inputs: 
    inp_size(int): Number of incoming synapses to the first layer
    num_classes(int): number of discrete output classes
    hidden_size(int): Number of outgoing synapses from each neuron
     
  Ouput: 
    params(tuple): tuple of parameter objects
  """
  w1=np.random.normal(0, 0.01, (inp_size, hidden_size))
  b1=np.random.normal(0,0.01, hidden_size)

  w2=np.random.normal(0, 0.01, (hidden_size, hidden_size))
  b2=np.random.normal(0,0.01, hidden_size) 

  w3=np.random.normal(0, 0.01, (2*hidden_size, hidden_size)) 
  b3=np.random.normal(0,0.01, hidden_size) 

  w4=np.random.normal(0, 0.01, (3*hidden_size, hidden_size)) 
  b4=np.random.normal(0,0.01, hidden_size) 

  w5=np.random.normal(0, 0.01, (4*hidden_size, hidden_size)) 
  b5=np.random.normal(0,0.01, hidden_size) 

  w6=np.random.normal(0, 0.01, (5*hidden_size, hidden_size)) 
  b6=np.random.normal(0,0.01, hidden_size) 

  v_board=np.random.normal(0,0.1, (5*hidden_size, num_classes)) 
  params = (w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, v_board)

  w1_v = np.zeros_like(w1)
  b1_v = np.zeros_like(b1)
  w2_v = np.zeros_like(w2)
  b2_v = np.zeros_like(b2)
  w3_v = np.zeros_like(w3)
  b3_v = np.zeros_like(b3)
  w4_v = np.zeros_like(w4)
  b4_v = np.zeros_like(b4)
  w5_v = np.zeros_like(w5)
  b5_v = np.zeros_like(b5)
  w6_v = np.zeros_like(w6)
  b6_v = np.zeros_like(b6)
  v_board_v = np.zeros_like(v_board)

  velocity = [w1_v, b1_v, w2_v, b2_v, w3_v, b3_v, w4_v, b4_v, w5_v, b5_v, w6_v, b6_v, v_board_v]
  return velocity, params

def forward(params, inputs, hidden_size):
  """Performs forward pass on net with batch of inputs
  Input: 
    params(tuple): tuple of network parameters
    inputs(numpy matrix): inputs of shape n x s, n is batch size and s is input size
    hidden_size(int): Number of outgoing synapses from each neuron

  Output:
    logit(numpy matrix): numpy matrix of size n x num_classes
    back_params(tuple): tuple of values required for the backward pass
  """

  w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, v_board = params

  l1_in = inputs
  l2_in = np.zeros((BATCH_SIZE, hidden_size))
  l3_in = np.zeros((BATCH_SIZE, 2*hidden_size))
  l4_in = np.zeros((BATCH_SIZE, 3*hidden_size))
  l5_in = np.zeros((BATCH_SIZE, 4*hidden_size))
  l6_in = np.zeros((BATCH_SIZE, 5*hidden_size))
  v_in = np.zeros((BATCH_SIZE, 5*hidden_size))

  #We break down function into time steps. With T1 being input fed 
  #into layer 1

  #T1
  #Calculate activations
  h1_1 = np.maximum(np.dot(l1_in, w1) + b1, 0) 
  dh1_1_w1 = np.zeros(w1.shape)
  dh = np.int16(h1_1 > 0)
  dh1_1_w1 = np.dot(l1_in.T, dh)
  dh1_1_w1 /= BATCH_SIZE
  dh1_1_b1 = np.sum( np.int16(h1_1>0) ,axis=0)

  #Prepare inputs for the next time step
  l2_in[:] = h1_1
  l3_in[:, 0:hidden_size]= h1_1
  l4_in[:, 0:hidden_size]= h1_1
  l5_in[:, 0:hidden_size]= h1_1
  l6_in[:, 0:hidden_size]= h1_1
  
  #T2
  #Calculate activations and backward pass derivatives 
  h2_2 = np.maximum(np.dot(l2_in, w2)+b2, 0)
  dh2_2_w2 = np.zeros(w2.shape)
  dh = np.int16(h2_2 > 0)
  dh2_2_w2 = np.dot(l2_in.T, dh)
  dh2_2_w2 /= BATCH_SIZE
  dh2_2_b2 = np.sum( np.int16(h2_2>0) ,axis=0)

  dh2_2_l2_in = np.zeros(l2_in.shape)
  dh2_2_l2_in = np.dot(dh, w2.T)

  h3_2 = np.maximum(np.dot(l3_in, w3)+b3, 0)
  dh3_2_w3 = np.zeros(w3.shape)
  dh = np.int16(h3_2 > 0)
  dh3_2_w3 = np.dot(l3_in.T, dh)
  dh3_2_w3 /= BATCH_SIZE
  dh3_2_b3 = np.sum( np.int16(h3_2>0) ,axis=0)

  dh3_2_l3_in = np.zeros(l3_in.shape)
  dh3_2_l3_in = np.dot(dh, w3.T)

  h4_2 = np.maximum(np.dot(l4_in, w4)+b4, 0)
  dh4_2_w4 = np.zeros(w4.shape)
  dh = np.int16(h4_2 > 0)
  dh4_2_w4 = np.dot(l4_in.T, dh)
  dh4_2_w4 /= BATCH_SIZE
  dh4_2_b4 = np.sum( np.int16(h4_2>0) ,axis=0)

  dh4_2_l4_in = np.zeros(l4_in.shape)
  dh4_2_l4_in = np.dot(dh, w4.T)

  h5_2 = np.maximum(np.dot(l5_in, w5)+b5, 0)
  dh5_2_w5 = np.zeros(w5.shape)
  dh = np.int16(h5_2 > 0)
  dh5_2_w5 = np.dot(l5_in.T, dh)
  dh5_2_w5 /= BATCH_SIZE
  dh5_2_b5 = np.sum( np.int16(h5_2>0) ,axis=0)

  dh5_2_l5_in = np.zeros(l5_in.shape)
  dh5_2_l5_in = np.dot(dh, w5.T)


  h6_2 = np.maximum(np.dot(l6_in, w6)+b6, 0)
  dh6_2_w6 = np.zeros(w6.shape)
  dh = np.int16(h6_2 > 0)
  dh6_2_w6 = np.dot(l6_in.T, dh)
  dh6_2_w6 /= BATCH_SIZE
  dh6_2_b6 = np.sum( np.int16(h6_2>0) ,axis=0)

  dh6_2_l6_in = np.zeros(l6_in.shape)
  dh6_2_l6_in = np.dot(dh, w6.T)


  #Prepare inputs for the next time step
  l3_in[:] = 0.
  l4_in[:] = 0.
  l5_in[:] = 0.
  l6_in[:] = 0.

  l3_in[:, hidden_size:2*hidden_size]= h2_2

  l4_in[:, hidden_size:2*hidden_size]= h2_2
  l4_in[:, 2*hidden_size:3*hidden_size] = h3_2

  l5_in[:, hidden_size:2*hidden_size]= h2_2
  l5_in[:, 2*hidden_size:3*hidden_size] = h3_2
  l5_in[:, 3*hidden_size:4*hidden_size] = h4_2
  
  l6_in[:, hidden_size:2*hidden_size]= h2_2
  l6_in[:, 2*hidden_size:3*hidden_size] = h3_2
  l6_in[:, 3*hidden_size:4*hidden_size] = h4_2
  l6_in[:, 4*hidden_size:5*hidden_size] = h5_2
   
  v_in[:,0:hidden_size] = h6_2

  #T3
  #Calculate activations 
  h3_3 = np.maximum(np.dot(l3_in, w3)+b3, 0)
  dh3_3_w3 = np.zeros(w3.shape)
  dh = np.int16(h3_3 > 0)
  dh3_3_w3 = np.dot(l3_in.T, dh)
  dh3_3_w3 /= BATCH_SIZE
  dh3_3_b3 = np.sum( np.int16(h3_3>0) ,axis=0)

  dh3_3_l3_in = np.zeros(l3_in.shape)
  dh3_3_l3_in = np.dot(dh, w3.T)

  h4_3 = np.maximum(np.dot(l4_in, w4)+b4, 0)
  dh4_3_w4 = np.zeros(w4.shape)
  dh = np.int16(h4_3 > 0)
  dh4_3_w4 = np.dot(l4_in.T, dh)
  dh4_3_w4 /= BATCH_SIZE
  dh4_3_b4 = np.sum( np.int16(h4_3>0) ,axis=0)

  dh4_3_l4_in = np.zeros(l4_in.shape)
  dh4_3_l4_in = np.dot(dh, w4.T)

  h5_3 = np.maximum(np.dot(l5_in, w5)+b5, 0)
  dh5_3_w5 = np.zeros(w5.shape)
  dh = np.int16(h5_3 > 0)
  dh5_3_w5 = np.dot(l5_in.T, dh)
  dh5_3_w5 /= BATCH_SIZE
  dh5_3_b5 = np.sum( np.int16(h5_3>0) ,axis=0)

  dh5_3_l5_in = np.zeros(l5_in.shape)
  dh5_3_l5_in = np.dot(dh, w5.T)


  h6_3 = np.maximum(np.dot(l6_in, w6)+b6, 0)
  dh6_3_w6 = np.zeros(w6.shape)
  dh = np.int16(h6_3 > 0)
  dh6_3_w6 = np.dot(l6_in.T, dh)
  dh6_3_w6 /= BATCH_SIZE
  dh6_3_b6 = np.sum( np.int16(h6_3>0) ,axis=0)

  dh6_3_l6_in = np.zeros(l6_in.shape)
  dh6_3_l6_in = np.dot(dh, w6.T)


  #Prepare inputs for the next time step
  l4_in[:] = 0.
  l5_in[:] = 0.
  l6_in[:] = 0.

  l4_in[:, 2*hidden_size:3*hidden_size] = h3_3

  l5_in[:, 2*hidden_size:3*hidden_size] = h3_3
  l5_in[:, 3*hidden_size:4*hidden_size] = h4_3
  
  l6_in[:, 2*hidden_size:3*hidden_size] = h3_3
  l6_in[:, 3*hidden_size:4*hidden_size] = h4_3
  l6_in[:, 4*hidden_size:5*hidden_size] = h5_3

  v_in[:,hidden_size:2*hidden_size] = h6_3

  #T4
  #Calculate activations 
  h4_4 = np.maximum(np.dot(l4_in, w4)+b4, 0)
  dh4_4_w4 = np.zeros(w4.shape)
  dh = np.int16(h4_4 > 0)
  dh4_4_w4 = np.dot(l4_in.T, dh)
  dh4_4_w4 /= BATCH_SIZE
  dh4_4_b4 = np.sum( np.int16(h4_4>0) ,axis=0)

  dh4_4_l4_in = np.zeros(l4_in.shape)
  dh4_4_l4_in = np.dot(dh, w4.T)

  h5_4 = np.maximum(np.dot(l5_in, w5)+b5, 0)
  dh5_4_w5 = np.zeros(w5.shape)
  dh = np.int16(h5_4 > 0)
  dh5_4_w5 = np.dot(l5_in.T, dh)
  dh5_4_w5 /= BATCH_SIZE
  dh5_4_b5 = np.sum( np.int16(h5_4>0) ,axis=0)

  dh5_4_l5_in = np.zeros(l5_in.shape)
  dh5_4_l5_in = np.dot(dh, w5.T)


  h6_4 = np.maximum(np.dot(l6_in, w6)+b6, 0)
  dh6_4_w6 = np.zeros(w6.shape)
  dh = np.int16(h6_4 > 0)
  dh6_4_w6 = np.dot(l6_in.T, dh)
  dh6_4_w6 /= BATCH_SIZE
  dh6_4_b6 = np.sum( np.int16(h6_4>0) ,axis=0)

  dh6_4_l6_in = np.zeros(l6_in.shape)
  dh6_4_l6_in = np.dot(dh, w6.T)


  #Prepare inputs for the next time step
  l5_in[:] = 0.
  l6_in[:] = 0.

  l5_in[:, 3*hidden_size:4*hidden_size] = h4_4
  
  l6_in[:, 3*hidden_size:4*hidden_size] = h4_4
  l6_in[:, 4*hidden_size:5*hidden_size] = h5_4

  v_in[:,2*hidden_size:3*hidden_size] = h6_4


  #T5
  #Calculate activations 
  h5_5 = np.maximum(np.dot(l5_in, w5)+b5, 0)
  dh5_5_w5 = np.zeros(w5.shape)
  dh = np.int16(h5_5 > 0)
  dh5_5_w5 = np.dot(l5_in.T, dh)
  dh5_5_w5 /= BATCH_SIZE
  dh5_5_b5 = np.sum( np.int16(h5_5>0) ,axis=0)

  dh5_5_l5_in = np.zeros(l5_in.shape)
  dh5_5_l5_in = np.dot(dh, w5.T)


  h6_5 = np.maximum(np.dot(l6_in, w6)+b6, 0)
  dh6_5_w6 = np.zeros(w6.shape)
  dh = np.int16(h6_5 > 0)
  dh6_5_w6 = np.dot(l6_in.T, dh)
  dh6_5_w6 /= BATCH_SIZE
  dh6_5_b6 = np.sum( np.int16(h6_5>0) ,axis=0)

  dh6_5_l6_in = np.zeros(l6_in.shape)
  dh6_5_l6_in = np.dot(dh, w6.T)

  #Prepare inputs for the next time step
  l6_in[:] = 0.

  l6_in[:, 4*hidden_size:5*hidden_size] = h5_5

  v_in[:,3*hidden_size:4*hidden_size] = h6_5

  #T6
  #Calculate activations 
  h6_6 = np.maximum(np.dot(l6_in, w6)+b6, 0)
  dh6_6_w6 = np.zeros(w6.shape)
  dh = np.int16(h6_6 > 0)
  dh6_6_w6 = np.dot(l6_in.T, dh)
  dh6_6_w6 /= BATCH_SIZE
  dh6_6_b6 = np.sum( np.int16(h6_6>0) ,axis=0)

  dh6_6_l6_in = np.zeros(l6_in.shape)
  dh6_6_l6_in = np.dot(dh, w6.T)

  v_in[:,4*hidden_size:5*hidden_size] = h6_6

  logit = np.dot(v_in, v_board)

  params_back = (dh1_1_w1,\
                 dh2_2_l2_in, dh2_2_w2, dh3_2_l3_in, dh3_2_w3, dh4_2_l4_in,\
                 dh4_2_w4, dh5_2_l5_in, dh5_2_w5, dh6_2_l6_in, dh6_2_w6,\
                 dh3_3_l3_in, dh3_3_w3, dh4_3_l4_in, dh4_3_w4, dh5_3_l5_in, dh5_3_w5, dh6_3_l6_in, dh6_3_w6,\
                 dh4_4_l4_in, dh4_4_w4, dh5_4_l5_in, dh5_4_w5, dh6_4_l6_in, dh6_4_w6, \
                 dh5_5_l5_in, dh5_5_w5,dh6_5_l6_in, dh6_5_w6,\
                 dh6_6_l6_in, dh6_6_w6, \
                 v_in, v_board)
  return logit, params_back

def evaluate_gradients(params, params_back, df, hidden_size, reg):
  """Evaluates gradients through network
  Input: 
    params_back(tuple): Tuple of values required for backward pass
    df(vector): 1 x c vector of gradients with respect to the loss
    hidden_size(int): Number of hidden layers
    reg(float): regularization strength

  Output: 
    gradiens(tuple): tuple of gradients of each param w.r.t loss
  """

  h = hidden_size
  w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, v_board = params
  dh1_1_w1,\
  dh2_2_l2_in, dh2_2_w2, dh3_2_l3_in, dh3_2_w3, dh4_2_l4_in,\
  dh4_2_w4, dh5_2_l5_in, dh5_2_w5, dh6_2_l6_in, dh6_2_w6,\
  dh3_3_l3_in, dh3_3_w3, dh4_3_l4_in, dh4_3_w4, dh5_3_l5_in, dh5_3_w5, dh6_3_l6_in, dh6_3_w6,\
  dh4_4_l4_in, dh4_4_w4, dh5_4_l5_in, dh5_4_w5, dh6_4_l6_in, dh6_4_w6, \
  dh5_5_l5_in, dh5_5_w5, dh6_5_l6_in, dh6_5_w6,\
  dh6_6_l6_in, dh6_6_w6, \
  v_in, v_board = params_back

  #dL/dv_in
  t1 = v_board * df
  t2 = np.ones((NUM_CLASSES, BATCH_SIZE))
  dL_v_in = np.dot(t1,t2)
  dL_v_in = np.sum(dL_v_in.T, axis=0)/BATCH_SIZE
  
  t1 = np.ones((BATCH_SIZE, NUM_CLASSES)) /BATCH_SIZE
  dL_v_board = np.dot(v_in.T, t1) 
  dL_v_board = dL_v_board*df

  #W6
  #dh62_dw6
  tmp = dL_v_in[0:h]
  dL_h6_2_w6 = np.zeros(dh6_2_w6.shape)
  dL_h6_2_w6[0:h, :] = dh6_2_w6[0:h, :] * tmp

  #dh63_dw6
  tmp =dL_v_in[h:2*h]
  tmp = np.concatenate((tmp, tmp, tmp, tmp))
  dL_h6_3_w6 = np.zeros(dh6_3_w6.shape)
  dL_h6_3_w6[h:5*h, :] = dh6_3_w6[h:5*h,:]*tmp[:,None]
 
  
  #dh64_dw6
  tmp =dL_v_in[2*h:3*h]
  tmp = np.concatenate((tmp, tmp, tmp))
  dL_h6_4_w6 = np.zeros(dh6_4_w6.shape)
  dL_h6_4_w6[2*h:5*h, :] = dh6_4_w6[2*h:5*h,:]*tmp[:,None]


  #dh65_dw6
  tmp =dL_v_in[3*h:4*h]
  tmp = np.concatenate((tmp, tmp))
  dL_h6_5_w6 = np.zeros(dh6_5_w6.shape)
  dL_h6_5_w6[3*h:5*h, :] = dh6_5_w6[3*h:5*h,:]*tmp[:,None]

  #dh66_dw6
  tmp =dL_v_in[4*h:5*h]
  dL_h6_6_w6 = np.zeros(dh6_6_w6.shape)
  dL_h6_6_w6[4*h:5*h, :] = dh6_6_w6[4*h:5*h,:]*tmp[:,None]

  dL_w6 = dL_h6_2_w6 + dL_h6_3_w6 + dL_h6_4_w6 +\
          dL_h6_5_w6 + dL_h6_6_w6

  #dL_b6
  dL_b6 = dL_v_in[0:h] + dL_v_in[h:2*h] + dL_v_in[2*h:3*h]+\
          dL_v_in[3*h:4*h] + dL_v_in[4*h:5*h]

  dL_b6 /= BATCH_SIZE
  #W5
  #dh62_dw5
  #dL_h6_2_w5 = np.zeros(dh5_2_w5.shape)

  #dh63_dw5
  t1 =dL_v_in[h:2*h]
  t2 = dh6_3_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  dL_h6_3_w5 = np.zeros(dh5_2_w5.shape)
  dL_h6_3_w5[0:h,:] = dh5_2_w5[0:h,:] * t2

  
  #dh64_dw5
  t1 =dL_v_in[2*h:3*h]
  t2 = dh6_4_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE
  t2 = np.concatenate((t2, t2, t2))

  dL_h6_4_w5 = np.zeros(dh5_3_w5.shape)
  dL_h6_4_w5[h:4*h,:] = dh5_3_w5[h:4*h,:] * t2[:,None]



  #dh65_dw5
  t1 = dL_v_in[3*h:4*h]
  t2 = dh6_5_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE
  t2 = np.concatenate((t2, t2))

  dL_h6_5_w5 = np.zeros(dh5_4_w5.shape)
  dL_h6_5_w5[2*h:4*h,:] = dh5_4_w5[2*h:4*h,:] * t2[:,None]


  #dh66_dw5
  t1 = dL_v_in[4*h:5*h]
  t2 = dh6_6_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  dL_h6_6_w5 = np.zeros(dh5_5_w5.shape)
  dL_h6_6_w5[3*h:4*h,:] = dh5_5_w5[3*h:4*h,:] * t2[:,None]

  dL_w5 = dL_h6_3_w5 + dL_h6_4_w5 +\
          dL_h6_5_w5 + dL_h6_6_w5
  
  #dL_b5
  dL_b5 = dL_v_in[h:2*h]*np.sum(dh6_3_l6_in[:,4*h:5*h], axis=0) +\
          dL_v_in[2*h:3*h]*np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0) +\
          dL_v_in[3*h:4*h]*np.sum(dh6_5_l6_in[:,4*h:5*h],axis=0) +\
          dL_v_in[4*h:5*h]*np.sum(dh6_6_l6_in[:,4*h:5*h],axis=0)
  dL_b5 /= BATCH_SIZE

  #W4
  #dh62_dw4
  #dL_h6_2_w4 = np.zeros(dh4_2_w4.shape)

  #dh63_dw4
  t1 =dL_v_in[h:2*h]
  t2 = dh6_3_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  dL_h6_3_w4 = np.zeros(dh4_2_w4.shape)
  dL_h6_3_w4[0:h,:] = dh4_2_w4[0:h,:] * t2

  
  #dh64_dw4
  t1 =dL_v_in[2*h:3*h]
  t2 = dh6_4_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE
  t2 = np.concatenate((t2, t2))

  dL_h6_4_w4 = np.zeros(dh4_3_w4.shape)
  dL_h6_4_w4[h:3*h,:] = dh4_3_w4[h:3*h,:] * t2[:,None]

  #dh65_dw4
  t1 = dL_v_in[3*h:4*h]
  t2 = dh6_5_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  dL_h6_5_w4_1 = np.zeros(dh4_4_w4.shape)
  dL_h6_5_w4_1[2*h:3*h,:] = dh4_4_w4[2*h:3*h,:] * t2[:,None]


  t3 = dh6_5_l6_in[:,4*h:5*h]*t1
  t3 = np.sum(t3, axis=0) /BATCH_SIZE

  t4 = dh5_4_l5_in[:,3*h:4*h]*t3
  t4 = np.sum(t4, axis=0) /BATCH_SIZE
  t4 = np.concatenate((t4,t4))

  dL_h6_5_w4_2 = np.zeros(dh4_3_w4.shape)
  dL_h6_5_w4_2[h:3*h, :] = dh4_3_w4[h:3*h,:]*t4[:,None]

  dL_h6_5_w4 = dL_h6_5_w4_1 + dL_h6_5_w4_2

  #dh66_dw4
  t1 = dL_v_in[4*h:5*h]
  t2 = dh6_6_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh5_5_l5_in[:,3*h:4*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  dL_h6_6_w4 = np.zeros(dh4_4_w4.shape)
  dL_h6_6_w4[2*h:3*h,:] = dh4_4_w4[2*h:3*h,:] * t3[:,None]

  dL_w4 = dL_h6_3_w4 + dL_h6_4_w4 +\
          dL_h6_5_w4 + dL_h6_6_w4


  #dL_b4
  dL_b4 = dL_v_in[h:2*h]*np.sum(dh6_3_l6_in[:,3*h:4*h], axis=0) +\
          dL_v_in[2*h:3*h]*(np.sum(dh6_4_l6_in[:,3*h:4*h], axis=0) +\
                           (np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0)*\
                                np.sum(dh5_3_l5_in[:,3*h:4*h],axis=0))) +\
         dL_v_in[3*h:4*h]*(np.sum(dh6_5_l6_in[:,3*h:4*h],axis=0) +\
                           (np.sum(dh6_5_l6_in[:,4*h:5*h],axis=0)*\
                             np.sum(dh5_4_l5_in[:,3*h:4*h],axis=0))) +\
         dL_v_in[4*h:5*h]*(np.sum(dh6_5_l6_in[:,4*h:5*h],axis=0)*\
                           np.sum(dh5_5_l5_in[:,3*h:4*h],axis=0))
  dL_b4 /= BATCH_SIZE


  #W3
  #dh62_dw3
  #dL_h6_2_w3 = np.zeros(dh3_2_w3.shape)

  #dh63_dw3
  t1 =dL_v_in[h:2*h]
  t2 = dh6_3_l6_in[:,2*h:3*h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  dL_h6_3_w3 = np.zeros(dh3_2_w3.shape)
  dL_h6_3_w3[0:h,:] = dh3_2_w3[0:h,:] * t2

  
  #dh64_dw3
  t1 =dL_v_in[2*h:3*h]
  t2 = dh6_4_l6_in[:,2*h:3*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  dL_h6_4_w3_1 = np.zeros(dh3_3_w3.shape)
  dL_h6_4_w3_1[h:2*h,:] = dh3_3_w3[h:2*h,:] * t2


  t3 = dh6_4_l6_in[:,3*h:4*h]*t1
  t3 = np.sum(t3, axis=0) / BATCH_SIZE
 
  t4 = dh4_3_l4_in[:,2*h:3*h] *t3
  t4 = np.sum(t4, axis=0) /BATCH_SIZE

  dL_h6_4_w3_2 = np.zeros(dh3_2_w3.shape)
  dL_h6_4_w3_2[0:h,:] = dh3_2_w3[0:h, :] * t4

  t5 = dh6_4_l6_in[:,4*h:5*h]*t1
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  t6 = dh5_3_l5_in[:, 2*h:3*h]*t5
  t6 = np.sum(t6, axis=0) / BATCH_SIZE

  dL_h6_4_w3_3 = np.zeros(dh3_2_w3.shape)
  dL_h6_4_w3_3[0:h,:] = dh3_2_w3[0:h,:]*t6

  dL_h6_4_w3 = dL_h6_4_w3_1 + dL_h6_4_w3_2 + dL_h6_4_w3_3

  #dh65_dw3
  t1 = dL_v_in[3*h:4*h]
  t2 = dh6_5_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh4_4_l4_in[:,2*h:3*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  dL_h6_5_w3_1 = np.zeros(dh3_3_w3.shape)
  dL_h6_5_w3_1[h:2*h,:] = dh3_3_w3[h:2*h,:] * t3


  t4 = dh6_5_l6_in[:,4*h:5*h]*t1
  t4 = np.sum(t4, axis=0) /BATCH_SIZE

  t5 = dh5_4_l5_in[:,2*h:3*h]*t4
  t5 = np.sum(t5, axis=0) /BATCH_SIZE

  dL_h6_5_w3_2_1 = np.zeros(dh3_3_w3.shape)
  dL_h6_5_w3_2_1[h:2*h,:] = dh3_3_w3[h:2*h, :]*t5

  t6 = dh5_4_l5_in[:,3*h:4*h]*t4
  t6 = np.sum(t6, axis=0) /BATCH_SIZE

  t7 = dh4_3_l4_in[:,2*h:3*h]*t6
  t7 = np.sum(t7, axis=0) / BATCH_SIZE

  dL_h6_5_w3_2_2 = np.zeros(dh3_2_w3.shape)
  dL_h6_5_w3_2_2[0:h,:] = dh3_2_w3[0:h,:]*t7

  dL_h6_5_w3_2 = dL_h6_5_w3_2_1 + dL_h6_5_w3_2_2
  dL_h6_5_w3 = dL_h6_5_w3_1 + dL_h6_5_w3_2

  #dh66_dw3
  t1 = dL_v_in[4*h:5*h]
  t2 = dh6_6_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh5_5_l5_in[:,3*h:4*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh4_4_l4_in[:,2*h:3*h]*t3
  t4 = np.sum(t4, axis=0) / BATCH_SIZE

  dL_h6_6_w3 = np.zeros(dh3_3_w3.shape)
  dL_h6_6_w3[h:2*h,:] = dh3_3_w3[h:2*h,:] * t4

  dL_w3 = dL_h6_3_w3 + dL_h6_4_w3 +\
          dL_h6_5_w3 + dL_h6_6_w3


  #dL_b3
  dL_b3 = dL_v_in[h:2*h]*np.sum(dh6_3_l6_in[:,2*h:3*h], axis=0) +\
          dL_v_in[2*h:3*h]*( (np.sum(dh6_4_l6_in[:,2*h:3*h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,3*h:4*h],axis=0)*\
                               np.sum(dh4_3_l4_in[:,2*h:3*h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,4*h:5*h], axis=0)*\
                               np.sum(dh5_3_l5_in[:,2*h:3*h],axis=0))) +\
          dL_v_in[3*h:4*h]*( (np.sum(dh6_4_l6_in[:,3*h:4*h],axis=0)*\
                               np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0)*\
                               ( (np.sum(dh5_4_l5_in[:,2*h:3*h],axis=0)) +\
                                 (np.sum(dh5_4_l5_in[:,3*h:4*h],axis=0)*\
                                  np.sum(dh4_3_l4_in[:,2*h:3*h],axis=0))))) +\
         dL_v_in[4*h:5*h]*(np.sum(dh6_6_l6_in[:,4*h:5*h],axis=0)*\
                           np.sum(dh5_5_l5_in[:,3*h:4*h],axis=0)*\
                           np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0))
  dL_b3 /= BATCH_SIZE

  #W2
  #dh62_dw2
  #dL_h6_2_w2 = np.zeros(dh2_2_w2.shape)

  #dh63_dw2
  t1 =dL_v_in[h:2*h]
  t2 = dh6_3_l6_in[:,h:2*h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  dL_h6_3_w2 = np.zeros(dh2_2_w2.shape)
  dL_h6_3_w2[:,:] = dh2_2_w2[:,:] * t2

  #dh64_dw3
  t1 =dL_v_in[2*h:3*h]
  t2 = dh6_4_l6_in[:,2*h:3*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh3_3_l3_in[:,h:2*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  dL_h6_4_w2_1 = np.zeros(dh2_2_w2.shape)
  dL_h6_4_w2_1[:,:] = dh2_2_w2[:,:] * t3

  t4 = dh6_4_l6_in[:,3*h:4*h]*t1
  t4 = np.sum(t4, axis=0) /BATCH_SIZE

  t5 = dh4_3_l4_in[:,h:2*h]*t4
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  dL_h6_4_w2_2 = np.zeros(dh2_2_w2.shape)
  dL_h6_4_w2_2[:,:] = dh2_2_w2[:,:] * t5


  t6 = dh6_4_l6_in[:,4*h:5*h]*t1
  t6 = np.sum(t6, axis=0) /BATCH_SIZE

  t7 = dh5_3_l5_in[:,h:2*h]*t6
  t7 = np.sum(t7, axis=0) / BATCH_SIZE

  dL_h6_4_w2_3 = np.zeros(dh2_2_w2.shape)
  dL_h6_4_w2_3[:,:] = dh2_2_w2[:,:] * t7

  dL_h6_4_w2 = dL_h6_4_w2_1 + dL_h6_4_w2_2 + dL_h6_4_w2_3

  #dh65_dw2
  t1 = dL_v_in[3*h:4*h]
  t2 = dh6_5_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh4_4_l4_in[:,h:2*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh3_3_l3_in[:,h:2*h]*t3
  t4 = np.sum(t4, axis=0) / BATCH_SIZE

  dL_h6_5_w2_1 = np.zeros(dh2_2_w2.shape)
  dL_h6_5_w2_1[:,:] = dh2_2_w2[:,:] * t4

  t5 = dh6_5_l6_in[:,4*h:5*h]*t1
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  t6 = dh5_4_l5_in[:,2*h:3*h]*t5
  t6 = np.sum(t6, axis=0) / BATCH_SIZE

  t7 = dh3_3_l3_in[:,h:2*h]*t6
  t7 = np.sum(t7, axis=0) / BATCH_SIZE

  dL_h6_5_w2_2_1 = np.zeros(dh2_2_w2.shape)
  dL_h6_5_w2_2_1 = dh2_2_w2[:, :]*t7

  t8 = dh5_4_l5_in[:,3*h:4*h]*t5
  t8 = np.sum(t8, axis=0) / BATCH_SIZE

  t9 = dh4_3_l4_in[:,h:2*h]*t8
  t9 = np.sum(t9, axis=0) / BATCH_SIZE

  dL_h6_5_w2_2_2 = np.zeros(dh2_2_w2.shape)
  dL_h6_5_w2_2_2 = dh2_2_w2[:, :]*t9

  dL_h6_5_w2 = dL_h6_5_w2_1 + dL_h6_5_w2_2_1 + dL_h6_5_w2_2_2


  #dh66_dw2
  t1 = dL_v_in[4*h:5*h]
  t2 = dh6_6_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh5_5_l5_in[:,3*h:4*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh4_4_l4_in[:,2*h:3*h]*t3
  t4 = np.sum(t4, axis=0) / BATCH_SIZE

  t5 = dh3_3_l3_in[:,h:2*h]*t4
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  dL_h6_6_w2 = np.zeros(dh2_2_w2.shape)
  dL_h6_6_w2[:,:] = dh2_2_w2[:,:] * t5

  dL_w2 = dL_h6_3_w2 + dL_h6_4_w2 +\
          dL_h6_5_w2 + dL_h6_6_w2

  #dL_b2
  dL_b2 = dL_v_in[h:2*h]*np.sum(dh6_3_l6_in[:,h:2*h],axis=0) +\
          dL_v_in[2*h:3*h]*( (np.sum(dh6_4_l6_in[:,2*h:3*h],axis=0)*\
                               np.sum(dh3_3_l3_in[:,h:2*h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,3*h:4*h], axis=0)*\
                               np.sum(dh4_3_l4_in[:,h:2*h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0)*\
                               np.sum(dh5_3_l5_in[:,h:2*h],axis=0))) +\
          dL_v_in[3*h:4*h]* ( (np.sum(dh6_5_l6_in[:,3*h:4*h],axis=0)*\
                                np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0)*\
                                  np.sum(dh3_3_l3_in[:,h:2*h],axis=0)) +\
                              (np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0)*\
                                 ( (np.sum(dh5_4_l5_in[:,2*h:3*h],axis=0)*\
                                     np.sum(dh3_3_l3_in[:,h:2*h],axis=0))+\
                                   (np.sum(dh5_4_l5_in[:,3*h:4*h],axis=0)*\
                                     np.sum(dh4_3_l4_in[:,h:2*h],axis=0))))) +\
          dL_v_in[4*h:5*h]* (np.sum(dh6_6_l6_in[:,4*h:5*h],axis=0)*\
                             np.sum(dh5_5_l5_in[:,3*h:4*h],axis=0)*\
                             np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0)*\
                             np.sum(dh3_3_l3_in[:,h:2*h],axis=0))
  dL_b2 /= BATCH_SIZE

  #W1
  #dh62_dw1
  t1 =dL_v_in[0:h]
  t2 = dh6_2_l6_in[:,0:h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  dL_h6_2_w1 = np.zeros(dh1_1_w1.shape)
  dL_h6_2_w1[:,:] = dh1_1_w1[:,:]*t2

  #dh63_dw1
  t1 =dL_v_in[h:2*h]
  t2 = dh6_3_l6_in[:,h:2*h]*t1
  t2 = np.sum(t2, axis=0) / BATCH_SIZE

  t3 = dh2_2_l2_in[:,:]*t2
  t3 = np.sum(t3,axis=0) / BATCH_SIZE

  dL_h6_3_w1_1 = np.zeros(dh1_1_w1.shape)
  dL_h6_3_w1_1[:,:] = dh1_1_w1[:,:] * t3

  t4 = dh6_3_l6_in[:,2*h:3*h]*t1
  t4 = np.sum(t4, axis=0) / BATCH_SIZE
  t5 = dh3_2_l3_in[:,0:h]*t4
  t5 = np.sum(t5,axis=0) / BATCH_SIZE

  dL_h6_3_w1_2 = np.zeros(dh1_1_w1.shape)
  dL_h6_3_w1_2[:,:] = dh1_1_w1[:,:] * t5

  t6 = dh6_3_l6_in[:,3*h:4*h]*t1
  t6 = np.sum(t6, axis=0) / BATCH_SIZE
  t7 = dh4_2_l4_in[:,0:h]*t6
  t7 = np.sum(t6,axis=0) / BATCH_SIZE

  dL_h6_3_w1_3 = np.zeros(dh1_1_w1.shape)
  dL_h6_3_w1_3[:,:] = dh1_1_w1[:,:] * t7


  t8 = dh6_3_l6_in[:,4*h:5*h]*t1
  t8 = np.sum(t8, axis=0) / BATCH_SIZE
  t9 = dh5_2_l5_in[:,0:h]*t8
  t9 = np.sum(t9,axis=0) / BATCH_SIZE

  dL_h6_3_w1_4 = np.zeros(dh1_1_w1.shape)
  dL_h6_3_w1_4[:,:] = dh1_1_w1[:,:] * t9


  dL_h6_3_w1 = dL_h6_3_w1_1 + dL_h6_3_w1_2 +\
               dL_h6_3_w1_3 + dL_h6_3_w1_4

  #dh64_dw1
  t1 =dL_v_in[2*h:3*h]
  t2 = dh6_4_l6_in[:,2*h:3*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh3_3_l3_in[:,h:2*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh2_2_l2_in[:,:]*t3
  t4 = np.sum(t4, axis=0) /BATCH_SIZE

  dL_h6_4_w1_1 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_1[:,:] = dh1_1_w1[:,:] * t4

  t5 = dh6_4_l6_in[:,3*h:4*h]*t1
  t5 = np.sum(t5, axis=0) /BATCH_SIZE

  t6 = dh4_3_l4_in[:,h:2*h]*t5
  t6 = np.sum(t6, axis=0) / BATCH_SIZE

  t7 = dh2_2_l2_in[:,:]*t6
  t7 = np.sum(t7, axis=0) /BATCH_SIZE

  dL_h6_4_w1_2 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_2[:,:] = dh1_1_w1[:,:] * t7


  t8 = dh4_3_l4_in[:,2*h:3*h]*t5
  t8 = np.sum(t8, axis=0) / BATCH_SIZE

  t9 = dh3_2_l3_in[:,0:h]*t8
  t9 = np.sum(t9, axis=0) /BATCH_SIZE

  dL_h6_4_w1_3 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_3[:,:] = dh1_1_w1[:,:] * t9


  t10 = dh6_4_l6_in[:,4*h:5*h]*t1
  t10 = np.sum(t10, axis=0) /BATCH_SIZE

  t11 = dh5_3_l5_in[:,h:2*h]*t10
  t11 = np.sum(t11, axis=0) / BATCH_SIZE

  t12 = dh2_2_l2_in[:,:]*t11
  t12 = np.sum(t12, axis=0) /BATCH_SIZE

  dL_h6_4_w1_4 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_4[:,:] = dh1_1_w1[:,:] * t12

  t13 = dh5_3_l5_in[:,2*h:3*h]*t10
  t13 = np.sum(t13, axis=0) / BATCH_SIZE

  t14 = dh3_2_l3_in[:,0:h]*t13
  t14 = np.sum(t14, axis=0) /BATCH_SIZE

  dL_h6_4_w1_5 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_5[:,:] = dh1_1_w1[:,:] * t14


  t15 = dh5_3_l5_in[:,3*h:4*h]*t10
  t15 = np.sum(t15, axis=0) / BATCH_SIZE

  t16 = dh4_2_l4_in[:,0:h]*t15
  t16 = np.sum(t16, axis=0) /BATCH_SIZE

  dL_h6_4_w1_6 = np.zeros(dh1_1_w1.shape)
  dL_h6_4_w1_6[:,:] = dh1_1_w1[:,:] * t16

  dL_h6_4_w1 = dL_h6_4_w1_1 + dL_h6_4_w1_2 +\
               dL_h6_4_w1_3 + dL_h6_4_w1_4 +\
               dL_h6_4_w1_5 + dL_h6_4_w1_6

  #dh65_dw1
  t1 = dL_v_in[3*h:4*h]
  t2 = dh6_5_l6_in[:,3*h:4*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh4_4_l4_in[:,2*h:3*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh3_3_l3_in[:,h:2*h]*t3
  t4 = np.sum(t4, axis=0) / BATCH_SIZE

  t5 = dh2_2_l2_in[:,:]*t4
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  dL_h6_5_w1_1 = np.zeros(dh1_1_w1.shape)
  dL_h6_5_w1_1[:,:] = dh1_1_w1[:,:] * t5

  t6 = dh6_5_l6_in[:,4*h:5*h]*t1
  t6 = np.sum(t6, axis=0) / BATCH_SIZE

  t7 = dh5_4_l5_in[:,2*h:3*h]*t6
  t7 = np.sum(t7, axis=0) / BATCH_SIZE

  t8 = dh3_3_l3_in[:,h:2*h]*t7
  t8 = np.sum(t8, axis=0) / BATCH_SIZE

  t9 = dh2_2_l2_in[:,:]*t8
  t9 = np.sum(t9, axis=0) / BATCH_SIZE

  dL_h6_5_w1_2 = np.zeros(dh1_1_w1.shape)
  dL_h6_5_w1_2 = dh1_1_w1[:, :]*t9

  t10 = dh5_4_l5_in[:,3*h:4*h]*t6
  t10 = np.sum(t10, axis=0) / BATCH_SIZE

  t11 = dh4_3_l4_in[:,h:2*h]*t10
  t11 = np.sum(t11, axis=0) / BATCH_SIZE

  t12 = dh2_2_l2_in[:,:]*t11
  t12 = np.sum(t12, axis=0) / BATCH_SIZE

  dL_h6_5_w1_3 = np.zeros(dh1_1_w1.shape)
  dL_h6_5_w1_3 = dh1_1_w1[:, :]*t12

  t13 = dh4_3_l4_in[:,2*h:3*h]*t10
  t13 = np.sum(t13, axis=0) / BATCH_SIZE

  t14 = dh3_2_l3_in[:,0:h]*t13
  t14 = np.sum(t14, axis=0) / BATCH_SIZE

  dL_h6_5_w1_4 = np.zeros(dh1_1_w1.shape)
  dL_h6_5_w1_4 = dh1_1_w1[:, :]*t14

  dL_h6_5_w1 = dL_h6_5_w1_1 + dL_h6_5_w1_2 +\
               dL_h6_5_w1_3 + dL_h6_5_w1_4

  #dh66_dw1
  t1 = dL_v_in[4*h:5*h]
  t2 = dh6_6_l6_in[:,4*h:5*h]*t1
  t2 = np.sum(t2, axis=0) /BATCH_SIZE

  t3 = dh5_5_l5_in[:,3*h:4*h]*t2
  t3 = np.sum(t3, axis=0) / BATCH_SIZE

  t4 = dh4_4_l4_in[:,2*h:3*h]*t3
  t4 = np.sum(t4, axis=0) / BATCH_SIZE

  t5 = dh3_3_l3_in[:,h:2*h]*t4
  t5 = np.sum(t5, axis=0) / BATCH_SIZE

  t6 = dh2_2_l2_in[:,:]*t5
  t6 = np.sum(t6, axis=0) / BATCH_SIZE

  dL_h6_6_w1 = np.zeros(dh1_1_w1.shape)
  dL_h6_6_w1[:,:] = dh1_1_w1[:,:] * t6

  dL_w1 = dL_h6_2_w1 + dL_h6_3_w1 + dL_h6_4_w1 +\
          dL_h6_5_w1 + dL_h6_6_w1


  #dL_b1
  dL_b1 = dL_v_in[0:h]*np.sum(dh6_2_l6_in[:,0:h],axis=0) +\
          dL_v_in[h:2*h]*( (np.sum(dh6_3_l6_in[:,h:2*h],axis=0)*\
                             np.sum(dh2_2_l2_in[:,0:h],axis=0))+\
                           (np.sum(dh6_3_l6_in[:,2*h:3*h],axis=0)*\
                             np.sum(dh3_2_l3_in[:,0:h],axis=0))+\
                           (np.sum(dh6_3_l6_in[:,3*h:4*h],axis=0)*\
                             np.sum(dh4_2_l4_in[:,0:h],axis=0))+\
                           (np.sum(dh6_3_l6_in[:,4*h:5*h],axis=0)*\
                             np.sum(dh5_2_l5_in[:,0:h],axis=0))) +\
          dL_v_in[2*h:3*h]*( (np.sum(dh6_4_l6_in[:,2*h:3*h],axis=0)*\
                                np.sum(dh3_3_l3_in[:,h:2*h],axis=0)*\
                                  np.sum(dh2_2_l2_in[:,0:h],axis=0)) +\
                             (np.sum(dh6_4_l6_in[:,3*h:4*h],axis=0)*\
                                 ( (np.sum(dh4_3_l4_in[:,h:2*h],axis=0)*\
                                      np.sum(dh2_2_l2_in[:,0:h],axis=0)) +\
                                   (np.sum(dh4_3_l4_in[:,2*h:3*h],axis=0)*\
                                      np.sum(dh3_2_l3_in[:,0:h],axis=0)))) +\
                             (np.sum(dh6_4_l6_in[:,4*h:5*h],axis=0)*\
                                 ( (np.sum(dh4_3_l4_in[:,h:2*h],axis=0)*\
                                      np.sum(dh2_2_l2_in[:,0:h],axis=0)) +\
                                   (np.sum(dh4_3_l4_in[:,2*h:3*h],axis=0)*\
                                      np.sum(dh3_2_l3_in[:,0:h],axis=0)) +\
                                   (np.sum(dh5_3_l5_in[:,3*h:4*h],axis=0)*\
                                      np.sum(dh4_2_l4_in[:,0:h],axis=0))))) +\
          dL_v_in[3*h:4*h]*( (np.sum(dh6_5_l6_in[:,3*h:4*h],axis=0)*\
                                np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0)*\
                                   np.sum(dh3_3_l3_in[:,h:2*h],axis=0)*\
                                      np.sum(dh2_2_l2_in[:,0:h],axis=0))+\
                             (np.sum(dh6_5_l6_in[:,4*h:5*h],axis=0)*\
                                ( (np.sum(dh5_4_l5_in[:,2*h:3*h],axis=0)*\
                                     np.sum(dh3_3_l3_in[:,h:2*h],axis=0)*\
                                        np.sum(dh2_2_l2_in[:,0:h],axis=0)) +\
                                  (np.sum(dh5_4_l5_in[:,3*h:4*h],axis=0)*\
                                     ( (np.sum(dh4_3_l4_in[:,h:2*h],axis=0)*\
                                          np.sum(dh2_2_l2_in[:,0:h],axis=0))+\
                                       (np.sum(dh4_3_l4_in[:,2*h:3*h],axis=0)*\
                                          np.sum(dh3_2_l3_in[:,0:h],axis=0))))))) +\
         dL_v_in[4*h:5*h]*(np.sum(dh6_6_l6_in[:,4*h:5*h],axis=0)*\
                             np.sum(dh5_5_l5_in[:,3*h:4*h],axis=0)*\
                               np.sum(dh4_4_l4_in[:,2*h:3*h],axis=0)*\
                                 np.sum(dh3_3_l3_in[:,h:2*h],axis=0)*\
                                   np.sum(dh2_2_l2_in[:,0:h],axis=0))

  dL_b1 /= BATCH_SIZE




  #Add regularization gradient as well
  dL_w1+= (reg*w1)
  dL_b1+=  (reg*b1)
  dL_w2+=  (reg*w2)
  dL_b2+=  (reg*b2)
  dL_w3+=  (reg*w3)
  dL_b3+=  (reg*b3)
  dL_w4+=  (reg*w4)
  dL_b4+=  (reg*b4)
  dL_w5+=  (reg*w5)
  dL_b5+=  (reg*b5)
  dL_w6+=  (reg*w6)
  dL_b6+=  (reg*b6)
  dL_v_board+=(reg*v_board)
  gradients = ( dL_w1, dL_b1, dL_w2, dL_b2, dL_w3, dL_b3,\
               dL_w4, dL_b4, dL_w5, dL_b5, dL_w6, dL_b6,\
               dL_v_board)
  return gradients

def update_params(params, gradients, v, lr, mu):
  """Update parameters with the provided gradients

  Args:
    params(tuple): tuple of params to be udpated
    gradients(tuple): tuple of gradients w.r.t loss
    v(list): list of velocity matrices
    lr(float): learning rate
    mu(float): momentum 

  Output:
     v(list): tuple of updated velocities
    params_updated(tuple): tuple of updated params
  """

  updated_params = list()  
  
  for idx, param in enumerate(params):
    v_prev = v[idx]
    v[idx] = mu*v[idx] - (lr * gradients[idx])
    updated_params.append(param + (-mu*v_prev) + ((1+mu)*v[idx])  )

  return v, tuple(updated_params)

def save_params(params,file_path):
  """Save network parameters for future use
  Args: 
    params(tuple): tuple of network parameters to be saved
    file_path(str): str path where to save the params

  """
  assert(os.path.dirname(file_path)), 'Invalid directory provided to save file'
  assert(os.access(os.path.dirname(file_path), os.W_OK)), 'Need write permissions to parent dir'

  with open(file_path, 'w') as f:
       pickle.dump(params,f)



def do_validation(params, hidden_size, mini_batch, labels):
  """Do a validation check on the mini_batch
  Args: 
   params(tuple): tuple of params for the network
   hidden_size(int): output dimension of each hidden layer
   mini_batch(np array):validation images
   lables(np array): corresponding validation labels

  """
  logit, pb = forward(params, mini_batch, hidden_size)
  f, norm, probs = softmax(logit)
  pred = np.argmax(probs, axis=0)
  acc = np.sum(np.int16(pred == labels))/float(np.shape(pred)[0])
  print ('Validation Accuracy = {}'.format(acc))


def main():
  """train and eval the network"""

  #Get input data
  mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=False)
  train = mnist_data.train
  val = mnist_data.validation
  train_images = train.images
  train_labels = train.labels
  val_images = val.images
  val_labels = val.labels

  #Set network size
  input_size = len(train_images[0])
  hidden_size = 500

  #Training iterations and epochs
  iterations = int(len(train_images)/BATCH_SIZE)
  eval_iter = int(len(train_images)/len(val_images))
  epochs = 4

  #Nestrov momentum update
  initial_lr= 0.0001
  terminal_lr = 0.001
  step_size =0.0001
  momentum = 0.9
  #Regularization strength
  reg = 0.001

  #Path to save models
  models_path = './models/params_'

  #Get network params
  velocity, params = get_params(input_size,NUM_CLASSES,hidden_size)
  learning_rates = np.arange(initial_lr, terminal_lr, step_size)

  for learning_rate in learning_rates:
    for e in range(epochs):
      val_idx = 0
      #train on mini-batches
      for i in range(iterations):
        #Check if validation is required
        if not (i %eval_iter):
          v_batch = np.array(val_images[val_idx*BATCH_SIZE:(val_idx+1)*BATCH_SIZE])
          v_batch -= np.mean(v_batch, axis=0)
          v_labels = np.array(val_labels[val_idx*BATCH_SIZE:(val_idx+1)*BATCH_SIZE])
          do_validation(params, hidden_size, v_batch, v_labels)
          val_idx+=1
        #j = time.time()
        mini_batch = np.array(train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        mini_batch -= np.mean(mini_batch, axis=0)
        labels = np.array(train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        #print ('Prepared input data in {}'.format(time.time() - j))
        #t = time.time()
        logit, params_back = forward(params,mini_batch,hidden_size)
        #print ('Completed forward in {}'.format(time.time() - t))
        #t = time.time()
        loss, probs=  softmax_cross_entropy_loss(logit, labels, params, BATCH_SIZE, reg)
        print ('Iter = {}; Epoch = {}; Loss = {}'.format(i, e, loss ))
        #t = time.time()
        df = softmax_cross_entropy_loss_derivative(probs, labels)
        #print ('Evaluated softmax derivative in {}'.format(time.time() -t))
        #t = time.time()
        gradients = evaluate_gradients(params, params_back, df, hidden_size, reg)
        #print ('Evaluated network gradients in {}'.format(time.time() -t))
        # Check numerical gradient
        # If uncommented, set BATCH_SIZE=2 and hidden_size = 5 (i.e. keep them small)
        # And just run for one iteration
        #check_gradients(gradients, params, mini_batch, labels, BATCH_SIZE, hidden_size, forward, softmax_cross_entropy_loss, reg)
        #return
        #t = time.time()
        velocity, params = update_params(params, gradients, velocity, learning_rate, momentum)
        #print ('Updated params in {}'.format(time.time() - t))
        #print ('Total train time for mini-batch was {}'.format(time.time() - j))

    #save model parameters
    save_params(params,models_path + str(learning_rate) + '_' + str(iterations) )

if __name__ == '__main__':
  main()
