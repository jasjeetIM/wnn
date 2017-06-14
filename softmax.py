#!/usr/bin/python
#Author: Jasjeet Dhaliwal
#Date: 6/8/2017

import numpy as np

def softmax(logit):
  """Calculate softmax of input
  Input: 
    logit(numpy matrix): Matrix of size n x num_classes containing scores
  Output: 
    probs(numpy matrix): matrix of size n x num_classes containing probs
  """
  #Shift scores for numerical stability
  max_v = np.max(logit,1)
  f = logit - max_v[:,None]
  z = np.exp(f)
  norm = np.sum(z,1)
  probs = z / norm[:,None]
  
  return f, norm, probs

def softmax_cross_entropy_loss_derivative(probs, label):
  df = probs
  df[np.arange(probs.shape[0]), label] -= 1
  return df.mean(0)

def softmax_cross_entropy_loss(logit, label):
  """Calculates the softmax cross entropy loss 
  Input: 
    logit(numpy matrix): n x num_classes vector of scores
    label(numpy matrix): n x 1 labels 

  Output:
    loss(float): loss scalar
  """

  f, norm, probs = softmax(logit)

  data_loss = -f[np.arange(f.shape[0]), label] + np.log(norm)
  data_loss = data_loss.sum()
  data_loss /= BATCH_SIZE
  reg_loss = 0.5 *\
             (np.square(w1).sum() + np.square(b1).sum() + \
             np.square(w2).sum() + np.square(b2).sum() + \
             np.square(w3).sum() + np.square(b3).sum() + \
             np.square(w4).sum() + np.square(b4).sum() + \
             np.square(w5).sum() + np.square(b5).sum() + \
             np.square(w6).sum() + np.square(b6).sum() + \
             np.square(v_board).sum() )

  loss = data_loss + reg_loss
  return loss, probs
