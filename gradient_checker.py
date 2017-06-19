#!/usr/bin/python
#Author: Jasjeet Dhaliwal

import numpy as np

def eval_gradient(f, x, param_name):
  """Evaluates numerical gradient of a function

  Args: 
    f(lambda function): function whose gradient will be evaluated
    x(numpy array): parameter to eval gradient for i.e. df/dx
    param_name(str): parameter to check gradient for
  Output:
    dx(numpy array): df/dx, same shape as x
  """

  fx = f(x, param_name)
  dx = np.zeros(x.shape)
  h = 0.00001

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    x_old = x[idx]
    x[idx] = x_old + h
    fx_plus_h = f(x, param_name)
    x[idx] = x_old - h
    fx_minus_h = f(x, param_name)
    x[idx] = x_old
    tmp = fx_plus_h - fx_minus_h
    dx[idx] = (fx_plus_h - fx_minus_h)/(2*h)
    it.iternext()
  return dx


def check_gradients(gradients, params, inputs, labels, batch_size, hidden_size, forward_fn, loss_fn, reg):
  """Checks whether analytical gradient is correct """

  params = list(params)
  
  def f(param, param_name):
    if param_name == 'w1':
      params[0] = param
    elif param_name == 'b1':
      params[1] = param
    elif param_name == 'w2':
      params[2] = param
    elif param_name == 'b2':
      params[3] = param
    elif param_name == 'w3':
      params[4] = param
    elif param_name == 'b3':
      params[5] = param
    elif param_name == 'w4':
      params[6] = param
    elif param_name == 'b4':
      params[7] = param
    elif param_name == 'w5':
      params[8] = param
    elif param_name == 'b5':
      params[9] = param
    elif param_name == 'w6':
      params[10] = param
    elif param_name == 'b6':
      params[11] = param
    elif param_name == 'v_board':
      params[12] = param
     
    logit, params_back = forward_fn(tuple(params), inputs, hidden_size)
    loss, probs = loss_fn(logit, labels, params, batch_size, reg)
    return loss

  param_names = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3',\
                   'w4', 'b4', 'w5', 'b5', 'w6', 'b6', 'v_board']
  gradients = list(gradients)
  for idx, param in enumerate(params):
    print ('Checking gradient for {}'.format(param_names[idx]))
    dparam = eval_gradient(f, param, param_names[idx])
    diff = dparam - gradients[idx]
    print diff[diff > 0.01]
    







   
                    
