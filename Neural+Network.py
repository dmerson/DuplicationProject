
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

def sigmoid(x, use_deriviative=False):  # Note: there is a typo on this line in the video
    if(use_deriviative==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))
 


# In[7]:

#input data
inputs= np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1] ])
#print(np.exp(-inputs))
#print(1 + np.exp(-inputs))
#print(1/1 + np.exp(-inputs))


# In[10]:

#output data
target_answer = np.array([[0],
             [1],
             [1],
             [0] ])


# In[11]:

np.random.seed(1)


# In[17]:

#synapses
weights_betwen_input_and_hidden_layer=   2*np.random.random((3,4)) - 1 # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
weight_between_hidden_layer_and_output =  2*np.random.random((4,1)) - 1 # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
#training step
for j in xrange(60000):
    input_layer=inputs
    hidden_layer_1=sigmoid(np.dot(input_layer,weights_betwen_input_and_hidden_layer))
    output_layer=sigmoid(np.dot(hidden_layer_1,weight_between_hidden_layer_and_output))
    output_layer_error=target_answer-output_layer
    
       
    if ((j % 10000)==0):
        print("Error:",str(np.mean(np.abs(output_layer_error))))
    
    output_layer_delta=output_layer_error * sigmoid(output_layer, True)
     
    hidden_layer_1_error=output_layer_delta.dot(weight_between_hidden_layer_and_output.T)
    
    hidden_layer_1_delta = hidden_layer_1_error * sigmoid(hidden_layer_1, True)
    
    #udpate weights
    weight_between_hidden_layer_and_output += hidden_layer_1.T.dot(output_layer_delta)
    weights_betwen_input_and_hidden_layer += input_layer.T.dot(hidden_layer_1_delta)
print ("Output after training ")
print(output_layer)


# In[ ]:



