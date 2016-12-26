
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

def sigmoid(x, use_deriviative=False):  # Note: there is a typo on this line in the video
    if(use_deriviative==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))
 


# In[3]:

#input data
inputs= np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
#print(np.exp(-inputs))
#print(1 + np.exp(-inputs))
#print(1/1 + np.exp(-inputs))


# In[4]:

#output data
target_answer = np.array([[0],
             [1],
             [1],
             [0]])


# In[5]:

np.random.seed(1)


# In[ ]:



 


# In[9]:

#synapses
synapse0=   2*np.random.random((3,4)) - 1 # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
synapse1 =  2*np.random.random((4,1)) - 1 # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
#training step
for j in xrange(60000):
    input_layer=inputs
    hidden_layer_1=sigmoid(np.dot(input_layer,synapse0))
    output_layer=sigmoid(np.dot(hidden_layer_1,synapse1))
    output_layer_error=target_answer-output_layer
    
       
    if ((j % 10000)==0):
        print("Error:",str(np.mean(np.abs(output_layer_error))))
    
    output_layer_delta=output_layer_error * sigmoid(output_layer, True)
     
    hidden_layer_1_error=output_layer_delta.dot(synapse1.T)
    
    hidden_layer_1_delta = hidden_layer_1_error * sigmoid(hidden_layer_1, True)
    
    #udpate weights
    synapse1 += hidden_layer_1.T.dot(output_layer_delta)
    synapse0 += input_layer.T.dot(hidden_layer_1_delta)
print ("Output after training ")
print(output_layer)



# In[7]:

#training step
print 'next one'
#input data
X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
            [0,1,1],
            [1,0,1],
            [1,1,1]])
#output data
y = np.array([[0],
             [1],
             [1],
             [0]])
syn0=   2*np.random.random((3,4)) - 1 # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 =  2*np.random.random((4,1)) - 1 # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
for k in xrange(60000):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(k % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print "Error: " + str(np.mean(np.abs(l2_error)))
        
    l2_delta = l2_error*sigmoid(l2, True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * sigmoid(l1,True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print "Output after training"
print l2


# In[ ]:



