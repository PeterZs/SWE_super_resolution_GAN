#!/usr/bin/python
#modify the original testGAN with using a dataset with velocity(modify the dataset)

#when we swith over between the training mode and the generating test image mode, we need a flag to mark the current mode. Because the tensorflow's static graph design, if there is no such a flag for mark the current situation, the disc will be wrong because of the full connection layer's variable number to be changed.
import os
import sys
import math
import time
import shutil 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from keras import backend as kb
import paramhelpers as ph

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
show all info #1
only show all warning and error #2
only show all error #3
'''

# main parameter 
#------------------------------------------------------
get_model=int(ph.getParam("get_model",0)) #0# state of the pipeline
learning_rate_ori=float(ph.getParam("learning_rate_ori",0.0002))#0.00001 #0.0002
decay_lr=float(ph.getParam("decay_lr",0.05))#0.05
#main file location parameter 
data_path=ph.getParam("data_path",'./SWE_solver/data/') #"./SWE_solver/data/" # include the test & train dir
log_path=ph.getParam("log_path",'./log/')#"./log/" # print and save the log file 
model_path=ph.getParam("model_path",'./model/')#"./model/" # save the model 
test_path=ph.getParam("test_path",'./test/')#"./test/"# save the generator's results on the test data sets which is located in the "./data/test/" directory
k1=float(ph.getParam("k1",0.1))#0.1 #5.0 #weight of l1 term on generator loss
k2=float(ph.getParam("k2",0.1))#0.1 #1.0 #weight of layer_loss term on generator loss
k2_l1=float(ph.getParam("k2_l1",0.00001))#0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=float(ph.getParam("k2_l2",0.00001))#0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=float(ph.getParam("k2_l3",0.00001))#0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=float(ph.getParam("k2_l4",0.00001))#0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=float(ph.getParam("k_tempo",1.0))#1.0 #1.0 # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=int(ph.getParam("epochs",20)) #40
use_velocities=int(ph.getParam("use_velocities",0)) #0
# data file's index of simulation sequence.such as, from 0-15 for training, from 16-19 for training
train_index_begin=int(ph.getParam("train_index_begin",0)) #0
train_index_end=int(ph.getParam("train_index_end",16)) # not be included
test_index_begin=int(ph.getParam("test_index_begin",16))#16
test_index_end=int(ph.getParam("test_index_end",20)) # not be included
frame_num_gen=int(ph.getParam("frame_num_gen",120))
disc_spatial_runs=int(ph.getParam("disc_spatial_runs",1)) #1
disc_tempo_runs=int(ph.getParam("disc_tempo_runs",1)) #1
gen_runs=int(ph.getParam("gen_runs",1)) #1

size_tile_low=int(ph.getParam("size_tile_low",16)) #16
print("size_tile_low: %d "%size_tile_low)
size_sim_low=int(ph.getParam("size_sim_low",64)) #64
print("size_sim_low: %d "%size_sim_low)
test_mode=int(ph.getParam("test_mode",0))
print("test_mode: %d"%test_mode)

gradient_loss_kind=int(ph.getParam("gradient_loss_kind",0))

file_name = open(log_path+"log.txt", 'a')
print >> file_name,"get_model: %d"%get_model
print >> file_name,"learning_rate_ori: %f"%learning_rate_ori
print >> file_name,"decay_lr: %f"%decay_lr
print >> file_name,"data_path: %s"%data_path
print >> file_name,"log_path: %s"%log_path
print >> file_name,"model_path: %s"%model_path
print >> file_name,"test_path: %s"%test_path
print >> file_name,"k1: %f"%k1
print >> file_name,"k2: %f"%k2
print >> file_name,"k2_l1: %f"%k2_l1
print >> file_name,"k2_l2: %f"%k2_l2
print >> file_name,"k2_l3: %f"%k2_l3
print >> file_name,"k2_l4: %f"%k2_l4
print >> file_name,"k_tempo: %f"%k_tempo
print >> file_name,"epochs: %d"%epochs
print >> file_name,"use_velocities: %d"%use_velocities
print >> file_name,"train_index_begin: %d"%train_index_begin
print >> file_name,"train_index_end: %d"%train_index_end
print >> file_name,"test_index_begin: %d"%test_index_begin
print >> file_name,"test_index_end: %d"%test_index_end
print >> file_name,"frame_num_gen: %d"%frame_num_gen
print >> file_name,"disc_spatial_runs: %d"%disc_spatial_runs
print >> file_name,"disc_tempo_runs: %d"%disc_tempo_runs
print >> file_name,"gen_runs: %d"%gen_runs

print >> file_name,"size_tile_low: %d"%size_tile_low
print >> file_name,"size_sim_low: %d"%size_sim_low
print >> file_name,"test_mode: %d"%test_mode
print >> file_name,"gradient_loss_kind: %d"%gradient_loss_kind

file_name.close()
      
#--------------------------------
scale=4
size_tile_high=scale*size_tile_low
size_sim_high=scale*size_sim_low
decay=0.999 #para for batch normal
beta=0.5 # para of adam optimizer
batch_size=16
batch_norm=1
advection=0 # whether use the advection operator to align. Here we firstly ignore this operator 
n_input_generator=size_tile_low**2 #size of low
n_output_generator=size_tile_high**2#size of high
num_input_channels=1
frame_num_per_sim=120
n_t=3
if use_velocities:
  num_input_channels+=2 #for SWE simulation, there are only two channels for velocity 
  n_input_generator*=num_input_channels #n_input_generator is X*Y's Y, X is the batch_size


#main defined functions of the NN structure 
#------------------------------------------------------
unflatten_shape=[]

def weight_variable(shape,name="weight",init_mean=0.0):
  stddev_here=0.04
  if init_mean==1.0:
    stddev_here=0.0
  w=tf.get_variable("weight",shape,initializer=tf.random_normal_initializer(stddev=stddev_here,mean=init_mean))
  print(w)
  return w

def bias_variable(shape,name="bias"):
  bias=tf.get_variable("bias",shape,initializer=tf.constant_initializer(0.1))
  print(bias)
  return bias

def conv2d(x,w,stride=[1]):
  if len(stride)==1:
    stride_here=[1,stride[0],stride[0],1]
  elif len(stride)==2:
    stride_here=[1,stride[0],stride[1],1]
  return tf.nn.conv2d(x,w,strides=stride_here,padding="SAME") # for convolution, only use SAME padding here

def convolutional_layer(input_layer,output_channels,patch_shape,activation_function=tf.nn.tanh,stride=[1],name="conv",batch_norm=False,train=None,reuse=False): # (name for determining the scope of the variable)can be changed
  # train is a placeholder
  with tf.variable_scope(name):
    input_channels=int(input_layer.get_shape()[-1])
    if len(patch_shape)==2:
      w=weight_variable([patch_shape[0],patch_shape[1],input_channels,output_channels],name=name)
      output_layer=conv2d(input_layer,w,stride)
    b=bias_variable([output_channels],name=name)
    # so there are output_channels feature map(convolutional kernal), so output_channels bias number
    output_layer=output_layer+b # for each number of the same feature map added by the counterpart bias 
    
    if batch_norm:      
      output_layer=tf.contrib.layers.batch_norm(output_layer,decay=decay,scale=True,scope=tf.get_variable_scope(),reuse=reuse,fused=False,is_training=train)
    linear_layer=output_layer
    if activation_function:
      output_layer=activation_function(output_layer) #activation_function is another name in python , is a per number operator
    return  output_layer,linear_layer

residual_block_id=0

def residual_block(input_layer,output_channels_a,output_channels_b,reuse,batch_norm,filter_size=3):
  '''
  def convolutional_layer(input_layer,output_channels,patch_shape,activation_function=tf.nn.tanh,stride=[1],name="conv",batch_norm=False,train=None,reuse=False)
  '''
  global residual_block_id
  filter_a =[filter_size,filter_size]
  filter_shortcut_connection=[1,1]
  gc1,_=convolutional_layer(input_layer,output_channels_a,filter_a,tf.nn.leaky_relu,stride=[1],name="gcA%d"%residual_block_id,batch_norm=batch_norm,train=train,reuse=reuse)
  # for the residual block's second convolutional layer, before the adding of the shortcut connection, residual block gets linear layer without relu activation function
  gc2,_=convolutional_layer(gc1,output_channels_b,filter_a,None,stride=[1],name="gcB%d"%residual_block_id,batch_norm=batch_norm,train=train,reuse=reuse)
  gs1,_=convolutional_layer(input_layer,output_channels_b,filter_shortcut_connection,None,stride=[1],name="gs%d"%residual_block_id,batch_norm=batch_norm,train=train,reuse=reuse)
  res_layer=tf.nn.leaky_relu(tf.add(gc2,gs1))
  residual_block_id+=1
  return res_layer

#In the paper, it says that they use the NN interpolation to resize the image, like 1234 into 11223344,but the code uses the keras's resize_images() API ,this API lets the 1234 into 10203040. In these both kinds of inputs, the trained convolution kernal will be different obviously. And the original max_depool API is not right. The resize operator is not the max depool operator.
def interpolation_layer(input_layer,length_factor=2,width_factor=2):
  if len(input_layer.get_shape())==4:
    input_layer=kb.resize_images(input_layer,length_factor,width_factor,'channels_last')
    return input_layer
  
#normally, before fully_connected_layer, we need to flatten the tensor of input_layer into a X*Y matrix form  X is the 0 dimension (batch size)
def fully_connected_layer(input_layer,num_hidden,activation_function,name="full"):
  with tf.variable_scope(name):
    num_input=int(input_layer.get_shape()[1])
    w=weight_variable([num_input,num_hidden],name=name)
    b=bias_variable([num_hidden],name=name)
    output_layer=tf.matmul(input_layer,w)+b

    if activation_function:
      output_layer=activation_function(output_layer)
    return output_layer


def max_pool(input_layer,window_size=[2],window_stride=[2]):
  if len(input_layer.get_shape())==4:
    input_layer=tf.nn.max_pool(input_layer,ksize=[1,window_size[0],window_size[0],1],strides=[1,window_stride[0],window_stride[0],1],padding="VALID")
  return input_layer

def avg_pool(input_layer,window_size=[2],window_stride=[2]):
  if len(input_layer.get_shape())==4:
    input_layer=tf.nn.avg_pool(input_layer,ksize=[1,window_size[0],window_size[0],1],strides=[1,window_stride[0],window_stride[0],1],padding="VALID")
  return input_layer


def flatten(input_layer):
  unflatten_shape_here=input_layer.get_shape()
  global unflatten_shape
  unflatten_shape.append(unflatten_shape_here)

  flat_size=int(unflatten_shape_here[1])*int(unflatten_shape_here[2])*int(unflatten_shape_here[3])
  input_layer=tf.reshape(input_layer,[-1,flat_size])
  return input_layer,flat_size

#here logic is wrong
def unflatten(input_layer):
  global unflatten_shape
  unflatten_shape_here=unflatten_shape.pop()
  input_layer=tf.reshape(input_layer,unflatten_shape_here)
  return input_layer

#input_data: X*Y may be one or four channels
def generator(input_data,reuse=False,batch_norm=False, train=None):
  global residual_block_id
  with tf.variable_scope("generator",reuse=reuse) as scope:
    input_data=tf.reshape(input_data,shape=[-1,size_tile_low,size_tile_low,num_input_channels]) #N H W C
    residual_block_id=0
    inter1=interpolation_layer(input_data,2,2)
    inter2=interpolation_layer(inter1,2,2)
    #def residual_block(input_layer,output_channels_a,output_channels_b,reuse,batch_norm,filter_size=3):
    res1=residual_block(inter2,num_input_channels*2,num_input_channels*8,reuse,batch_norm,filter_size=5)
    res2=residual_block(res1,128,128,reuse,batch_norm,filter_size=5)
    res3=residual_block(res2,32,8,reuse,batch_norm,filter_size=5)
    res4=residual_block(res3,2,1,reuse,False,filter_size=5)
    #for the input of disc, the output of the generator will be reshaped
    output_gen=tf.reshape(res4,shape=[-1,n_output_generator])
    output_inter=tf.reshape(inter2,shape=[-1,n_output_generator,num_input_channels])

    #maybe we need to count the number of the weight_variable and bias_variable, maybe not
    return output_gen,output_inter #we need to check the difference between the output_gen and the inter2

#input_low: X*Y one channel, input_high: X*Y one channel
# flag: different from testGAN_0.py
def disc_spatial(input_low,input_high,reuse=False,batch_norm=False,train=None):
  #input_low: low res reference input, same as generator input(condition)
  #input_high: real or generated high res input to classify
  # reuse: whether variable reuse
  # batch_norm: bool
  # train : if  batch_norm , tf bool placeholder expresses whether training the batch weight

  with tf.variable_scope("disc_spatial",reuse=reuse):
    shape=tf.shape(input_low)
    
    #tf.slice() return the sliced tensor
    #here we use a condition just using density.

    #better choice compaced to {reshape && [:,:,:,0]}
    input_low=tf.slice(input_low,[0,0],[shape[0],int(n_input_generator/num_input_channels)]) # because the input_low may have 2 channels for velocity, so we need to slice the first channel which is not velocity component
    #input_low=tf.reshape(input_low,shape=[-1,size_tile_low,size_tile_low,num_input_channels])
    #input_low=input_low[:,:,:,0]
    input_low=tf.reshape(input_low,shape=[-1,size_tile_low,size_tile_low,1])
    
    #here we use a condition using density and velocity.
    #input_low=tf.reshape(input_low,shape=[-1,size_tile_low,size_tile_low,num_input_channels])

    input_low=interpolation_layer(input_low,scale,scale)
    input_high=tf.reshape(input_high,shape=[-1,size_tile_high,size_tile_high,1])
    filter=[4,4]
    stride=[2]
    print("scaled input_low: {}".format(input_low.get_shape()))
    print("input_high: {}".format(input_high.get_shape()))

    #merge input_low & input_high to [-1,size_tile_high,size_tile_high,2]
    #tf.concat() return the concated tensor 
    input_merge=tf.concat([input_low,input_high],axis=-1)
    #convolutional_layer(input_layer,output_channels,patch_shape,activation_function=tf.nn.tanh,stride=[1],name="conv",batch_norm=False,train=None,reuse=False)
    con1,_=convolutional_layer(input_merge,32,filter,tf.nn.leaky_relu,stride=stride,name="con1",reuse=reuse,batch_norm=False,train=False) # the first convolutional_layer does not use the batch_norm
    con2,_=convolutional_layer(con1,64,filter,tf.nn.leaky_relu,stride=stride,name="con2",reuse=reuse,batch_norm=batch_norm,train=train)
    con3,_=convolutional_layer(con2,128,filter,tf.nn.leaky_relu,stride=stride,name="con3",reuse=reuse,batch_norm=batch_norm,train=train)

    #here in tempoGAN, we use the stride=[1] for the last layer!!!
    con4,_=convolutional_layer(con3,256,filter,tf.nn.leaky_relu,stride=[1],name="con4",reuse=reuse,batch_norm=batch_norm,train=train)
    
    #con4,_=convolutional_layer(con3,256,filter,tf.nn.leaky_relu,stride=stride,name="con4",reuse=reuse,batch_norm=batch_norm,train=train)
    
    flattened_con4,_=flatten(con4)
    #fully_connected_layer(input_layer,num_hidden,activation_function,name="full")
    output=fully_connected_layer(flattened_con4,1,None,name="full") # here we do not use sigmoid, because loss after here will use sigmoid function , no nonlinear here
    low_output=tf.reshape(input_low,shape=[-1,n_output_generator])
    return output,con1,con2,con3,con4,low_output   

#input_high: X*Y*3 (3frame )
def disc_tempo(input_high,num_tempo_channels=3,reuse=False,batch_norm=False,train=None):
  with tf.variable_scope("disc_tempo",reuse=reuse):
    input_high=tf.reshape(input_high,shape=[-1,size_tile_high,size_tile_high,num_tempo_channels])
    filter=[4,4]
    stride=[2]

    con1,_=convolutional_layer(input_high,32,filter,tf.nn.leaky_relu,stride=stride,name="con1",reuse=reuse,batch_norm=False,train=False) # the first convolutional_layer does not use the batch_norm
    con2,_=convolutional_layer(con1,64,filter,tf.nn.leaky_relu,stride=stride,name="con2",reuse=reuse,batch_norm=batch_norm,train=train)
    con3,_=convolutional_layer(con2,128,filter,tf.nn.leaky_relu,stride=stride,name="con3",reuse=reuse,batch_norm=batch_norm,train=train)

    #here in tempoGAN, we use the stride=[1] for the last layer!!!
    con4,_=convolutional_layer(con3,256,filter,tf.nn.leaky_relu,stride=[1],name="con4",reuse=reuse,batch_norm=batch_norm,train=train)
    #con4,_=convolutional_layer(con3,256,filter,tf.nn.leaky_relu,stride=stride,name="con4",reuse=reuse,batch_norm=batch_norm,train=train)
    flattened_con4,_=flatten(con4)
    #fully_connected_layer(input_layer,num_hidden,activation_function,name="full")
    output=fully_connected_layer(flattened_con4,1,None,name="full") # here we do not use  sigmoid, because loss after here will use sigmoid function, no nonlinear here 
    return output,input_high

#main tensorflow graph definition
#tf.xxx() most of xxx is a operator and if it need input(may be tensor or variable or constant) and it will output(tensor or variable) 
#------------------------------------------------------
device_str='/device:GPU:0'
with  tf.device(device_str):
    #input for generator, set as tensorflow's placeholder
    x=tf.placeholder(tf.float32,shape=[None,n_input_generator])
    #input reference x for condition spatial disc
    x_disc_spatial=tf.placeholder(tf.float32,shape=[None,n_input_generator])
    #input for generator passed into a disc_tempo
    x_t=tf.placeholder(tf.float32,shape=[None,n_input_generator])
    #real input for spatial disc
    y=tf.placeholder(tf.float32,shape=[None,n_output_generator])
    #real input for tempo disc
    y_t=tf.placeholder(tf.float32,shape=[None,n_output_generator])
    print("x: {}".format(x.get_shape()))
    print("y: {}".format(y.get_shape()))
    print("x_t: {}".format(x_t.get_shape()))
    print("y_t: {}".format(y_t.get_shape()))
    k_k1=tf.placeholder(tf.float32)
    k_k2=tf.placeholder(tf.float32)
    k_k_tempo=tf.placeholder(tf.float32)
    train=tf.placeholder(tf.bool)

    lr_global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.polynomial_decay(learning_rate_ori,lr_global_step,epochs//2,learning_rate_ori*decay_lr,power=1.1)

    if(test_mode):
        gen_output,inter_output=generator(x,batch_norm=batch_norm,train=False)
    elif(not test_mode):
        #get gen disc_spatial disc_tempo output
        gen_output,inter_output=generator(x,batch_norm=batch_norm,train=train) # x is right
        # disc_spatial(input_low,input_high,reuse=False,batch_norm=False,train=None):
        #x x_disc_spatial may be not the same !!!!
        disc_spatial_output_real,disc_spatial_con1_real,disc_spatial_con2_real,disc_spatial_con3_real,disc_spatial_con4_real,_=disc_spatial(x_disc_spatial,y,batch_norm=batch_norm,train=train) # x_disc_spatial is right y is right
        # need reuse= True !!!
        disc_spatial_output_fake,disc_spatial_con1_fake,disc_spatial_con2_fake,disc_spatial_con3_fake,disc_spatial_con4_fake,low_output=disc_spatial(x_disc_spatial,gen_output,reuse=True,batch_norm=batch_norm,train=train) # x_disc_spatial is right gen_output is right
        #for batch_norm, get each's collection
        #here because the disc_spatial_loss's passed batch and the gen_loss's passed batch are the same partly, so they have the same batch update_ops collection!!!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gen_update_ops = update_ops[:] # for generator with disc_tempo
        disc_spatial_update_ops=update_ops[:]

        #reuse generator() for passing into disc_tempo
        gen_t,inter_output_t=generator(x_t,reuse=True,batch_norm=batch_norm,train=train)

        # different from testGAN_0.py
        '''
        gen_t=tf.reshape(gen_t,shape=[n_t,-1,size_tile_high*size_tile_high])
        gen_t=tf.transpose(gen_t,perm=[1,2,0])
        gen_t=tf.reshape(gen_t,shape=[-1,size_tile_high*size_tile_high,n_t])

        y_t=tf.reshape(y_t,shape=[n_t,-1,size_tile_high*size_tile_high])
        y_t=tf.transpose(y_t,perm=[1,2,0])
        y_t=tf.reshape(y_t,shape=[-1,size_tile_high*size_tile_high,n_t])
        '''        
        
        gen_t=tf.reshape(gen_t,shape=[-1,n_t,size_tile_high*size_tile_high])
        gen_t=tf.transpose(gen_t,perm=[0,2,1])

        y_t=tf.reshape(y_t,shape=[-1,n_t,size_tile_high*size_tile_high])
        y_t=tf.transpose(y_t,perm=[0,2,1])
        
        #  print(y_t)

        #def disc_tempo(input_high,num_tempo_channels=3,reuse=False,batch_norm=False,train=None)
        disc_tempo_output_fake,output_high_fake=disc_tempo(gen_t,num_tempo_channels=n_t,batch_norm=batch_norm,train=train)

        disc_tempo_update_ops=[]
        #extra update_ops collection for gen_loss
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        for update_op in update_ops:
            if ("generator" in update_op.name) and (not(update_op in gen_update_ops)):
                gen_update_ops.append(update_op)
                disc_tempo_update_ops.append(update_op)
            if("disc_tempo" in update_op.name):
                gen_update_ops.append(update_op)
                disc_tempo_update_ops.append(update_op)

        #need reuse= True !!!
        disc_tempo_output_real,output_high_real=disc_tempo(y_t,num_tempo_channels=n_t,reuse=True,batch_norm=batch_norm,train=train)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        for update_op in update_ops:
            if("disc_tempo" in update_op.name) and (not(update_op in disc_tempo_update_ops)):
                disc_tempo_update_ops.append(update_op)

        #get loss funtion
        # two parts of loss of disc_spatial
        # 1.cross_entropy loss of disc_spatial for real data
        disc_spatial_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_spatial_output_real,labels=tf.ones_like(disc_spatial_output_real)))
        # 2.cross_entropy loss of disc_spatial for fake data
        disc_spatial_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_spatial_output_fake,labels=tf.zeros_like(disc_spatial_output_fake)))
        disc_spatial_loss=disc_spatial_loss_fake+disc_spatial_loss_real

        # four parts of loss of generator
        # 1.loss_layer_disc
        loss_layer_disc=k2_l1*tf.reduce_mean(tf.nn.l2_loss(disc_spatial_con1_fake-disc_spatial_con1_real))+k2_l2*tf.reduce_mean(tf.nn.l2_loss(disc_spatial_con2_fake-disc_spatial_con2_real))+k2_l3*tf.reduce_mean(tf.nn.l2_loss(disc_spatial_con3_fake-disc_spatial_con3_real))+k2_l4*tf.reduce_mean(tf.nn.l2_loss(disc_spatial_con4_fake-disc_spatial_con4_real))
        # 2.gen_spatial_loss
        gen_spatial_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_spatial_output_fake,labels=tf.ones_like(disc_spatial_output_fake)))
        # 3.l1_loss
        gen_l1_loss=tf.reduce_mean(tf.abs(gen_output-y))
        # 4.gen_tempo_loss
        gen_tempo_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_tempo_output_fake,labels=tf.ones_like(disc_spatial_output_fake)))
        gen_loss=gen_spatial_loss+k_k2*loss_layer_disc+k_k1*gen_l1_loss+k_k_tempo*gen_tempo_loss

        
        gen_l2_loss=tf.reduce_mean(tf.nn.l2_loss(gen_output-y))

        gen_output_reshape=tf.reshape(gen_output,shape=[batch_size,size_tile_high,size_tile_high])
        y_reshape=tf.reshape(y,shape=[batch_size,size_tile_high,size_tile_high])

        gen_gradient_l2_loss=tf.constant(0.0,shape=[1])
        gen_gradient_loss=tf.constant(0.0,shape=[1])
        gen_gradient_l2_loss_no_abs=tf.constant(0.0,shape=[1])
        gen_gradient_loss_no_abs=tf.constant(0.0,shape=[1])
        
        shape=gen_output_reshape.get_shape()
        for a in range(1,int(shape[1])):
          gen_gradient_l2_loss=gen_gradient_l2_loss+tf.reduce_mean(tf.nn.l2_loss(tf.abs(tf.subtract(gen_output_reshape[:,a,:],gen_output_reshape[:,a-1,:]))-tf.abs(tf.subtract(y_reshape[:,a,:],y_reshape[:,a-1,:]))))
          gen_gradient_loss=gen_gradient_loss+tf.reduce_mean(tf.abs(tf.abs(tf.subtract(gen_output_reshape[:,a,:],gen_output_reshape[:,a-1,:]))-tf.abs(tf.subtract(y_reshape[:,a,:],y_reshape[:,a-1,:]))))
          gen_gradient_l2_loss_no_abs=gen_gradient_l2_loss_no_abs+tf.reduce_mean(tf.nn.l2_loss(tf.subtract(gen_output_reshape[:,a,:],gen_output_reshape[:,a-1,:])-tf.subtract(y_reshape[:,a,:],y_reshape[:,a-1,:])))
          gen_gradient_loss_no_abs=gen_gradient_loss_no_abs+tf.reduce_mean(tf.abs(tf.subtract(gen_output_reshape[:,a,:],gen_output_reshape[:,a-1,:])-tf.subtract(y_reshape[:,a,:],y_reshape[:,a-1,:])))

        for b in range(1,int(shape[2])):
          gen_gradient_l2_loss=gen_gradient_l2_loss+tf.reduce_mean(tf.nn.l2_loss(tf.abs(tf.subtract(gen_output_reshape[:,:,b],gen_output_reshape[:,:,b-1]))-tf.abs(tf.subtract(y_reshape[:,:,b],y_reshape[:,:,b-1]))))
          gen_gradient_loss=gen_gradient_loss+tf.reduce_mean(tf.abs(tf.abs(tf.subtract(gen_output_reshape[:,:,b],gen_output_reshape[:,:,b-1]))-tf.abs(tf.subtract(y_reshape[:,:,b],y_reshape[:,:,b-1]))))
          gen_gradient_l2_loss_no_abs=gen_gradient_l2_loss_no_abs+tf.reduce_mean(tf.nn.l2_loss(tf.subtract(gen_output_reshape[:,:,b],gen_output_reshape[:,:,b-1])-tf.subtract(y_reshape[:,:,b],y_reshape[:,:,b-1])))
          gen_gradient_loss_no_abs=gen_gradient_loss_no_abs+tf.reduce_mean(tf.abs(tf.subtract(gen_output_reshape[:,:,b],gen_output_reshape[:,:,b-1])-tf.subtract(y_reshape[:,:,b],y_reshape[:,:,b-1])))
            

        size_all=float(int(shape[1]))
        gen_gradient_loss=tf.divide(gen_gradient_loss,size_all)
        gen_gradient_l2_loss=tf.divide(gen_gradient_l2_loss,size_all)
        gen_gradient_loss_no_abs=tf.divide(gen_gradient_loss_no_abs,size_all)
        gen_gradient_l2_loss_no_abs=tf.divide(gen_gradient_l2_loss_no_abs,size_all)

        # test gen_loss is a combination of gen_l2_loss and a kind of gradient based loss

        k_k_gradient=1
        if(gradient_loss_kind==0):
          gen_loss+=gen_l2_loss+k_k_gradient*gen_gradient_loss

        elif(gradient_loss_kind==1):
          gen_loss+=gen_l2_loss+k_k_gradient*gen_gradient_l2_loss

        elif(gradient_loss_kind==2):
          gen_loss+=gen_l2_loss+k_k_gradient*gen_gradient_loss_no_abs

        elif(gradient_loss_kind==3):
          gen_loss+=gen_l2_loss+k_k_gradient*gen_gradient_l2_loss_no_abs

        elif(gradient_loss_kind==4):
          gen_loss+=k_k_gradient*gen_gradient_loss

        elif(gradient_loss_kind==5):
          gen_loss+=k_k_gradient*gen_gradient_l2_loss

        elif(gradient_loss_kind==6):
          gen_loss+=k_k_gradient*gen_gradient_loss_no_abs

        elif(gradient_loss_kind==7):
          gen_loss+=k_k_gradient*gen_gradient_l2_loss_no_abs
                  
        
        

        #loss for disc_tempo has two parts
        # 1.cross_entropy loss of disc_tempo for real data
        disc_tempo_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_tempo_output_real,labels=tf.ones_like(disc_tempo_output_real)))
        # 2.cross_entropy loss of disc_tempo for fake data
        disc_tempo_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_tempo_output_fake,labels=tf.zeros_like(disc_tempo_output_fake)))
        disc_tempo_loss=disc_tempo_loss_fake+disc_tempo_loss_real
        

        #variables used in different optimization steps
        vars_all=tf.trainable_variables()
        gen_variable=[var for var in vars_all if "generator" in var.name]
        disc_spatial_variable=[var for var in vars_all if "disc_spatial" in var.name]
        disc_tempo_variable=[var for var in vars_all if "disc_tempo" in var.name]

        print(gen_variable)
        print(disc_spatial_variable)
        print(disc_tempo_variable)
        
        # beta1 beta one !!!!
        with tf.control_dependencies(disc_tempo_update_ops):
          disc_tempo_optimizer=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(disc_tempo_loss,var_list=disc_tempo_variable)

        with tf.control_dependencies(disc_spatial_update_ops):
          disc_spatial_optimizer=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(disc_spatial_loss,var_list=disc_spatial_variable)

        with tf.control_dependencies(gen_update_ops):
          gen_optimizer=tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(gen_loss,var_list=gen_variable)


        
    #main training or test starting with the session
    #-----------------------------------------------------------------------------
    # create session which allows soft placement and log which device to be used!
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.InteractiveSession(config = config)
    #save model. Here we will save the model graph (meta) and variables' value (ckpt)
    saver = tf.train.Saver()
    writer=tf.summary.FileWriter(model_path+"/graph",sess.graph)
    writer.flush()
    writer.close()
    if get_model==0: # training -> save model -> test
        sess.run(tf.global_variables_initializer())
    elif get_model==1: # restore model variables -> test
      #pass      
      saver.restore(sess,model_path+"model.ckpt")
    else:
      pass
   # saved_model_number=0
    num_per_data=(size_sim_low//size_tile_low)
    print("num_per_data: %d"%num_per_data)
    
    #main tensorflow's data's needed IO function definition. Here includes 1.the get batch function, 2.the generate test result as  image function, 3.the save model function, 4.the save log function
    #-----------------------------------------------------
    #for number data, we prefer Binary storage method
    def get_input(index_now,frame_start,frame_seq_num,high_low):

      if(frame_start+frame_seq_num>frame_num_per_sim):
        frame_start=frame_num_per_sim-frame_seq_num
        
      if(high_low=="high"):
        size_tile_here=size_tile_high
        size_sim_here=size_sim_high
      elif (high_low=="low"):
        size_tile_here=size_tile_low
        size_sim_here=size_sim_low

      data=np.zeros(shape=(frame_seq_num,size_sim_here*size_sim_here))
      for frame_now in range (frame_start,frame_start+frame_seq_num):
        frame_offset=frame_now-frame_start
        data[frame_offset,:]=np.fromfile(data_path+"/"+str(index_now)+"/"+high_low+"_"+str(frame_now)+".bin")
        #np.fromfile return one dimension array
      data=data.reshape((frame_seq_num,size_sim_here,size_sim_here))
      data_final=np.zeros(shape=(frame_seq_num,num_per_data*num_per_data,size_tile_here,size_tile_here))
      for frame_now in range (frame_seq_num):
        for i in range(num_per_data):
          for j in range(num_per_data):
            data_final[frame_now,i*num_per_data+j,:,:]=data[frame_now,i*size_tile_here:i*size_tile_here+size_tile_here,j*size_tile_here:j*size_tile_here+size_tile_here]
      data_final=data_final.transpose((1,0,2,3))
      data_final=data_final.reshape((num_per_data*num_per_data,frame_seq_num,-1))
      return data_final

    def get_input_vx(index_now,frame_start,frame_seq_num,high_low):

      if(frame_start+frame_seq_num>frame_num_per_sim):
        frame_start=frame_num_per_sim-frame_seq_num
        
      if(high_low=="high"):
        size_tile_here=size_tile_high
        size_sim_here=size_sim_high
      elif (high_low=="low"):
        size_tile_here=size_tile_low
        size_sim_here=size_sim_low

      data=np.zeros(shape=(frame_seq_num,size_sim_here*size_sim_here))
      for frame_now in range (frame_start,frame_start+frame_seq_num):
        frame_offset=frame_now-frame_start
        data[frame_offset,:]=np.fromfile(data_path+"/"+str(index_now)+"/"+high_low+"_vx_"+str(frame_now)+".bin")
        #np.fromfile return one dimension array
      data=data.reshape((frame_seq_num,size_sim_here,size_sim_here))
      data_final=np.zeros(shape=(frame_seq_num,num_per_data*num_per_data,size_tile_here,size_tile_here))
      for frame_now in range (frame_seq_num):
        for i in range(num_per_data):
          for j in range(num_per_data):
            data_final[frame_now,i*num_per_data+j,:,:]=data[frame_now,i*size_tile_here:i*size_tile_here+size_tile_here,j*size_tile_here:j*size_tile_here+size_tile_here]
      data_final=data_final.transpose((1,0,2,3))
      data_final=data_final.reshape((num_per_data*num_per_data,frame_seq_num,-1))
      return data_final

    def get_input_vy(index_now,frame_start,frame_seq_num,high_low):

      if(frame_start+frame_seq_num>frame_num_per_sim):
        frame_start=frame_num_per_sim-frame_seq_num
        
      if(high_low=="high"):
        size_tile_here=size_tile_high
        size_sim_here=size_sim_high
      elif (high_low=="low"):
        size_tile_here=size_tile_low
        size_sim_here=size_sim_low

      data=np.zeros(shape=(frame_seq_num,size_sim_here*size_sim_here))
      for frame_now in range (frame_start,frame_start+frame_seq_num):
        frame_offset=frame_now-frame_start
        data[frame_offset,:]=np.fromfile(data_path+"/"+str(index_now)+"/"+high_low+"_vy_"+str(frame_now)+".bin")
        #np.fromfile return one dimension array
      data=data.reshape((frame_seq_num,size_sim_here,size_sim_here))
      data_final=np.zeros(shape=(frame_seq_num,num_per_data*num_per_data,size_tile_here,size_tile_here))
      for frame_now in range (frame_seq_num):
        for i in range(num_per_data):
          for j in range(num_per_data):
            data_final[frame_now,i*num_per_data+j,:,:]=data[frame_now,i*size_tile_here:i*size_tile_here+size_tile_here,j*size_tile_here:j*size_tile_here+size_tile_here]
      data_final=data_final.transpose((1,0,2,3))
      data_final=data_final.reshape((num_per_data*num_per_data,frame_seq_num,-1))
      return data_final

    # output is high resolution image which is real or fake
    def generate_image(index_now,frame_now,image_matrix,kind): #kind can be high_real, high_fake , high_interpolation, low_real
      if(kind=="high_real" or kind=="high_fake" or ("high_interpolation" in kind ) or ("output_high_real" in kind)):
        size_tile_here=size_tile_high
        size_sim_here=size_sim_high
      elif(kind=="low_real" or kind=="low_fake" or ("low_interpolation" in kind ) or ("output_low_real" in kind)):
        size_tile_here=size_tile_low
        size_sim_here=size_sim_low
        
      data=np.zeros(shape=(1,size_sim_here,size_sim_here))
      for i in range(num_per_data):
        for j in range(num_per_data):
          data[0,i*size_tile_here:i*size_tile_here+size_tile_here,j*size_tile_here:j*size_tile_here+size_tile_here]=image_matrix[i*num_per_data+j,:,:]
      fig=plt.figure(figsize=(1,1))
      gs=gridspec.GridSpec(1,1)
      gs.update(wspace=0,hspace=0)
      for i,sample in enumerate(data):
        ax=plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample,cmap="Greys_r")
      final_path=test_path+"/"+str(index_now)+"/"
      if ( not os.path.isdir(final_path)):
        os.mkdir(final_path)      
      plt.savefig(final_path+kind+"_"+str(frame_now)+".png",dpi=size_sim_here)      
      plt.close(fig)
      #  print("final_path :%s" %final_path)

    
    def save_model(model_path=model_path):
      #save model. Here we will save the model graph (meta) and variables' value (ckpt)
      save_path=saver.save(sess,model_path+"model.ckpt")
      print("model saved in path: %s" % save_path)
      
    def save_log(gen_loss_return,disc_tempo_loss_return,disc_spatial_loss_return):
      #      print(gen_loss_return)
      file_name = open(log_path+"log.txt", 'a')
      print >> file_name,"gen_loss: %lf disc_tempo_loss: %lf disc_spatial_loss: %lf"%(gen_loss_return,disc_tempo_loss_return,disc_spatial_loss_return)
      file_name.close()
      
    #main train process
    #------------------------------------------------------
    if(not get_model):
      run_options = None; run_metadata = None
      for epoch in  range(epochs):
        
        #test here!
        batch_xs=get_input(0,62,1,"low")
        batch_ys=get_input(0,62,1,"high")
        batch_xs=batch_xs.reshape(batch_size,size_tile_low,size_tile_low,1)
        if(use_velocities):
          batch_vxs=get_input_vx(0,62,1,"low")
          batch_vys=get_input_vy(0,62,1,"low")
          batch_vxs=batch_vxs.reshape(batch_size,size_tile_low,size_tile_low,1)
          batch_vys=batch_vys.reshape(batch_size,size_tile_low,size_tile_low,1)
          batch_xs=np.concatenate((batch_xs,batch_vxs,batch_vys),-1)
          
        batch_xs=batch_xs.reshape(batch_size,-1)
        #print("batch_xs's shape: {}".format(batch_xs.shape))
          
        dict_train={x:batch_xs,x_disc_spatial:batch_xs,train:False}
        result_gen,result_inter=sess.run([gen_output,inter_output],feed_dict=dict_train,options=run_options,run_metadata=run_metadata)

        #print(result_gen)
        #generate_test_data(index_now,frame_now,image_matrix,real_fake): #function definition
        result_gen=result_gen.reshape(batch_size,size_tile_high,size_tile_high)
        generate_image(999,epoch,result_gen,"high_fake")                   

        mark=max(0,epoch-(epochs//2))
        #for disc_spatial
        for i in range(train_index_begin,train_index_end):
          j=0
          while(j<frame_num_per_sim):
            for runs in range(disc_spatial_runs):
              #def get_input(index_now,frame_start,frame_seq_num,high_low):
              batch_xs=get_input(0,62,1,"low")
              batch_ys=get_input(0,62,1,"high")
              
              batch_xs=batch_xs.reshape(batch_size,size_tile_low,size_tile_low,1)
              batch_ys=batch_ys.reshape(batch_size,-1)

              if(use_velocities):
                batch_vxs=get_input_vx(0,62,1,"low")
                batch_vys=get_input_vy(0,62,1,"low")
                batch_vxs=batch_vxs.reshape(batch_size,size_tile_low,size_tile_low,1)
                batch_vys=batch_vys.reshape(batch_size,size_tile_low,size_tile_low,1)                
                batch_xs=np.concatenate((batch_xs,batch_vxs,batch_vys),-1)

              batch_xs=batch_xs.reshape(batch_size,-1)
              dict_train={x:batch_xs,x_disc_spatial:batch_xs,y:batch_ys,train:True,lr_global_step:mark}
              _,disc_spatial_loss_return=sess.run([disc_spatial_optimizer,disc_spatial_loss],feed_dict=dict_train,options=run_options,run_metadata=run_metadata)
              print("disc_spatial_loss_return: %f"%disc_spatial_loss_return)

            for runs in range(disc_tempo_runs):
              batch_xts=get_input(0,61,3,"low")
              batch_yts=get_input(0,61,3,"high")
              
              batch_xts=batch_xts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)              
              batch_yts=batch_yts.reshape(batch_size,n_t,-1)
              batch_yts=batch_yts.transpose((0,2,1))

              if(use_velocities):
                batch_vxts=get_input_vx(0,61,3,"low")
                batch_vyts=get_input_vy(0,61,3,"low")
                batch_vxts=batch_vxts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)
                batch_vyts=batch_vyts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)
                
                batch_xts=np.concatenate((batch_xts,batch_vxts,batch_vyts),-1)

              batch_xts=batch_xts.reshape(batch_size*n_t,-1)

              #print("batch_yts's shape: {}".format(batch_yts.shape))
              #print("y_t's shape:{}".format(y_t.shape))
              
              dict_train={x_t:batch_xts,y_t:batch_yts,train:True,lr_global_step:mark}
              _,disc_tempo_loss_return,inter_output_t_return,output_high_real_return=sess.run([disc_tempo_optimizer,disc_tempo_loss,inter_output_t,output_high_real],feed_dict=dict_train,options=run_options,run_metadata=run_metadata)

              '''
              print("inter_output_t_return's shape: {}".format(inter_output_t_return.shape))
              inter_output_t_return=inter_output_t_return.reshape(batch_size*n_t,size_tile_high,size_tile_high,num_input_channels)
              for a in range(3):
                for b in range(3):
                  inter_output_t_return_here=inter_output_t_return[a*batch_size:(a+1)*batch_size,:,:,b]                  
                  generate_image(i,j,inter_output_t_return_here,"high_interpolation_gen_t_"+str(a)+"_"+str(b))

              for a in range(3):
                output_high_real_return_here=output_high_real_return[:,:,:,a]
                generate_image(i,j,output_high_real_return_here,"output_high_real_"+str(a))
              '''              
                
              print("disc_tempo_loss_return: %f"%disc_tempo_loss_return)

            for runs in range(gen_runs):
              batch_xs=get_input(0,62,1,"low")
              batch_ys=get_input(0,62,1,"high")              
              batch_xts=get_input(0,61,3,"low")

              batch_xs=batch_xs.reshape(batch_size,size_tile_low,size_tile_low,1)
              batch_ys=batch_ys.reshape(batch_size,-1)
              batch_xts=batch_xts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)

              if(use_velocities):
                batch_vxs=get_input_vx(0,62,1,"low")
                batch_vys=get_input_vy(0,62,1,"low")
                batch_vxs=batch_vxs.reshape(batch_size,size_tile_low,size_tile_low,1)
                batch_vys=batch_vys.reshape(batch_size,size_tile_low,size_tile_low,1)
                batch_xs=np.concatenate((batch_xs,batch_vxs,batch_vys),-1)

              batch_xs=batch_xs.reshape(batch_size,-1)
                
              if(use_velocities):
                batch_vxts=get_input_vx(0,61,3,"low")
                batch_vyts=get_input_vy(0,61,3,"low")
                batch_vxts=batch_vxts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)
                batch_vyts=batch_vyts.reshape(batch_size,n_t,size_tile_low,size_tile_low,1)
                
                batch_xts=np.concatenate((batch_xts,batch_vxts,batch_vyts),-1)
                
              batch_xts=batch_xts.reshape(batch_size*n_t,-1)

              dict_train={x:batch_xs,x_disc_spatial:batch_xs,y:batch_ys,train:True,lr_global_step:mark,x_t:batch_xts,k_k_tempo:k_tempo,k_k2:k2,k_k1:k1}
              _,gen_loss_return=sess.run([gen_optimizer,gen_loss],feed_dict=dict_train,options=run_options,run_metadata=run_metadata)
              print("gen_loss_return: %f"%gen_loss_return)
            j=j+1
            
            save_log(gen_loss_return,disc_tempo_loss_return,disc_spatial_loss_return)        
      #save model
      save_model(model_path)
      print("******Training finished******")
    else:
      pass

    #main generate process for test
    #------------------------------------------------------
    if(test_mode):
      run_options = None; run_metadata = None
      for i in range (train_index_begin,train_index_begin+1):
        for j in range(frame_num_gen):
          batch_xs=get_input(i,j,1,"low")
          batch_ys=get_input(i,j,1,"high")
          batch_xs=batch_xs.reshape(-1,size_tile_low,size_tile_low,1)
          if(use_velocities):
            batch_vxs=get_input_vx(i,j,1,"low")
            batch_vys=get_input_vy(i,j,1,"low")
            batch_vxs=batch_vxs.reshape(-1,size_tile_low,size_tile_low,1)
            batch_vys=batch_vys.reshape(-1,size_tile_low,size_tile_low,1)
            batch_xs=np.concatenate((batch_xs,batch_vxs,batch_vys),-1)

          batch_xs=batch_xs.reshape(-1,n_input_generator)
          print("batch_xs's shape: {}".format(batch_xs.shape))

          dict_train={x:batch_xs,train:False}
          result_gen,result_inter=sess.run([gen_output,inter_output],feed_dict=dict_train,options=run_options,run_metadata=run_metadata)
          #print(result_gen)
          #generate_test_data(index_now,frame_now,image_matrix,real_fake): #function definition
          result_gen=result_gen.reshape(-1,size_tile_high,size_tile_high)
          generate_image(i,j,result_gen,"high_fake")

          batch_ys=batch_ys.reshape(-1,size_tile_high,size_tile_high)
          generate_image(i,j,batch_ys,"high_real")
          result_inter_image=result_inter[:,:,0]
          result_inter_image=result_inter_image.reshape(-1,size_tile_high,size_tile_high)
          generate_image(i,j,result_inter_image,"high_interpolation_a")

          if(use_velocities):
            result_inter_image_1=result_inter[:,:,1]
            result_inter_image_1=result_inter_image_1.reshape(-1,size_tile_high,size_tile_high)
            generate_image(i,j,result_inter_image_1,"high_interpolation_b")
            result_inter_image_2=result_inter[:,:,2]
            result_inter_image_2=result_inter_image_2.reshape(-1,size_tile_high,size_tile_high)
            generate_image(i,j,result_inter_image_2,"high_interpolation_c")

          batch_xs=batch_xs.reshape(-1,size_tile_low*size_tile_low,num_input_channels)
          batch_xs_image=batch_xs[:,:,0]
          batch_xs_image=batch_xs_image.reshape(-1,size_tile_low,size_tile_low)
          generate_image(i,j,batch_xs_image,"low_real")
      print("******Test finished******")

#That is all
