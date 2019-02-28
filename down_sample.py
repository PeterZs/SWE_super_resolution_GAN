#!usr/bin/python
#get low resolution data set by gauss filtering the high resolution and interpolation the blur result to low resolution version.
# maybe there are some smoothing filter and the interpolation method's combination. What is the best for the recovering(super resolution problem)

#from scipy import misc
import cv2
import numpy as np
import math
import paramhelpers as ph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data_path=ph.getParam("data_path",'../SWE_solver/data/data_velocity_2')
index_begin=int(ph.getParam("index_begin",0))
index_end=int(ph.getParam("index_end",20))
frame_num_per_sim=int(ph.getParam("frame_num_per_sim",120))
scale=float(ph.getParam("scale",4.0))
size_sim_high=int(ph.getParam("size_sim_high",256))
size_sim_low=int(ph.getParam("size_sim_low",64))
interp=ph.getParam("interp",'bicubic')

sigma=scale/2.0/math.sqrt(math.pi)
kernal_size=(int(6*sigma+1),int(6*sigma+1))

print(kernal_size)
file_name=open(data_path+"/log.txt",'a')
print >> file_name,"data_path: %s"%data_path
print >> file_name,"index_begin: %d"%index_begin
print >> file_name,"index_end: %d"%index_end
print >> file_name,"frame_num_per_sim: %d"%frame_num_per_sim
print >> file_name,"scale: %f"%scale
print >> file_name,"size_sim_high: %d"%size_sim_high
print >> file_name,"size_sim_low: %d"%size_sim_low
print >> file_name,"interp: %s"%interp

file_name.close()

#---------------------------------------------------
def down_sample(kind_high,kind_low,i,j):
    high_data_path=data_path+"/"+str(i)+"/"+kind_high+str(j)+".bin"
    high_data=np.zeros(shape=(size_sim_high*size_sim_high),dtype=np.float64)
    high_data[:]=np.fromfile(high_data_path)
    high_data=high_data.reshape((size_sim_high,size_sim_high))      
    high_data_blur=cv2.GaussianBlur(high_data,kernal_size,sigma)
    # print("high_data_blur.dtype: {}".format(high_data_blur.dtype))
    low_data=cv2.resize(high_data_blur,(size_sim_low,size_sim_low),interpolation=cv2.INTER_LINEAR)
    # print("low_data.dtype: {}".format(low_data.dtype))

    low_data_png_path=data_path+"/"+str(i)+"/"+kind_low+str(j)+".png"
    low_data=low_data.reshape((1,size_sim_low,size_sim_low))
    generate_image(low_data,low_data_png_path,size_sim_low)
    
    low_data=low_data.reshape((size_sim_low*size_sim_low))
    # print("low_data.dtype: {}".format(low_data.dtype))
    low_data_path=data_path+"/"+str(i)+"/"+kind_low+str(j)+".bin"
    low_data.tofile(low_data_path)

    #high_data=cv2.resize(high_data,(size_sim_high,size_sim_high),interpolation=cv2.INTER_NEAREST)
    high_data=high_data.reshape((1,size_sim_high,size_sim_high))
    high_data_png_path=data_path+"/"+str(i)+"/"+kind_high+str(j)+".png"
    generate_image(high_data,high_data_png_path,size_sim_high)


def generate_image(image_matrix,image_path,size_sim_here):
    fig=plt.figure(figsize=(1,1))
    gs=gridspec.GridSpec(1,1)
    gs.update(wspace=0,hspace=0)
    for i,sample in enumerate(image_matrix):
        ax=plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample,cmap="Greys_r")

    plt.savefig(image_path,dpi=size_sim_here)
    plt.close(fig)

for i in range(index_begin,index_end):
    for j in range(frame_num_per_sim):
        print(i,j)
        kind_high="high_"
        kind_low="low_"
        down_sample(kind_high,kind_low,i,j)

        kind_high="high_vx_"
        kind_low="low_vx_"
        down_sample(kind_high,kind_low,i,j)

        kind_high="high_vy_"
        kind_low="low_vy_"
        down_sample(kind_high,kind_low,i,j)


        
        
        
        
