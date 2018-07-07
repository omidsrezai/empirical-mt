import numpy as np

def tent(x,centre_indx=0):
    centres =  np.exp(np.arange(0,5,.45))
    centre = centres[centre_indx+1]
    begin_ = centres[centre_indx]# - 16#np.log(centre-.4)
    end_ = centres[centre_indx+2] #+ 16#np.log(centre+.4)
    y = np.zeros(x.shape)    
    #print('x.shape',x.shape)
    y[(x>=begin_)&(x<=centre)] = (x[(x>=begin_)&(x<=centre)]-begin_)/(centre-begin_)
    y[(x>=centre)&(x<=end_)]   = (end_-x[(x>=centre)&(x<=end_)])/(end_-centre)    


    return y

def get_tent_responses(flow_speed,n_tent, unit_conv):

    speed_deg_per_second = unit_conv * flow_speed 
    speed_tent_response = np.zeros([flow_speed.shape[0],flow_speed.shape[1],flow_speed.shape[2],flow_speed.shape[3],n_tent])

    for i in range(n_tent): 
        speed_tent_response[:,:,:,:,i] = tent(speed_deg_per_second,centre_indx=i)
    return speed_tent_response 
