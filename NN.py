from __future__ import division
import numpy
import csv
import random

import time

start_time = time.time()


#Reading All the inputs into variables

dest_file = 'D:\Zmisc\Github\CS\Neural_Networks\\breast_cancer_data_modified.csv'


Nh=60


eps = 0.02


eta = 0.07


epochs = 300


with open(dest_file,'r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',')
    data = [data for data in data_iter]


data_array = numpy.asarray(data)  

num_cols = len(data_array[0])
num_rows = len(data_array[:,0])

y = list(data_array[:,-1])
x = data_array[:,:-1]
x = numpy.asarray(x,dtype=float)

possible_y_vals = list(set(y))
num_y_vals = len(possible_y_vals)

for counter in range(num_rows):
    i = 0
    while i < num_y_vals:
        if possible_y_vals[i]==y[counter]:
            y[counter] = numpy.zeros(num_y_vals)
            y[counter][i] = 1
            break
        i+=1

y = numpy.asarray(y,dtype=float)
#print y
data_array = numpy.append(x,y,axis=1) 

#for x in range(num_rows):
 #   print data_array[x,-num_y_vals:]

weights_inner_hidden = numpy.random.rand(num_cols+1-1,Nh)-0.5
weights_hidden_output = numpy.random.rand(Nh+1,num_y_vals)-0.5       


data_array_copy = numpy.copy(data_array)
random.shuffle(data_array)
for outer in range(epochs):
      
    #for z in range(num_rows):
      #  print data_array[z,:]
    for counter in range(num_rows):
        
        data_point = data_array[counter,:-num_y_vals]
        y_correct = data_array[counter,-num_y_vals:]
        data_point = numpy.append(data_point,1)
        
        while True:
            #Feedforward
            step1_pre_act = numpy.dot(data_point,weights_inner_hidden)            
            step1 = numpy.divide(1,1+numpy.exp(-step1_pre_act))
            
            step1 = numpy.append(step1,1)
            
            step2_pre_act = numpy.dot(step1,weights_hidden_output)
            step2 = numpy.divide(1,1+numpy.exp(-step2_pre_act))
            
            #print step2
            #break

            error = numpy.subtract(step2,y_correct)
            error = numpy.dot(error,error)/2
            if error <= eps:
                break
            else:
                #Back propagation
                del_outer = numpy.zeros(num_y_vals)
                for p in range(num_y_vals):
                    del_outer[p] = (y_correct[p]-step2[p])*step2[p]*(1-step2[p])
                
                
                
                del_inner = numpy.zeros(Nh)
                for p in range(Nh):
                    del_inner[p] = step1[p]*(1-step1[p])*numpy.dot(del_outer,weights_hidden_output[p,:])
                
                
                for c1 in range(Nh+1): 
                    for c2 in range(num_y_vals):
                        weights_hidden_output[c1,c2] = weights_hidden_output[c1,c2] + (eta * del_outer[c2] * step1[c1])                
                
                for c1 in range(num_cols):
                    for c2 in range(Nh):
                        weights_inner_hidden[c1,c2] = weights_inner_hidden[c1,c2] + (eta * del_inner[c2] * data_point[c1])
                
        
        y_plus = list(y_correct).index(1)
        step2 = list(step2)
        if y_plus!= step2.index(max(step2)):
            print "Error in %d epoch" %outer
             
             
num_errors = 0 
for counter in range(num_rows):
    data_point = data_array_copy[counter,:-num_y_vals]
    y_correct = data_array_copy[counter,-num_y_vals:]
    data_point = numpy.append(data_point,1)
    step1_pre_act = numpy.dot(data_point,weights_inner_hidden)            
    step1 = numpy.divide(1,1+numpy.exp(-step1_pre_act))
            
    step1 = numpy.append(step1,1)
            
    step2_pre_act = numpy.dot(step1,weights_hidden_output)
    step2 = numpy.divide(1,1+numpy.exp(-step2_pre_act))
    
    y_plus = list(y_correct).index(1)
    step2 = list(step2)
    if y_plus!= step2.index(max(step2)):
        num_errors+=1

accuracy = 1 - (num_errors/num_rows)


#print "Nh=",Nh,"Epsilon=",eps,"eta=",eta,"epochs=",epochs
print "Accuracy=",accuracy*100,"%"        
print time.time() - start_time, "seconds"
