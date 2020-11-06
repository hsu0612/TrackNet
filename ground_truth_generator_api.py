import glob
import csv
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import expanduser

class ground_truth_generator():
    def __init__(self):
        self.size = 20
    #create gussian heatmap 
    def gaussian_kernel(self, variance):
        x, y = numpy.mgrid[-self.size:self.size+1, -self.size:self.size+1]
        g = numpy.exp(-(x**2+y**2)/float(2*variance))
        return g 
    def ground_truth_generator(self, images_path):
        #make the Gaussian by calling the function
        variance = 10
        gaussian_kernel_array = self.gaussian_kernel(variance)
        #rescale the value to 0-255
        gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
        #change type as integer
        gaussian_kernel_array = gaussian_kernel_array.astype(int)

        #create the heatmap as ground truth
        dirs = glob.glob(images_path+'/Clip*')
        for index in dirs:
            #################change the path####################################################
            pics = glob.glob(index + "/*.jpg")
            output_pics_path = images_path+'groundtruth/' + os.path.split(index)[-1]
            label_path = index + "/Label.csv"
            ####################################################################################
                
            #check if the path need to be create
            if not os.path.exists(output_pics_path ):
                os.makedirs(output_pics_path)
                    
            #read csv file
            with open(label_path, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                #skip the headers
                next(spamreader, None)  
                    
                for row in spamreader:
                    visibility = int(float(row[1]))
                    FileName = row[0]
                    #if visibility == 0, the heatmap is a black image
                    if visibility == 0:
                        heatmap = Image.new("RGB", (1280, 1024))
                        pix = heatmap.load()
                        for i in range(1280):
                            for j in range(720):
                                pix[i,j] = (0,0,0)
                    else:
                        x = int(float(row[2]))
                        y = int(float(row[3]))
                                
                        #create a black image
                        heatmap = Image.new("RGB", (1280, 1024))
                        pix = heatmap.load()
                        for i in range(1280):
                            for j in range(1024):
                                pix[i,j] = (0,0,0)
                                            
                        #copy the heatmap on it
                        for i in range(-self.size,self.size+1):
                            for j in range(-self.size,self.size+1):
                                if x+i<1280 and x+i>=0 and y+j<1024 and y+j>=0 :
                                    temp = gaussian_kernel_array[i+self.size][j+self.size]
                                    if temp > 0:
                                        pix[x+i,y+j] = (temp,temp,temp)
                    #save image
                    heatmap.save(output_pics_path + "/" + FileName.split('.')[-2] + ".png", "PNG")
