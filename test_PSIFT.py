#coding: utf-8

#user:rain
#function:

import numpy as np
from PSIFT import Psift
import pylab as pl
from scipy import misc
from PIL import Image
from scipy import ndimage
import scipy
import os
import cv2
if __name__ == '__main__':
        filename = ".\\DATAResult\\"
        #path = ['skin'+np.str(x).zfill(3) for x in np.arange(1,5)]
        #for Dname in path:
        Dname = "skin011\\"
        file = filename+Dname+'\\'
        #im1 = np.asarray(Image.open("D:\\py\\PR15_matching\\data\\orig\\bs000_N_N_0.png").convert('L'))
        #im2 = np.asarray(Image.open("D:\\py\\PR15_matching\\data\\orig\\bs000_YR_R30_0.png").convert('L'))
        im1 = np.asarray(Image.open(file+'im1.png').convert('L'))
        im2 = np.asarray(Image.open(file+'im2.png').convert('L'))
        im2 = ndimage.rotate(im2, 180)
        sampling = 2
        im1 = misc.imresize(im1, [int(im1.shape[0]/sampling), int(im1.shape[1]/sampling)])
        im2 =  misc.imresize(im2, [int(im2.shape[0]/sampling), int(im2.shape[1]/sampling)])

        psift1 = Psift(im1,octave_num=2,sigma1=1.1,upSampling=False)
        psift1.CreatePyramid()
        #psift1.DisPlayPyramid()
        psift1.SavePlayPyramid(file+'im1')
        point_list1 = psift1.ScolePoint(rmedge=False,curvature=10,specofic_hd=0)
        feature1, new_point1 = psift1.GetFeature(point_list1)

        psift2 = Psift(im2, octave_num=2, sigma1=1.1,upSampling=False)
        psift2.CreatePyramid()

        psift2.SavePlayPyramid(file+'im2')

        point_list2 = psift2.ScolePoint(rmedge=False, curvature=10, specofic_hd=0)
        feature2,new_point2 = psift2.GetFeature(point_list2)

        a_pts,b_pts = Psift.match(np.array([feature1,new_point1,im1]),np.array([feature2,new_point2,im2]),file,ratio=0.8,RANSAC=True,kmeans=False)

        Psift.plot_match(im1,im2,a_pts,b_pts,file+'RANSAC',gray=True)
        pl.close()