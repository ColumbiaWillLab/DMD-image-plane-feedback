# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:34:16 2020

@author: user
"""
import numpy as np
from ALP4 import *
import time
from typing import Optional
import cv2
from pymba import Frame
from pymba import frame
from pymba import Frame
from pymba import Vimba
from pymba import camera
from pymba import vimba_pixelformat
from pymba import vimba_c
from typing import Optional
import cv2
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.colors as col

ImgShow = 'true'
#ImgShow = 'false'

Feedback = 'true'
#Feedback = 'false'
err_mult = 1

CameraPixelSize = 4.8*10**-6
CameraXPix = 1024
CameraYPix = 1280


# Load the Vialux .dll
DMD = ALP4(version = '4.2', libDir = 'C:/Users/user/Documents/Columbia/Will Lab/DMD')
DMD.Initialize()
bitDepth = 1
'''
Variables
'''
PixelSizeRealDMD = 13.69*10**-6 #Size of a DMD pixel#
Mag = 1/(5*30) #maging system magnification - using Waseem's for now
    
#DMD Pixel numbers (DMD.nSizeY,DMD.nSizeX)
xPixels = 1024
yPixels = 768
#DMD center pizels (for shifting) 
x0 = xPixels/2
y0 =  yPixels/2

Sigma = 333 #accordion lattice potential waist
NA = 0.6 #microscope NA
Lambda = 532*10**-9 #light wavelenght used [m]


CamDMDMag = 3
CAMimgSizeX = int(np.round(xPixels*PixelSizeRealDMD/(CameraPixelSize*CamDMDMag)))
CAMimgSizeY = int(np.round(yPixels*PixelSizeRealDMD/(CameraPixelSize*CamDMDMag)))

PixelSizeAtoms = PixelSizeRealDMD * Mag #size of a pixel's image at the atoms' position
PSFRadius = (1.22 * Lambda)/(2 * NA) #Radius of the microscope's PSF [m]
PSFPixelRadius = np.around( PSFRadius / PixelSizeAtoms) #Radius of the PSF in pixels (at the atoms' position)

def Flat(Array):
    '''
    Parameters
    ----------
    Array : 2D array
        Whichever image needs to be shown on the DMD.

    Returns
    -------
    Upload : 1D array
        Returns a 1D array that can be uploaded on the DMD.
    '''
    Upload = np.zeros(np.size(Array),dtype = int)
    counter = 0
    for i in range(0,yPixels):
        for j in range(0,xPixels):
            Upload[counter] = Array[i,j]
            counter = counter + 1
    return Upload

def Dither(Array):
    '''
    Parameters
    ----------
    Array : 2D array
        Whichever image you need to dither through a Floyd Steinberg algorithm.

    Returns
    -------
    Upload : 2D array
        Dithered version of Array.
    '''
    Dither_pic = Array
    for i in range(0,yPixels-1):
        for j in range(1,xPixels-1):
            OldPixel = Dither_pic[i,j]
            if OldPixel < 0.5:
                NewPixel = 0
            else:
               NewPixel = 1
               QuantError = -(NewPixel - OldPixel)
               Dither_pic[i,j] = NewPixel
               Dither_pic[i + 1, j - 1] = Dither_pic[i + 1, j - 1] + QuantError*3/16
               Dither_pic[i + 1, j] = Dither_pic[i + 1, j] + QuantError*5/16
               Dither_pic[i + 1, j + 1] = Dither_pic[i + 1, j + 1] + QuantError*1/16
               Dither_pic[i, j + 1] = Dither_pic[i, j + 1] + QuantError*7/16
               Dither_pic = Dither_pic.astype(int)
        return Dither_pic

'''
Make Gaussian with box
'''
GreyScaleGaussian = np.ones([yPixels,xPixels])

for i in range(0,xPixels):
    for j in range(0,yPixels):
        GreyScaleGaussian[j,i] = 1 - np.exp(-((i - x0)**2 + (j - y0)**2)/(2 * Sigma**2))
for i in range(0,xPixels):
    for j in range(0,yPixels):
        if ((i - x0)**2 + (j - y0)**2)**0.5 > Sigma:
            GreyScaleGaussian[j,i] = 1 

error_matrix = np.ones([yPixels,xPixels])
Dithered = Dither(GreyScaleGaussian)

while np.max(error_matrix) > 0.1: 

    DMD.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
    DMD.SeqPut(imgData = Flat(Dithered)*(2**8-1))
    DMD.Run(loop = True)

    with Vimba() as vimba:
            ID = vimba.camera_ids()[0]
            cam = camera.Camera(1, ID)
            cam.open()
            cam.arm(camera.SINGLE_FRAME)
            pic = cam.acquire_frame()
            cam.disarm()
            pic_data = pic.buffer_data_numpy()
            # if ImgShow == 'true':
            #     plt.imshow(pic_data, cmap = 'Greys')#, norm = col.Normalize(0,1))

    DMD.Halt()
    DMD.FreeSeq()

    pic_data_crop = pic_data[:CAMimgSizeX,:CAMimgSizeY]
    pic_data_resize = cv2.resize(pic_data_crop,(xPixels,yPixels),interpolation = cv2.INTER_NEAREST)
    pic_data_resize = pic_data_resize/np.max(pic_data_resize)
    error_matrix = GreyScaleGaussian - pic_data_resize
    Dithered = Dither(Dithered - error_matrix * err_mult)

DMD.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
DMD.SeqPut(imgData = Flat(Dithered)*(2**8-1))
DMD.Run(loop = True)

time.sleep(10)

DMD.Halt()
DMD.FreeSeq()
DMD.Free()     

if ImgShow == 'true':
    plt.imshow(pic_data_resize, cmap = 'Greys')#, norm = col.Normalize(0,1))

