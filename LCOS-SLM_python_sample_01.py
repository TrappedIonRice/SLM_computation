# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:10:00 2020

@author: HPK
"""

from PIL import Image
import numpy as np
from ctypes import *
import copy

def main():
    #pixelpitch(0: 20um 1: 1.25um)
    pitch = 1
    
    #LCOS pixel size
    x = 1272
    y = 1024
    
    #LCOS-SML monitor number setting
    monitorNo = 2
    windowNo = 0
    xShift = 0
    yShift = 0
    
    #pixel number
    array_size = x * y

    # make the 8bit unsigned integer array type
    FARRAY = c_uint8 * array_size

    # make the 8bit unsigned integer array instance
    farray = FARRAY(0)
    farray2 = FARRAY(0)
    farray3 = FARRAY(0)
    
    #Display axiconLens pattern array with using dll
    top = 10.0 #top level
    makeAxiconLensArray(top, pitch, x, y, farray)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray)
    
    #Display cylindricalLens pattern array with using dll
    forcus = 1000
    wavelength = 1064
    modeSelect = 0
    makeCylindricalLensArray(forcus, wavelength, pitch, modeSelect, x, y, farray)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray)
    
    #Display rotated pattern array with using dll
    degree = 30.0
    imageRotation(farray, degree, x, y, farray2)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray2)
    
    #Display diffraction grating pattern array with using dll
    rowOrColumn = 0
    gradiationNo = 16
    gradiationWidth = 16
    slipFactor = 0
    makeDiffractionPatternArray(rowOrColumn, gradiationNo, gradiationWidth, slipFactor, x, y, farray)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray)
    
    #Display Laguerre Gauss Mode pattern array with using dll
    p = 1
    m = 1
    pitch = 1
    beamSize = 20.0
    makeLaguerreGaussModeArray(p, m, pitch, beamSize, x, y, farray)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray)
    
    #Display FresnelLens pattern array with using dll
    forcus = 1000
    wavelength = 1064
    makeFresnelLensArray(forcus, wavelength, pitch, x, y, farray2)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray2)
    
    #Display synthesize FresnelLens and Laguerre Gauss Mode pattern array with using dll
    phaseSynthsizer([farray, farray2], farray3)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray3)
 
    #Display CGH pattern from image file with using dll
    filepath = "Target image sample\\char_hpk_128x128.bmp"
    makeBmpArray(filepath, x, y, farray)
    showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, farray)
    
    return 0

'''
the function for making AxiconLens pattern array
double top: Top level of AxiconLens pattern. (/pi rad)
int pitch: Pixel pitch. 0: 20um 1: 1.25um
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned integer array array: output array
'''
def makeAxiconLensArray(top, pitch, x, y, array):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll") #for cdll
    Lcoslib = windll.LoadLibrary("Image_Control.dll") #for windll
    AxiconLens = Lcoslib.AxiconLens
    AxiconLens.argtyes = [c_double, c_int, c_int, c_int, c_void_p, c_void_p]
    AxiconLens.restype = c_int
    if(pitch != 0 and pitch != 1):
        print("Error: AxiconLensFunction. invalid argument (pitch).")
        return -1
    # input argument to dll function.
    AxiconLens(c_double(top), pitch, x, y, byref(c_int(x*y)), byref(array))
    return 0

'''
the function for making CylindricalLens pattern array
int focus: the forcus of cylindrical lens. (mm)
int wavelength: the wavelength of light. (nm)
int pitch: Pixel pitch. 0: 20um 1: 1.25um
int modeSelect: 0: horizontal or 1: vertical
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned int array array: output array
'''
def makeCylindricalLensArray(forcus, wavelength, pitch, modeSelect, x, y, array):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    CylindricalLens = Lcoslib.CylindricalLens
    CylindricalLens.argtyes = [c_int, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
    CylindricalLens.restype = c_int
    if(pitch != 0 and pitch != 1):
        print("Error: CylindricalLensFunction. invalid argument (pitch).")
        return -1
    CylindricalLens(forcus, wavelength, pitch, modeSelect, x, y, byref(c_int(x*y)), byref(array))
    return 0

'''
the function for making Diffraction pattern array
int rowOrColumn: 0: horizontal or 1: vertical
int gradiationNo: the number of gradiation.
int gradiationWidth: the width of gradiation.
int slipFactor: slip factor.
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned int array array: output array
'''
def makeDiffractionPatternArray(rowOrColumn, gradiationNo, gradiationWidth, slipFactor, x, y, array):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    Diffraction_pattern = Lcoslib.Diffraction_pattern
    Diffraction_pattern.argtyes = [c_int, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
    Diffraction_pattern.restype = c_int
    Diffraction_pattern(rowOrColumn, gradiationNo, gradiationWidth, slipFactor, x, y, byref(c_int(x*y)), byref(array))
    return 0

'''
the function for making LaguerreGaussMode pattern array
int p: radial index
int m: azimuthal index 
int pitch: Pixel pitch. 0: 20um 1: 1.25um
double beamSize: Beam size (mm)
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned int array array: output array
'''
def makeLaguerreGaussModeArray(p, m, pitch, beamSize, x, y, array):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    LaguerreGaussMode = Lcoslib.LaguerreGaussMode
    LaguerreGaussMode.argtyes = [c_int, c_int, c_int, c_double, c_int, c_int, c_void_p, c_void_p]
    LaguerreGaussMode.restype = c_int
    if(pitch != 0 and pitch != 1):
        print("Error: LaguerreGaussModeFunction. invalid argument (pitch).")
        return -1
    LaguerreGaussMode(p, m, pitch, c_double(beamSize), x, y, byref(c_int(x*y)), byref(array))
    return 0

'''
the function for making FresnelLens pattern array
int focus: the forcus of cylindrical lens. (mm)
int wavelength: the wavelength of light. (nm)
int pitch: Pixel pitch. 0: 20um 1: 1.25um
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned int array array: output array
'''
def makeFresnelLensArray(forcus, wavelength, pitch, x, y, array):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    FresnelLens = Lcoslib.FresnelLens
    FresnelLens.argtyes = [c_int, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
    FresnelLens.restype = c_int
    if(pitch != 0 and pitch != 1):
        print("Error: FresnelLensFunction. invalid argument (pitch).")
        return -1
    FresnelLens(forcus, wavelength, pitch, x, y, byref(c_int(x*y)), byref(array))
    return 0

'''
the function for making FresnelLens pattern array
String filepath: image file path.
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
8bit unsigned int array outArray: output array
'''
def makeBmpArray(filepath, x, y, outArray):
    im = Image.open(filepath)
    imageHeight, imageWidth = im.size
    im_gray = im.convert("L")
    
    print("Imagesize = {} x {}".format(imageWidth, imageHeight))
    
    for i in range(imageWidth):
        for j in range(imageHeight):
            outArray[i+imageWidth*j] = im_gray.getpixel((i,j))
    
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    
    #Create CGH
    inArray = copy.deepcopy(outArray)
    Create_CGH_OC = Lcoslib.Create_CGH_OC
    Create_CGH_OC.argtyes = [c_void_p, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
    Create_CGH_OC.restype = c_int
    
    repNo = 100
    progressBar = 1
    Create_CGH_OC(byref(inArray), repNo, progressBar, imageWidth, imageHeight, byref(c_int(imageHeight*imageWidth)), byref(outArray))
    
    #Tilling the image
    inArray = copy.deepcopy(outArray)
    Image_Tiling = Lcoslib.Image_Tiling
    Image_Tiling.argtyes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
    Image_Tiling.restype = c_int
    
    Image_Tiling(byref(inArray), imageWidth, imageHeight, imageHeight*imageWidth, x, y, byref(c_int(x*y)), byref(outArray))
    
    return 0

'''
the function for showing on LCOS display
int monitorNo: 
int windowNo:
int x: Pixel number of x-dimension
int xShift: shift pixels of x-dimension
int y: Pixel number of y-dimension
int yShift: shift pixels of y-dimension
8bit unsigned int array array: output array
'''
def showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, array):
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    
    #Select LCOS window
    Window_Settings = Lcoslib.Window_Settings
    Window_Settings.argtypes = [c_int, c_int, c_int, c_int]
    Window_Settings.restype = c_int
    Window_Settings(monitorNo, windowNo, xShift, yShift)
    
    #Show pattern
    Window_Array_to_Display = Lcoslib.Window_Array_to_Display
    Window_Array_to_Display.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
    Window_Array_to_Display.restype = c_int
    Window_Array_to_Display(array, x, y, windowNo, x*y)
    
    #wait until enter key input
    input("please input enter key...")
    
    #close the window
    Window_Term = Lcoslib.Window_Term
    Window_Term.argtyes = [c_int]
    Window_Term.restype = c_int
    Window_Term(windowNo)
    
    return 0

'''
the function for Srotating image
input 1D array inputArray: input array. 
double degree: rotation degree (deg.)
int x: Pixel number of x-dimension
int y: Pixel number of y-dimension
output 1D array outputArray: output array. 
'''  
def imageRotation(inputArray, degree, x, y, outputArray):
    #Lcoslib = cdll.LoadLibrary("Image_Control.dll")
    Lcoslib = windll.LoadLibrary("Image_Control.dll")
    
    Image_Rotation = Lcoslib.Image_Rotation
    Image_Rotation.argtyes = [c_void_p, c_double, c_int, c_int, c_void_p, c_void_p]
    Image_Rotation.restype = c_int
    Image_Rotation(byref(inputArray), c_double(degree), x, y, byref(c_int(x*y)), byref(outputArray))
    
    return 0

'''
the function for Synthesizing image pattaerns
input arrays 2D array inputPatterns: the compornents will be synthesized.
output 1D array outputArray: output array. 
'''  
def phaseSynthsizer(inputPatterns, outputArray):
    n = len(inputPatterns[0])
    outputPattern = np.zeros(n, dtype=int)
    for pattern in inputPatterns:
        outputPattern = outputPattern + pattern
    
    for i in range(n):
        outputArray[i] = c_uint8(outputPattern[i] % 256)    
    
    return 0
    
main()