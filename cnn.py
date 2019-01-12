import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('bird.jpg')  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
# print("After conversion to numerical representation: \n\n %r" % arr) 
imgplot = plt.imshow(arr, cmap='gray')
#print("\n Input image converted to gray scale: \n")
# plt.show(imgplot)
kernel = np.array([[ 0, 1, 0],
                   [ 1,-4, 1],
                   [ 0, 1, 0],]) 

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255