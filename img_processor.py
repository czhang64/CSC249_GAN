import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import skimage
import os

data=[]
dir = (r"Impressionism\\")
for files in os.listdir(dir):
    temp = Image.open(dir+files)
    temp = temp.resize((128,128))
    #plt.imshow(temp)
    #plt.show()
    temp = np.asarray(temp)
    data.append(temp)

data = np.reshape(data,(-1,128,128,3))
data = data.astype(np.float32)
data = data / 127.5 - 1.
np.save('impressionism_training_data_128_128.npy',data)
print(data.shape)
print("done")
    
