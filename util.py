import caffe
import skimage
import cv2
from config import *
import numpy as np
from skimage import io

def get_net():
    return caffe.Classifier(DEPLOY_FILE, MODEL_FILE, mean=np.load(MEAN_FILE), \
            channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))

def get_feature(x, net):
    try:
        image = skimage.img_as_float(get_cropped_image(x))
        image = net.preprocess('data', image)
        net.blobs['data'].data[...] = image
        net.forward()

        return net.blobs['fc7'].data[4].flatten()
    except:
        return []

def get_cropped_image(x):
    print x
    image = skimage.img_as_float(io.imread(x)).astype(np.float32)

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (256,256))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * 256.0/height), 256))
        cropping_length = int((resized_image.shape[1] - 256.0) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (256, int(height * 256.0 / width)))
        cropping_length = int((resized_image.shape[0] - 256.0) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return resized_image

