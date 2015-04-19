#-*- coding: utf-8 -*-
import os

image_path = './images'
data_path = './data'
model_path = './models'

CAFFE_ROOT= "/home/taeksoo/Package/caffe"
MODEL_FILE = os.path.join(CAFFE_ROOT,"models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")
DEPLOY_FILE = os.path.join(CAFFE_ROOT,"models/bvlc_reference_caffenet/deploy.prototxt")
MEAN_FILE = os.path.join(CAFFE_ROOT,"python/caffe/imagenet/ilsvrc_2012_mean.npy")
SOLVER_FILE= os.path.join(os.path.abspath(model_path), 'solver.prototxt')
DATASET = "oxford"


