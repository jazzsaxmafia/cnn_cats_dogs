import pandas as pd
import os
from config import *
from util import *
import caffe

def process_image(fn, image_path, net):
    image_file_path = os.path.join(image_path, fn+'.jpg')
    return get_feature(image_file_path, net)

def main():
    net = get_net()
    caffe.set_mode_gpu()
    data_file = os.path.join(data_path, 'annotations', 'list.txt')
    data= pd.read_table(data_file, header=None, sep=' ', comment='#')
    data.columns = ['filename', 'species', 'class', 'dummy']

    data = data.drop(['species', 'dummy'], 1)
    data['feature'] = data['filename'].map(lambda fn: process_image(fn, image_path, net))

    data.to_pickle(os.path.join(data_path, 'feature.pickle'))

if __name__=="__main__":
    main()
