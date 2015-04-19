import pandas as pd
import numpy as np
import os
from config import *

def set_dataset():
    image_abspath = os.path.abspath(image_path)
    image_list = pd.read_table(os.path.join(data_path, 'annotations', 'list.txt'), \
            header=None, index_col=None, comment='#', sep=' ',\
            names=['filename','species', 'class', 'dummy'])

    image_list = image_list.drop(['species', 'dummy'], 1)
    image_list['class'] = image_list['class'].map(lambda x: int(x) - 1)
    image_list['filepath'] = image_list['filename'].map(lambda x: os.path.join(image_abspath, x+'.jpg'))

    index = list(image_list.index)
    np.random.shuffle(index)

    shuffled_list = image_list.ix[index]
    num_train = int(len(shuffled_list) * 8.0 / 10.0) # split dataset into train/test

    train_data = shuffled_list[:num_train]
    test_data = shuffled_list[num_train:]

    train_data[['filepath', 'class']].to_csv(os.path.join(model_path, 'train.txt'), header=None, \
                                                index=None, sep='\t')
    test_data[['filepath', 'class']].to_csv(os.path.join(model_path, 'test.txt'), header=None, \
                                                index=None, sep='\t')

def main():
    set_dataset()
    os.system("cd "+CAFFE_ROOT + "; " +\
            "./build/tools/caffe train -solver "+ SOLVER_FILE \
            + " -weights "+ MODEL_FILE + \
            " -gpu 0")

if __name__=="__main__":
    main()
