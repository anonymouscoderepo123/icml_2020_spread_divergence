# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import random
import logging
import tensorflow as tf
import numpy as np
from six.moves import cPickle
import utils
import PIL
from utils import ArraySaver
from PIL import Image
import sys

datashapes = {}
datashapes['celebA'] = [64, 64, 3]

def _data_dir(opts):
    if opts['data_dir'].startswith("/"):
        return opts['data_dir']
    else:
        return os.path.join('./', opts['data_dir'])



class Data(object):
    """
    If the dataset can be quickly loaded to memory self.X will contain np.ndarray
    Otherwise we will be reading files as we train. In this case self.X is a structure:
        self.X.paths        list of paths to the files containing pictures
        self.X.dict_loaded  dictionary of (key, val), where key is the index of the
                            already loaded datapoint and val is the corresponding index
                            in self.X.loaded
        self.X.loaded       list containing already loaded pictures
    """
    def __init__(self, opts, X, paths=None, dict_loaded=None, loaded=None):
        """
        X is either np.ndarray or paths
        """
        data_dir = _data_dir(opts)
        self.X = None
        self.normalize = opts['input_normalize_sym']
        self.paths = None
        self.dict_loaded = None
        self.loaded = None
        if isinstance(X, np.ndarray):
            self.X = X
            self.shape = X.shape
        else:
            assert isinstance(data_dir, str), 'Data directory not provided'
            assert paths is not None and len(paths) > 0, 'No paths provided for the data'
            self.data_dir = data_dir
            self.paths = paths[:]
            self.dict_loaded = {} if dict_loaded is None else dict_loaded
            self.loaded = [] if loaded is None else loaded
            self.crop_style = opts['celebA_crop']
            self.dataset_name = opts['dataset']
            self.shape = (len(self.paths), None, None, None)

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return len(self.X)
        else:
            # Our dataset was too large to fit in the memory
            return len(self.paths)

    def drop_loaded(self):
        if not isinstance(self.X, np.ndarray):
            self.dict_loaded = {}
            self.loaded = []

    def __getitem__(self, key):
        if isinstance(self.X, np.ndarray):
            return self.X[key]
        else:
            # Our dataset was too large to fit in the memory
            if isinstance(key, int):
                keys = [key]
            elif isinstance(key, list):
                keys = key
            elif isinstance(key, np.ndarray):
                keys = list(key)
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                step = key.step
                start = start if start is not None else 0
                if start < 0:
                    start += len(self.paths)
                stop = stop if stop is not None else len(self.paths) - 1
                if stop < 0:
                    stop += len(self.paths)
                step = step if step is not None else 1
                keys = range(start, stop, step)
            else:
                print (type(key))
                raise Exception('This type of indexing yet not supported for the dataset')
            res = []
            new_keys = []
            new_points = []
            for key in keys:
                if key in self.dict_loaded:
                    idx = self.dict_loaded[key]
                    res.append(self.loaded[idx])
                else:
                    if self.dataset_name == 'celebA':
                        point = self._read_celeba_image(self.data_dir, self.paths[key])
                    else:
                        raise Exception('Disc read for this dataset not implemented yet...')
                    if self.normalize:
                        point = (point - 0.5) * 2.
                    res.append(point)
                    new_points.append(point)
                    new_keys.append(key)
            n = len(self.loaded)
            cnt = 0
            for key in new_keys:
                self.dict_loaded[key] = n + cnt
                cnt += 1
            self.loaded.extend(new_points)
            return np.array(res)

    def _read_celeba_image(self, data_dir, filename):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(utils.o_gfile((data_dir, filename), 'rb'))
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height)/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), PIL.Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), PIL.Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        return np.array(im).reshape(64, 64, 3) / 255.

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self._load_data(opts)

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'] == 'celebA':
            self._load_celebA(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        # sym_applicable = ['celebA']
        #
        # if opts['input_normalize_sym'] and opts['dataset'] not in sym_applicable:
        #     raise Exception('Can not normalyze this dataset')
        #
        # if opts['input_normalize_sym'] and opts['dataset'] in sym_applicable:
        #     # Normalize data to [-1, 1]
        #     if isinstance(self.data.X, np.ndarray):
        #         self.data.X = (self.data.X - 0.5) * 2.
        #         self.test_data.X = (self.test_data.X - 0.5) * 2.
        #     # Else we will normalyze while reading from disk


    def _load_celebA(self, opts):
        """Load CelebA
        """
        logging.debug('Loading CelebA dataset')

        num_samples = 202599

        datapoint_ids = list(range(1, num_samples + 1))
        paths = ['%.6d.jpg' % i for i in range(1, num_samples + 1)]
        seed = 123
        random.seed(seed)
        random.shuffle(paths)
        random.shuffle(datapoint_ids)
        random.seed()

        saver = ArraySaver('disk', workdir=opts['work_dir'])
        saver.save('shuffled_training_ids', datapoint_ids)

        self.data_shape = (64, 64, 3)
        test_size = 20000
        self.data = Data(opts, None, paths[:-test_size])
        self.test_data = Data(opts, None, paths[-test_size:])
        self.num_points = num_samples - test_size
        self.labels = np.array(self.num_points * [0])
        self.test_labels = np.array(test_size * [0])

        logging.debug('Loading Done.')