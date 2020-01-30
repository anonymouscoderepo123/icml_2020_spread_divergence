import tensorflow as tf
import numpy as np
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)


def celeba_show_many(image,number_sqrt):
    shape=np.shape(image)[1:]
    canvas_recon = np.empty((shape[0] * number_sqrt, shape[0] * number_sqrt, shape[2]))
    count = 0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * shape[0]:(i + 1) * shape[0], j * shape[0]:(j + 1) * shape[0], :] = image[count].reshape(shape)
            count += 1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.imshow(canvas_recon)
    plt.axis('off')
    plt.show()

def celeba_save_many(image,number_sqrt,name):
    shape=np.shape(image)[1:]
    canvas_recon = np.empty((shape[0] * number_sqrt, shape[0] * number_sqrt, shape[2]))
    count = 0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * shape[0]:(i + 1) * shape[0], j * shape[0]:(j + 1) * shape[0], :] = image[count].reshape(shape)
            count += 1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imsave(name,canvas_recon)
    plt.close()


def show_many(image,number_sqrt):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()


def os_save(opt):
    import os
    if not os.path.exists(opt['path']):
        os.makedirs(opt['path'])

    fout = opt['path'] + "opt.txt"
    fo = open(fout, "w")

    for k, v in opt.items():
        fo.write(str(k) + ' : ' + str(v) + '\n\n')
    fo.close()



def show_cov(cov):
    plt.matshow(cov)
    plt.colorbar()
    plt.show()  
    
def save_many(image,number_sqrt,name):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imsave(name,canvas_recon, origin="upper", cmap="gray")
    plt.close()


def os_save(opt):
    import os
    if not os.path.exists(opt['path']):
        os.makedirs(opt['path'])
    
    fout = opt['path']+"opt.txt"
    fo = open(fout, "w")

    for k, v in opt.items():
        fo.write(str(k) + ' : '+ str(v) + '\n\n')
    fo.close()
    
def transform(x):
    return x/255.



class ArraySaver(object):
    """A simple class helping with saving/loading numpy arrays from files.
    This class allows to save / load numpy arrays, while storing them either
    on disk or in memory.
    """

    def __init__(self, mode='ram', workdir=None):
        self._mode = mode
        self._workdir = workdir
        self._global_arrays = {}

    def save(self, name, array):
        if self._mode == 'ram':
            self._global_arrays[name] = copy.deepcopy(array)
        elif self._mode == 'disk':
            create_dir(self._workdir)
            np.save(o_gfile((self._workdir, name), 'wb'), array)
        else:
            assert False, 'Unknown save / load mode'

    def load(self, name):
        if self._mode == 'ram':
            return self._global_arrays[name]
        elif self._mode == 'disk':
            return np.load(o_gfile((self._workdir, name), 'rb'))
        else:
            assert False, 'Unknown save / load mode'

def create_dir(d):
    if not tf.gfile.IsDirectory(d):
        tf.gfile.MakeDirs(d)


class File(tf.gfile.GFile):
    """Wrapper on GFile extending seek, to support what python file supports."""
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)

def o_gfile(filename, mode):
    """Wrapper around file open, using gfile underneath.
    filename can be a string or a tuple/list, in which case the components are
    joined to form a full path.
    """
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)

def listdir(dirname):
    return tf.gfile.ListDirectory(dirname)

def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)

def logits(x):
    return np.log(x/(1-x))

def cut(x):
    x=np.clip(x,0.01,0.99)
    return x

def convert_to_logits(x):
    return logits(cut(x))

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
        # one can do include or exclude operations here.
        if v.name.split('/')[1] in var_keep_dic:
            print("Variables restored: %s" % v.name)
            variables_to_restore.append(v)

    return variables_to_restore



