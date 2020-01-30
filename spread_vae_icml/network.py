import tensorflow as tf
import numpy as np
from tensorflow import layers
from math import ceil, sqrt

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops._ops
import ops.resnet

def spectral_norm(w, c=0.1, iteration=10):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = c* w / sigma
       w_norm = tf.reshape(w_norm, w_shape)


   return w_norm


def injective_conv(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        

        return conv

def injective_conv_spread(x, reuse=tf.AUTO_REUSE):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,64,64,3))
        net = injective_conv(x, 3, name='en_conv3')
    return tf.layers.flatten(net)

def injective_conv_spread_sigmoid(x, reuse=tf.AUTO_REUSE):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,64,64,3))
        net = tf.nn.sigmoid(injective_conv(x, 3, name='en_conv3'))
    return tf.layers.flatten(net)

def injective_deep_conv_spread(x, reuse=tf.AUTO_REUSE):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,64,64,3))
    net = tf.nn.elu(injective_conv(x, 32, name='en_conv1'))
    net = tf.nn.elu(injective_conv(net, 32, name='en_conv2'))
    net = injective_conv(net, 3, name='en_conv3')
    return tf.layers.flatten(net)


def one_one_conv(x):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,64,64,3))
    h,w,c = x.shape[1:]
    w_init = np.float32(np.linalg.qr(np.random.randn(c,c))[0])
    w = tf.get_variable("yo", initializer=w_init)
    w = tf.reshape(w, [1,1,c,c])
    out = tf.nn.conv2d(x, w, strides=[1,1,1,1],  padding='SAME')
    return tf.layers.flatten(out)


def injective_conv_spread_sperate(x, reuse=tf.AUTO_REUSE):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,64,64,3))
        x1=tf.reshape(x[:,:,:,0],(-1,64,64,1))
        x2=tf.reshape(x[:,:,:,1],(-1,64,64,1))
        x3=tf.reshape(x[:,:,:,2],(-1,64,64,1))

        net1 = injective_conv(x1, 1, name='en_conv1')
        net2 = injective_conv(x2, 1, name='en_conv2')
        net3 = injective_conv(x3, 1, name='en_conv3')
        net=tf.stack([net1,net2,net3],axis=-1)
        print(net)
    return tf.layers.flatten(net)

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)
def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
        
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
        
def small_conv_encoder(opt, x, is_training=True, reuse=False):
    if len(x.shape.as_list())==2:
        x=tf.reshape(x,(-1,28,28,1))
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
    with tf.variable_scope("encoder", reuse=reuse):

        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
        net = tf.reshape(net, [opt['batch_size'], -1])
        net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
        gaussian_params = linear(net, 2 * opt['z_dim'], scope='en_fc4')

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :opt['z_dim']]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, opt['z_dim']:])

    return mean, stddev
    
    
def small_conv_decoder(opt, z, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("decoder", reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
        net = tf.reshape(net, [opt['batch_size'], 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [opt['batch_size'], 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
               scope='de_bn3'))

        out = tf.nn.sigmoid(deconv2d(net, [opt['batch_size'], 28, 28, 1], 4, 4, 2, 2, name='de_dc4'))
        out = tf.reshape(out,[-1,784])
    return out


def large_conv_encoder(opt, inputs, is_training=True, reuse=False, batch_norm=True,  data_format="channels_last"):

    with tf.variable_scope("encoder", reuse=reuse):

        inputs = tf.reshape(inputs, [opt['batch_size'], opt['image_dim'], opt['image_dim'], opt['channels']])
        num_units = opt['num_units']#1024
        num_layers = opt['encoder_layers']#4
        layer_x = inputs
        for i in range(num_layers):
            scale = 2 ** (num_layers - i - 1)
            channels = int(num_units / scale)
            layer_x = layers.conv2d(layer_x, channels, kernel_size=5, strides=(2, 2), padding='same',
                                data_format=data_format)
            print('encoder:', i)
            print(layer_x)
            if batch_norm:
                layer_x = layers.batch_normalization(layer_x, training=is_training, reuse=reuse)
            layer_x = tf.nn.relu(layer_x)

        layer_x = tf.contrib.layers.flatten(layer_x)
        mean = layers.dense(layer_x, opt['z_dim'])
        sigma = tf.exp(layers.dense(layer_x, opt['z_dim']))
    return mean, sigma


def large_conv_decoder(opt, z, batch_norm=True, is_training = True, reuse=None, data_format="channels_last"):
    with tf.variable_scope("decoder", reuse=reuse):
        output_shape = [opt['image_dim'], opt['image_dim'], opt['channels']]
        num_units = opt['num_units']
        num_layers = opt['decoder_layers']

        height = int(output_shape[0] / 2 ** num_layers)
        width = int(output_shape[1] / 2 ** num_layers)

        h0 = layers.dense(z, num_units * height * width)
        h0 = tf.reshape(h0, [-1, height, width, num_units])
        h0 = tf.nn.relu(h0)
        layer_x = h0
        print('decoder_layer_x:', layer_x)
        for i in range(num_layers):
            scale = 2 ** (i + 1)
            # out_shape = [batch_size, height * scale,width * scale, num_units / scale]
            channels = int(num_units / scale)
            layer_x = layers.conv2d_transpose(layer_x, channels, kernel_size=5, strides=2, padding='same',
                                          data_format=data_format)
            print('decoder:', i)
            print(layer_x)

            if batch_norm:
                layer_x = layers.batch_normalization(layer_x, training=is_training, reuse=reuse)
            layer_x = tf.nn.relu(layer_x)

        layer_x = layers.conv2d_transpose(layer_x, 3, kernel_size=5, strides=1, padding='same', data_format=data_format)
        print('decoder_final_layer')
        print(layer_x)
        layer_x=tf.reshape(layer_x, [-1,opt['all_pic_dim']])
    return tf.nn.sigmoid(layer_x)


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def ff_encoder(opt, x, is_training=True, reuse=False):
    z_dim=opt['z_dim']
    h_dim=opt['h_dim']
    non_linearity=leaky_relu
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        h = tf.layers.dense(x, h_dim, activation=non_linearity)
        #h = tf.layers.dense(h, h_dim, activation=non_linearity)
        h = tf.layers.dense(h, h_dim, activation=non_linearity)
        mean_and_sigma = tf.layers.dense(h, 2*z_dim)
        mean = mean_and_sigma[:, :z_dim]
        sigma = tf.exp(mean_and_sigma[:, z_dim:])

    return mean, sigma
    
    
def ff_decoder(opt, z, is_training=True, reuse=False):
    h_dim=opt['h_dim']
    x_dim=opt['x_dim']
    non_linearity=leaky_relu
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        h=tf.layers.dense(z, h_dim, activation=non_linearity)
        #h=tf.layers.dense(h, h_dim, activation=non_linearity)
        h=tf.layers.dense(h, h_dim, activation=non_linearity)      
        x_delta= tf.layers.dense(h, x_dim,activation=tf.nn.sigmoid)
    return x_delta



def resnet_encoder(opts, input, is_training=False, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        input = tf.reshape(input, [opts['batch_size'], opts['image_dim'], opts['image_dim'], opts['channels']])
        output_dim=opts['z_dim']
        output = ops.resnet.OptimizedResBlockEnc1(opts,input,output_dim)
        output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res2', resample='down', reuse=reuse, is_training=is_training)
        output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res3', resample='down', reuse=reuse, is_training=is_training)
        output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res4', resample='down', reuse=reuse, is_training=is_training)
        output = ops._ops.non_linear(output,'relu')
        output = tf.reduce_mean(output, axis=[1,2])
        output = ops.linear.Linear(opts, output, np.prod(output.get_shape().as_list()[1:]), 2*output_dim, scope='hid_final')
        mean = output[:, :output_dim]
        sigma = tf.exp(output[:, output_dim:])
    return mean,sigma


def resnet_decoder(opts, input, is_training=False, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        input_dim = np.prod(input.get_shape().as_list()[1:])
        output = ops.linear.Linear(opts,input,input_dim,4*4*input_dim,scope='hid0/lin')
        output = tf.reshape(output, [-1, 4, 4, input_dim])
        output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res1', resample='up', reuse=reuse, is_training=is_training)
        output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res2', resample='up', reuse=reuse, is_training=is_training)
        output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res3', resample='up', reuse=reuse, is_training=is_training)
        output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res3', resample='up', reuse=reuse, is_training=is_training)
        output = ops.batchnorm.Batchnorm_layers(opts, output, 'hid3/bn', is_training, reuse)
        output = ops._ops.non_linear(output,'relu')
        output = ops.conv2d.Conv2d(opts, output, input_dim, 3, 3, scope='hid_final/conv', init='normilized_glorot')
        layer_x=tf.reshape(output, [-1,opts['all_pic_dim']])
        print('layer_x',layer_x)
    return tf.nn.sigmoid(layer_x)


