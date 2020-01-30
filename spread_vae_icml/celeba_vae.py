import os
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow import layers
from utils import *
from network import *
from distributions import *
np.random.seed(46)


def qz(opt):
    with tf.variable_scope('qz', reuse=tf.AUTO_REUSE):
        aux_mean = tf.zeros([opt['batch_size'], opt['z_dim']])
        Qa = tfd.MultivariateNormalDiag(loc=aux_mean, name='qz')
    return Qa


def py_x(x, sigma, opt):
    x_dim = opt['x_dim']
    with tf.variable_scope('py_x', reuse=tf.AUTO_REUSE):
        if opt['dis'] == 'learned_cov':
            L_init=tf.get_variable("L", [x_dim,opt['noise_rank']], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.0,stddev=0.01))
            L=opt['L_norm']*L_init/tf.sqrt(tf.linalg.det(tf.linalg.matmul(L_init,L_init,transpose_a=True,transpose_b=False)))
            if opt['det_norm']:
                py_x_dis=low_rank_gaussian(x,L,sigma)
            else:
                py_x_dis=low_rank_gaussian(x,L_init,sigma)

        else:
            sigma_diag=tf.ones_like(x)*(sigma)
            py_x_dis = tfd.MultivariateNormalDiag(loc=x, scale_diag=sigma_diag, name='py_x')

    return py_x_dis


def pz_y(x, opt, is_training=True, reuse=False):
    if opt['net']=='res':
        mean, sigma = resnet_encoder(opt, x, is_training, reuse)
    else:
        mean, sigma = large_conv_encoder(opt, x, is_training, reuse)
    
    pz_y_dis = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
    return pz_y_dis


def qx_z(qz_sample, opt, is_training=True, reuse=False):
    if opt['net']=='res':
        x_delta = resnet_decoder(opt, qz_sample, is_training, reuse)
    else:
        x_delta = large_conv_decoder(opt, qz_sample, is_training, reuse)
    return x_delta




def main(opt, dataset):
    os_save(opt)

    with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
        X = tf.placeholder("float32", [opt['batch_size'], opt['image_dim'], opt['image_dim'], opt['channels']],name='X')
        data = tf.reshape(X, [opt['batch_size'], opt['x_dim']])
        sigma = tf.placeholder("float32", ())
        noise = tf.placeholder("float32", [100, opt['z_dim']])

        kl_qz_dis = qz(opt)

        ### p(y|x)
        kl_py_x_dis = py_x(data, sigma, opt)
        kl_py_x_sample = kl_py_x_dis.sample()

        ### p(z|y)
        if opt['spread']:
            kl_pz_y_dis = pz_y(kl_py_x_sample, opt)
        else:
            kl_pz_y_dis = pz_y(data, opt)

        kl_pz_y_sample = kl_pz_y_dis.sample()

        ### q(y|z)
        kl_x_delta = qx_z(kl_pz_y_sample, opt)
        kl_qy_z_dis = py_x(kl_x_delta, sigma, opt)

        ### 3 log probability
        ### logqz
        kl_logqz = kl_qz_dis.log_prob(kl_pz_y_sample)

        if opt['low_variance'] == True:
            kl_logqy_z = kl_qy_z_dis.log_prob(data)
        else:
            kl_logqy_z = kl_qy_z_dis.log_prob(kl_py_x_sample)

        ### logpz_y
        kl_logpz_y = kl_pz_y_dis.log_prob(kl_pz_y_sample)
        kl_loss = tf.reduce_mean(kl_logpz_y - kl_logqz - kl_logqy_z)
        noise_loss = -kl_loss

        global_step = tf.Variable(0, trainable=False)
        if opt['lr_decay']:
            starter_learning_rate = opt['kl_lr']
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   opt['decay_period'], opt['decay_rate'], staircase=True)
            optimizer_kl = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            optimizer_kl = tf.train.AdamOptimizer(learning_rate=opt['kl_lr'])
        
        if opt['optimizer']=='adam':
            optimizer_noise = tf.train.AdamOptimizer(learning_rate=opt['noise_lr'])
        elif opt['optimizer']=='rmsprop':
            optimizer_noise = tf.train.RMSPropOptimizer(learning_rate=opt['noise_lr'])
        else:
            optimizer_noise = tf.train.GradientDescentOptimizer(learning_rate=opt['noise_lr'])
        

        kl_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/encoder') + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/decoder')
        noise_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/py_x')


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            kl_op = optimizer_kl.minimize(kl_loss, var_list=kl_vars, global_step = global_step)
  
        if opt['dis'] in ['learned', 'learned_cov']:
            noise_op = optimizer_noise.minimize(noise_loss, var_list=noise_vars)
        else:
            pass

        ### define test sample
        qz_sample_test = kl_qz_dis.sample()
        x_delta_test = tf.reshape(qx_z(qz_sample_test, opt, is_training=False, reuse=True),[-1, opt['image_dim'], opt['image_dim'], opt['channels']])
        x_con_test = tf.reshape(qx_z(noise, opt, is_training=False, reuse=True),[-1, opt['image_dim'], opt['image_dim'], opt['channels']])

        saver = tf.train.Saver(tf.global_variables())

        

    sess = tf.Session()

    batches_num = int(dataset.num_points / opt['batch_size'])
    train_size = dataset.num_points

    noise_name = './other/random_noise_' + str(opt['z_dim']) + '.npy'
    exists = os.path.isfile(noise_name)
    if exists:
        noise_input = np.load(noise_name)
    else:
        noise_input = np.random.normal(size=(100, opt['z_dim']))
        np.save(noise_name, noise_input)

    variables = tf.global_variables()
    sess.run(tf.variables_initializer(variables, name='init'))


    kl_sgx = opt['kl_sgx']
    num_sqrt = 10
    kl_list = []

    if opt['reconstruction']
    if opt['restore']:
        restore_variable=opt['restore_variable']
        restore_path = opt['restore_path']
        variables_to_restore = get_variables_to_restore(variables, restore_variable)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, restore_path)
        print("Model restore from path: %s" % restore_path)
        

        con_samples = sess.run(x_con_test, {sigma: kl_sgx, noise: noise_input})
        name=opt['path']+'init_samples.png'
        celeba_save_many(con_samples, num_sqrt,name)

    else:
        pass

    for t in range(1, opt['training_epoch'] + 1):
        
        print('epoch: ', t)
        if t>1:
            for it in range(0, batches_num ):
    #         for it in range(0, 10):
    #             print(it)
                data_ids = np.random.choice(train_size, opt['batch_size'], replace=False)
                X_mb = dataset.data[data_ids].astype(np.float)
                _, kl_loss_v = sess.run([kl_op, kl_loss], {sigma: kl_sgx, X: X_mb})

                print(kl_loss_v)
                kl_list.append(kl_loss_v)

            plt.plot(kl_list)
            plt.savefig(opt['path']+str(t)+'_epoch_kl.png')
            plt.close()
            con_samples = sess.run(x_con_test, {sigma: kl_sgx, noise: noise_input})
            name=opt['path']+str(t)+'_samples.png'
            celeba_save_many(con_samples, num_sqrt,name)

            

            if opt['save']:
                if t in opt['save_epoch']:
                    save_path = saver.save(sess, opt['path'] + str(t) + '_epoch_kl.ckpt')
                    print("Model saved in path: %s" % save_path)
            else:
                pass
                
        noise_kl_list = []
        if t>=opt['noise_start_epoch'] and (t-1) % opt['noise_learning_period'] == 0:
            print('noise learning')
            if opt['dis'] in ['learned', 'learned_cov']:
                for it in range(0, opt['noise_epoch'] * batches_num):
                    data_ids = np.random.choice(train_size, opt['batch_size'], replace=False)
                    X_mb = dataset.data[data_ids].astype(np.float)
                    _, kl_loss_v = sess.run([noise_op, kl_loss], {sigma: kl_sgx, X: X_mb})
                    if kl_loss_v<5000:
                        noise_kl_list.append(kl_loss_v)
                    print('noise epoch',it)
                    print(kl_loss_v)

                plt.plot(noise_kl_list)
                plt.savefig(opt['path'] + str(t) + '_epoch_noise_kl.png')
                plt.close()

                learned_cov = sess.run(kl_py_x_dis.cov(), {sigma: kl_sgx, X: X_mb})
                learned_cov = learned_cov.reshape(opt['x_dim'], opt['x_dim'])
                cov_name = opt['path']+'cov.npy'
                np.save(cov_name,learned_cov)
            
            else:
                pass



        if opt['annealing']:
            if t > opt['start_annealing_epoch'] and t % opt['annealing_period'] == 0:
                if opt['kl_sgx'] > opt['annealing_end_std']:
                    opt['kl_sgx'] = opt['kl_sgx'] * 0.9
                print(opt['kl_sgx'])




    sess.close()
