import tensorflow as tf
import math
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


class low_rank_gaussian():
    def __init__(self,mean,L,sigma):
        self.mean=mean
        self.dim=mean.get_shape().as_list()[1]
        self.L=L
        self.rank=L.get_shape().as_list()[1]
        self.sigma=sigma
        self.batch_size=mean.get_shape().as_list()[0]
        
        
    def log_prob(self,x,test_mode=False):
        return self.fast_loglikelighood(x, self.mean, self.L, self.sigma, test_mode)
      
    def cov(self):
        var_diag = tf.ones(self.dim)*(self.sigma**2)
        return tf.matrix_diag(var_diag)+tf.linalg.matmul(a=self.L,b=self.L,transpose_a=False,transpose_b=True)
        
    def sample(self,amount=1):
        eps_z=tf.random.normal(shape=(self.batch_size,self.rank))
        eps=tf.random.normal(shape=(self.batch_size,self.dim))
        return self.mean+tf.matmul(eps_z,tf.matrix_transpose(self.L))+eps*self.sigma
      
    def sample_from_noise(self):
        eps_z=tf.random.normal(shape=(self.batch_size,self.rank))
        eps=tf.random.normal(shape=(self.batch_size,self.dim))
        return tf.matmul(eps_z,tf.matrix_transpose(self.L))+eps*self.sigma
    
#     def logdet(self):
#         cov=self.cov()
#         return tf.linalg.logdet(cov)
    def logdet(self):
        noise=self.sigma**2
        batch_size=self.batch_size
        dim=self.dim
        inverse_noise=1./noise
        rank=self.rank
        L=self.L
        inner = tf.matmul(L,L,transpose_a=True,transpose_b=False)*inverse_noise+tf.matrix_diag(tf.ones([rank]))
        logdet=tf.linalg.logdet(inner)+dim*tf.log(noise)
        return logdet
    
  
    def fast_loglikelighood(self,x,mean,L,sigma,test_mode=False):
        noise=sigma**2
        batch_size=self.batch_size
        dim=self.dim
        inverse_noise=1./noise
        rank=self.rank
        inner = tf.matmul(L,L,transpose_a=True,transpose_b=False)*inverse_noise+tf.matrix_diag(tf.ones([rank]))
        inner_inverse = tf.linalg.inv(inner)
        inverse_cov =  inverse_noise*tf.matrix_diag(tf.ones([dim]))-inverse_noise**2*tf.matmul(L,tf.matmul(inner_inverse,L,transpose_a=False,transpose_b=True))
        log_det=tf.linalg.logdet(inner)+dim*tf.log(noise)
        diff=x-mean
        y=tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(diff,inverse_cov),axis=1),tf.expand_dims(diff,axis=2)))
        logZ =  (dim/2)*tf.log(2* math.pi)+0.5*log_det
        loglikelihood=-0.5*y-logZ
        if test_mode:
            return loglikelihood,y,logZ,diff,log_det
        else:
            return loglikelihood
        
# class low_rank_gaussian():
#     def __init__(self,mean,rank,sigma):
#         self.mean=mean
#         self.rank=rank
#         self.dim=mean.get_shape().as_list()[1]
#         self.L=tf.get_variable("L", [self.dim,rank], tf.float32)
#         self.sigma=sigma
#         self.batch_size=mean.get_shape().as_list()[0]
        
        
#     def log_prob(self,x,test_mode=False):
#         return self.fast_loglikelighood(x, self.mean, self.L, self.sigma, test_mode)
      
#     def cov(self):
#         var_diag = tf.ones(self.dim)*(self.sigma**2)
#         return tf.matrix_diag(var_diag)+tf.linalg.matmul(a=self.L,b=self.L,transpose_a=False,transpose_b=True)
        
#     def sample(self):
#         eps_z=tf.random.normal(shape=(self.batch_size,self.rank))
#         eps=tf.random.normal(shape=(self.batch_size,self.dim))
#         return self.mean+tf.matmul(eps_z,tf.matrix_transpose(self.L))+eps*self.sigma
        
#     def sample_from_noise(self):
#         eps_z=tf.random.normal(shape=(self.batch_size,self.rank))
#         eps=tf.random.normal(shape=(self.batch_size,self.dim))
#         return tf.matmul(eps_z,tf.matrix_transpose(self.L))+eps*self.sigma
  
#     def fast_loglikelighood(self,x,mean,L,sigma,test_mode=False):
#         noise=sigma**2
#         batch_size=self.batch_size
#         dim=self.dim
#         inverse_noise=1./noise
#         rank=L.get_shape().as_list()[1]
#         inner = tf.matmul(L,L,transpose_a=True,transpose_b=False)*inverse_noise+tf.matrix_diag(tf.ones([rank]))
#         inner_inverse = tf.linalg.inv(inner)
#         inverse_cov =  inverse_noise*tf.matrix_diag(tf.ones([dim]))-inverse_noise**2*tf.matmul(L,tf.matmul(inner_inverse,L,transpose_a=False,transpose_b=True))
#         log_det=tf.linalg.logdet(inner)+dim*tf.log(noise)
#         diff=x-mean
#         y=tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(diff,inverse_cov),axis=1),tf.expand_dims(diff,axis=2)))
#         logZ =  (dim/2)*tf.log(2* math.pi)+0.5*log_det
#         loglikelihood=-0.5*y-logZ
#         if test_mode:
#             return loglikelihood,y,logZ,diff,log_det
#         else:
#             return loglikelihood


class lower_triangle_gaussian():
    def __init__(self,mean,chol_lower):
        self.mean=mean
        self.batch_size=mean.get_shape().as_list()[0]
        self.dim=mean.get_shape().as_list()[1]
        self.chol_lower=chol_lower
    
    def sample(self,amount=1):
        eps=tf.random.normal(shape=(self.batch_size,self.dim))
        return self.mean+tf.matmul(eps,self.chol_lower)
    
    def logdet(self):
        return 2*tf.reduce_sum(tf.log(tf.diag_part(self.chol_lower)))
    
    def cov(self):
        return tf.matmul(self.chol_lower,self.chol_lower,transpose_a=False,transpose_b=True)
    
    def log_prob(self,x):
        dim=self.dim
        log_det=self.logdet()
        diff=x-self.mean
        inverse_lower=tf.matrix_inverse(self.chol_lower)
        inverse_cov=tf.matmul(inverse_lower,inverse_lower,transpose_a=True,transpose_b=False)
        y=tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(diff,inverse_cov),axis=1),tf.expand_dims(diff,axis=2)))
        logZ =  (dim/2)*tf.log(2* math.pi)+0.5*log_det
        loglikelihood=-0.5*y-logZ
        return loglikelihood

