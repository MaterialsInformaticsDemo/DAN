# multiple kernel variant of MMD
# based on tensorflow 2
# cao bin, HKUST, China, binjacobcao@gmail.com
# free to charge for academic communication

import tensorflow as tf

# based on rbf kernel functions
@tf.function
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    # source: source domain data
    # target: target domain data
    # kernel_num: kernel number of MK_MMD; Eq.(2)
    n_s = tf.shape(source)[0]
    n_t = tf.shape(target)[0]
    # total number of samples 
    n = n_s + n_t
    total = tf.concat([source, target], axis=0)
    # (1, n, m)
    row_expand = tf.expand_dims(total, axis=0)
    # (n, 1, m)
    column_expand = tf.expand_dims(total, axis=1)
    # (row_expand - column_expand) ** 2 is (n, n, m), broadcasting is applied
    # sum on feature dimension, viz., calculate L2_distance as (n, n)
    L2_distance = tf.reduce_sum((row_expand - column_expand) ** 2, axis=2)
    # evaluate the lengthscales
    length_scale = tf.reduce_sum(L2_distance) / tf.cast(n ** 2 - n, tf.float32)
    length_scale /= kernel_mul ** (kernel_num // 2)
    length_scale_list = [length_scale * (kernel_mul ** i) for i in range(kernel_num)]
    # define the multi-kernels
    M_kernel = [tf.exp(-L2_distance / i) for i in length_scale_list]
    # return n*n data correlation matrix
    return tf.reduce_sum(M_kernel, axis=0)

# define the function of MK_MMD
# solve MMD directlt, ref : TCA in my implementation : https://github.com/MaterialsInformaticsDemo/TCA
@tf.function
def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5):
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num)
    n_s = tf.shape(source)[0]
    n_t = tf.shape(target)[0]
    XX = tf.reduce_sum(kernels[:n_s, :n_s]) / tf.cast(n_s ** 2, tf.float32)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:]) / tf.cast(n_t ** 2, tf.float32)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:]) / tf.cast(n_s * n_t, tf.float32)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s]) / tf.cast(n_s * n_t, tf.float32)
    
    return XX + YY - XY - YX