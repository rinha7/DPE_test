import os, sys
import tensorflow as tf
import numpy as np
import random, cv2, operator, os

# configure flags

FLAGS = {}

FLAGS['method'] = 'WGAN-v24-cycleganD2'
FLAGS['mode_use_debug'] = False
FLAGS['num_exp'] = 736
FLAGS['num_gpu'] = '4'
FLAGS['sys_use_unix'] = True
FLAGS['sys_is_dgx'] = True

FLAGS['netD_init_method'] = 'var_scale' #var_scale, rand_uniform, rand_normal, truncated_normal
FLAGS['netD_init_weight'] = 1e-3
FLAGS['netD_base_learning_rate'] = 1e-5
FLAGS['netD_base_learning_decay'] = 75
FLAGS['netD_base_learning_decay_epoch'] = 75
FLAGS['netD_regularization_weight'] = 0
FLAGS['netD_times'] = 50
FLAGS['netD_times_grow'] = 1
FLAGS['netD_buffer_times'] = 50 #it depends on batch size
FLAGS['netD_init_times'] = 0
FLAGS['netG_init_method'] = 'var_scale' #var_scale, rand_uniform, rand_normal, truncated_normal
FLAGS['netG_init_weight'] = 1e-3
FLAGS['netG_base_learning_rate'] = 1e-5
FLAGS['netG_base_learning_decay'] = 75
FLAGS['netG_base_learning_decay_epoch'] = 75
FLAGS['netG_regularization_weight'] = 0
FLAGS['loss_source_data_term'] = 'l2' # l1, l2, PR, GD
FLAGS['loss_source_data_term_weight'] = 1e3
FLAGS['loss_constant_term'] = 'l2' # l1, l2, PR, GD
FLAGS['loss_constant_term_weight'] = 1e4
FLAGS['loss_photorealism_is_our'] =  True
FLAGS['loss_wgan_lambda'] = 10
FLAGS['loss_wgan_lambda_grow'] = 2.0
FLAGS['loss_wgan_lambda_ignore'] = 1
FLAGS['loss_wgan_use_g_to_one'] = False
FLAGS['loss_wgan_gp_times'] = 1
FLAGS['loss_wgan_gp_use_all'] = False
FLAGS['loss_wgan_gp_bound'] = 5e-2
FLAGS['loss_wgan_gp_mv_decay'] = 0.99

FLAGS['loss_data_term_use_local_weight'] = False
FLAGS['loss_constant_term_use_local_weight'] = False
FLAGS['data_csr_buffer_size'] = 1500
FLAGS['sys_use_all_gpu_memory'] = True
FLAGS['loss_pr'] = (FLAGS['loss_constant_term'] == 'PR' and FLAGS['loss_constant_term_weight'] > 0) or (FLAGS['loss_source_data_term'] == 'PR' and FLAGS['loss_source_data_term_weight'] > 0)
FLAGS['loss_heavy'] = (FLAGS['loss_constant_term_weight'] > 0)

FLAGS['data_augmentation_size'] = 8
FLAGS['data_use_random_pad'] = False
FLAGS['data_train_batch_size'] = 3
FLAGS['load_previous_exp']   = 0
FLAGS['load_previous_epoch'] = 0

FLAGS['process_run_first_testing_epoch'] = True
FLAGS['process_write_test_img_count'] = 498
FLAGS['process_train_log_interval_epoch'] = 20
FLAGS['process_test_log_interval_epoch'] = 2
FLAGS['process_max_epoch'] = 150

FLAGS['format_log_step'] = '%.3f'
FLAGS['format_log_value'] = '{:6.4f}'
if FLAGS['sys_use_unix']:
    FLAGS['path_char'] = '/'
    if FLAGS['sys_is_dgx']:
        FLAGS['path_data'] = '/dataset/LPGAN'
        FLAGS['path_result_root'] = '/dataset/LPGAN-Result/%03d-DGX-LPGAN'
    else:
        FLAGS['path_data'] = '/tmp3/nothinglo/dataset/LPGAN'
        FLAGS['path_result_root'] = '/tmp3/nothinglo/dataset/LPGAN-Result/%03d-DGX-LPGAN'
else:
    FLAGS['path_char'] = '\\'
    FLAGS['path_data'] = 'D:\\G\\LPGAN'
    FLAGS['path_result_root'] = 'D:\\LPGAN\\%03d-DGX-LPGAN'

FLAGS['path_result'] = FLAGS['path_result_root'] % FLAGS['num_exp']
FLAGS['load_path'] = FLAGS['path_result_root'] % FLAGS['load_previous_exp'] + FLAGS['path_char']
FLAGS['load_model_path']         = FLAGS['load_path'] + 'model'      + FLAGS['path_char'] + '%s.ckpt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_loss_path']    = FLAGS['load_path'] + 'train_netG_loss' + FLAGS['path_char'] + '%s.txt'  % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_indices_input_path'] = FLAGS['load_path'] + 'train_ind_input'  + FLAGS['path_char'] + '%s.txt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_indices_label_path'] = FLAGS['load_path'] + 'train_ind_label'  + FLAGS['path_char'] + '%s.txt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])

FLAGS['load_model_need'] = FLAGS['load_previous_exp'] > 0
FLAGS['process_epoch'] = 0
FLAGS['process_train_drop_summary_step'] = 5
FLAGS['process_test_drop_summary_step'] = 1
FLAGS['process_train_data_loader_count'] = (8 if FLAGS['sys_use_unix'] else 4) if FLAGS['loss_pr'] else 2

# data
FLAGS['data_input_ext'] = '.tif'
FLAGS['data_input_dtype']   = np.uint8
FLAGS['data_label_dtype']   = np.uint8
FLAGS['data_compute_dtype'] = np.float32
FLAGS['data_image_size'] = 512
FLAGS['data_image_channel'] = 3
FLAGS['process_random_seed'] = 2
FLAGS['process_load_test_batch_capacity']  = (8 if FLAGS['sys_use_unix'] else 4) if FLAGS['loss_pr'] else 32
FLAGS['process_load_train_batch_capacity'] = (16 if FLAGS['sys_use_unix'] else 8) if FLAGS['loss_pr'] else 64

# net
FLAGS['net_gradient_clip_value'] = 1e8

# input
FLAGS['folder_input'] = FLAGS['path_data'] + FLAGS['path_char'] + 'input' + FLAGS['path_char']
FLAGS['folder_label'] = FLAGS['path_data'] + FLAGS['path_char'] + 'label' + FLAGS['path_char']
FLAGS['folder_label_HDR'] = FLAGS['path_data'] + FLAGS['path_char'] + 'label_HDR' + FLAGS['path_char']

FLAGS['folder_csrs'] = FLAGS['path_data'] + FLAGS['path_char'] + 'csrs' + FLAGS['path_char']
FLAGS['folder_csrs_rgb'] = FLAGS['path_data'] + FLAGS['path_char'] + 'csrs_rgb' + FLAGS['path_char']
FLAGS['txt_test']   = FLAGS['path_data'] + FLAGS['path_char'] + 'test.txt'
FLAGS['txt_train_input']  = FLAGS['path_data'] + FLAGS['path_char'] + 'train_input.txt'
FLAGS['txt_train_label']  = FLAGS['path_data'] + FLAGS['path_char'] + 'train_label.txt'
if FLAGS['sys_use_unix']:
    FLAGS['folder_test_csrs'] = FLAGS['folder_csrs']
else:
    FLAGS['folder_test_csrs'] = FLAGS['path_data'] + FLAGS['path_char'] + 'test_csrs' + FLAGS['path_char']

# output
FLAGS['folder_model']     = FLAGS['path_result'] + FLAGS['path_char'] + 'model' + FLAGS['path_char']
FLAGS['folder_log']       = FLAGS['path_result'] + FLAGS['path_char'] + 'log' + FLAGS['path_char']
FLAGS['folder_weight']    = FLAGS['path_result'] + FLAGS['path_char'] + 'weight' + FLAGS['path_char']
FLAGS['folder_test_img']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_img' + FLAGS['path_char']
FLAGS['folder_train_ind_input'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_ind_input' + FLAGS['path_char']
FLAGS['folder_train_ind_label'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_ind_label' + FLAGS['path_char']

FLAGS['folder_test_netG_loss']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_loss' + FLAGS['path_char']
FLAGS['folder_test_netG_psnr1']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_psnr1' + FLAGS['path_char']
FLAGS['folder_test_netG_psnr2']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_psnr2' + FLAGS['path_char']
FLAGS['folder_train_netG_loss'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_netG_loss' + FLAGS['path_char']

FLAGS['netG_mat'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-netG.mat' % FLAGS['num_exp']
FLAGS['netD_mat'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-netD.mat' % FLAGS['num_exp']
FLAGS['txt_log'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-log.txt' % FLAGS['num_exp']

# Loss 및 다양한 측정 지표들을 정의한 함수들

# Generator의 loss 측정에 사용되는 photorealism loss

def tf_photorealism_loss(img, df, i, is_our):
    rec_t = df.rect[i]
    img_t = img[i, rec_t[0]:rec_t[1], rec_t[2]:rec_t[3], :]
    img_t = tf.image.rot90(img_t, 4 - tf.floordiv(df.rot[i], 2))
    img_t = tf.cond(tf.equal(tf.mod(df.rot[i], 2), 0), lambda: img_t, lambda: tf.image.flip_left_right(img_t))
    img_t = tf.transpose(img_t, [1, 0, 2])
    img_r = tf.reshape(img_t, [-1, 3])
    h = rec_t[1] - rec_t[0]
    w = rec_t[3] - rec_t[2]
    k = tf.cast((h - 2) * (w - 2), tf.float32)
    if is_our:
        epsilon1 = 1
        e = tf.constant(np.sqrt(epsilon1), dtype=tf.float32, shape=[1, 3])
        img_r = tf.concat(0, [img_r, e])
        mat_t_r = df.csr_mat_r[i]
        mat_t_g = df.csr_mat_g[i]
        mat_t_b = df.csr_mat_b[i]
        img_r_b, img_r_g, img_r_r = tf.split(1, 3, img_r)
        d_mat_r = tf.sparse_tensor_dense_matmul(mat_t_r, img_r_r)
        d_mat_g = tf.sparse_tensor_dense_matmul(mat_t_g, img_r_g)
        d_mat_b = tf.sparse_tensor_dense_matmul(mat_t_b, img_r_b)
        result_r = tf.reduce_sum(img_r_r * d_mat_r)
        result_g = tf.reduce_sum(img_r_g * d_mat_g)
        result_b = tf.reduce_sum(img_r_b * d_mat_b)
        result = tf.reduce_mean(tf.pack([result_r, result_b, result_g])) / k
    else:
        mat_t = df.csr_mat[i]
        d_mat = tf.sparse_tensor_dense_matmul(mat_t, img_r)
        result = tf.reduce_sum(img_r * d_mat) / (k * 3)
    return result

def tf_imgradient(tensor):
    B, G, R = tf.unpack(tensor, axis=-1)
    tensor = tf.pack([R, G, B], axis=-1)
    tensor = tf.image.rgb_to_grayscale(tensor)
    #tensor = tensor * 255;
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    #tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    fx = tf.nn.conv2d(tensor, sobel_x_filter, strides=[1,1,1,1], padding='VALID')
    fy = tf.nn.conv2d(tensor, sobel_y_filter, strides=[1,1,1,1], padding='VALID')
    g = tf.sqrt(tf.square(fx) + tf.square(fy))
    return g

def img_L2_loss(img1, img2, use_local_weight):
    if use_local_weight:
        w = -tf.log(tf.cast(img2, tf.float64) + tf.exp(tf.constant(-99, dtype=tf.float64))) + 1
        w = tf.cast(w * w, tf.float32)
        return tf.reduce_mean(w * tf.square(tf.sub(img1, img2)))
    else:
        return tf.reduce_mean(tf.square(tf.sub(img1, img2)))

def img_L1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(tf.sub(img1, img2)))

def img_GD_loss(img1, img2):
    img1 = tf_imgradient(tf.pack([img1]))
    img2 = tf_imgradient(tf.pack([img2]))
    return tf.reduce_mean(tf.square(tf.sub(img1, img2)))

def regularization_cost(net_info):
    cost = 0
    for w, p in zip(net_info.weights, net_info.parameter_names):
        if p[-2:] == "_w":
            cost = cost + (tf.nn.l2_loss(w))
    return cost

def flatten_list(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten_list(x))
    else:
        result.append(xs)
    return result