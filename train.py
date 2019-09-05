import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# functions
def kaiming(shape):
    return (tf.truncated_normal(shape)*tf.sqrt(2/float(shape[0])))

def xavier(shape):
    return (tf.truncated_normal(shape,0.0, stddev=tf.sqrt(1.0/(0.5*(shape[2]+shape[3])))))

def relu(x):
  return tf.maximum(x, 0)

def CNN_1(_x, _w, _b):
    _conv1 = tf.nn.conv2d(_x, tf.clip_by_norm(_w['c1'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv1 = tf.nn.bias_add(_conv1, _b['c1'])
    _conv1 = relu(_conv1)

    _conv2 = tf.nn.conv2d(_conv1, tf.clip_by_norm(_w['c2'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv2 = tf.nn.bias_add(_conv2, _b['c2'])
    _conv2 = relu(_conv2)

    _conv3 = tf.nn.conv2d(_conv2, tf.clip_by_norm(_w['c3'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv3 = tf.nn.bias_add(_conv3, _b['c3'])
    _conv3 = relu(_conv3)

    _conv3 = tf.add(_conv3, _conv1)

    _conv4 = tf.nn.conv2d(_conv3, tf.clip_by_norm(_w['c4'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv4 = tf.nn.bias_add(_conv4, _b['c4'])
    _conv4 = relu(_conv4)

    _conv5 = tf.nn.conv2d(_conv4, tf.clip_by_norm(_w['c5'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv5 = tf.nn.bias_add(_conv5, _b['c5'])
    _conv5 = relu(_conv5)

    _conv5 = tf.add(_conv5, _conv3)

    _conv6 = tf.nn.conv2d(_conv5, tf.clip_by_norm(_w['c8'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv6 = tf.nn.bias_add(_conv6, _b['c8'])

    return _conv6

def CNN_shapes(_x, _w, _b, _nrd):
    _ins = [[] for i in range(_nrd+1)]
    _outs = [[] for i in range(_nrd)]

    _conv1 = tf.nn.conv2d(_x, tf.clip_by_norm(_w[0]['in'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
    _conv1 = tf.nn.bias_add(_conv1, _b[0]['in'])
    _conv1 = relu(_conv1)

    _ins[0] =_conv1

    for i in range(_nrd):
        _conv2 = tf.nn.conv2d(_ins[i], tf.clip_by_norm(_w[i+1]['b1'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
        _conv2 = tf.nn.bias_add(_conv2, _b[i+1]['b1'])
        _conv2 = relu(_conv2)

        _conv3 = tf.nn.conv2d(_conv2, tf.clip_by_norm(_w[i+1]['b2'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
        _conv3 = tf.nn.bias_add(_conv3, _b[i+1]['b2'])
        _conv3 = relu(_conv3)

        _ins[i+1] = tf.add(_conv3, _ins[i])

        _conv4 = tf.nn.conv2d(_ins[i+1], tf.clip_by_norm(_w[i+1]['out'], 1, [2, 3]), strides=[1, 1, 1, 1], padding='VALID')
        _outs[i] = tf.nn.bias_add(_conv4, _b[i+1]['out'])

    return _outs


def _parse_function(example_proto):
    features = {"image_gt2d": tf.FixedLenFeature([], tf.string),
                "image_gt3d": tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)

    image_gt2d = tf.decode_raw((parsed_features['image_gt2d']), tf.float64)
    image_gt2d = tf.reshape(image_gt2d, [1, 2*npts])
    image_gt2d = tf.cast(image_gt2d, tf.float32)

    image_gt3d = tf.decode_raw((parsed_features['image_gt3d']), tf.float64)
    image_gt3d = tf.reshape(image_gt3d, [1, 3*npts])
    image_gt3d = tf.cast(image_gt3d, tf.float32)

    return image_gt2d, image_gt3d


def rotation_decomposition(_esti):
  _, u, v = tf.svd(_esti, full_matrices=True)
  _rot = tf.matmul(u, v, transpose_b=True)

  _det = tf.matrix_determinant(_rot)

  _zero = tf.tile(tf.reshape(0., [1, 1, 1]), [tf.shape(_esti)[0], tf.shape(_esti)[1], 1])
  _one = tf.tile(tf.reshape(1., [1, 1, 1]), [tf.shape(_esti)[0], tf.shape(_esti)[1], 1])
  _di = tf.concat([_one, _zero, _zero, _zero, _one, _zero, _zero, _zero, tf.reshape(_det,[tf.shape(_esti)[0], tf.shape(_esti)[1], 1])], 2)
  _di = tf.reshape(_di, (tf.shape(_esti)[0], tf.shape(_esti)[1], 3, 3))

  _rot1 = tf.matmul(u, _di)
  _rot2 = tf.matmul(_rot1, v, transpose_b=True)
  return _rot2


tf.set_random_seed(1234)

npts = 8
datalen = 637

nframe = 32
training_epochs = 100

nrk = 12
nrd = 12

# for rotation matrix
nins = np.array([(2) * npts])
nouts = np.array([3 * 3])
nfts = np.array([2048])

# for estimating weights
nins = np.concatenate((nins, np.array([2 * npts])), axis=None)
nouts = np.concatenate((nouts, np.array([nrk])), axis=None)
nfts = np.concatenate((nfts, np.array([1024])), axis=None)

# for estimating shapes
nins = np.concatenate((nins, np.array([2 * npts])), axis=None)
nouts = np.concatenate((nouts, np.array([3 * npts])), axis=None)
nfts = np.concatenate((nfts, np.array([32])), axis=None)

with tf.variable_scope("rotation") as scope:
    weightMatrix_r = {'c1': tf.Variable(xavier([1, 1, nins[0], nfts[0]])),
                      'c2': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c3': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c4': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c5': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c6': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c7': tf.Variable(xavier([1, 1, nfts[0], nfts[0]])),
                      'c8': tf.Variable(xavier([1, 1, nfts[0], nouts[0]]))}

    biasMatrix_r = {'c1': tf.Variable(kaiming([nfts[0]])),
                    'c2': tf.Variable(kaiming([nfts[0]])),
                    'c3': tf.Variable(kaiming([nfts[0]])),
                    'c4': tf.Variable(kaiming([nfts[0]])),
                    'c5': tf.Variable(kaiming([nfts[0]])),
                    'c6': tf.Variable(kaiming([nfts[0]])),
                    'c7': tf.Variable(kaiming([nfts[0]])),
                    'c8': tf.Variable(kaiming([nouts[0]]))}


with tf.variable_scope("scale") as scope:
    weightMatrix_s = {'c1': tf.Variable(xavier([1, 1, nins[1], nfts[1]])),
                      'c2': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c3': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c4': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c5': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c6': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c7': tf.Variable(xavier([1, 1, nfts[1], nfts[1]])),
                      'c8': tf.Variable(xavier([1, 1, nfts[1], nouts[1]]))}

    biasMatrix_s = {'c1': tf.Variable(kaiming([nfts[1]])),
                    'c2': tf.Variable(kaiming([nfts[1]])),
                    'c3': tf.Variable(kaiming([nfts[1]])),
                    'c4': tf.Variable(kaiming([nfts[1]])),
                    'c5': tf.Variable(kaiming([nfts[1]])),
                    'c6': tf.Variable(kaiming([nfts[1]])),
                    'c7': tf.Variable(kaiming([nfts[1]])),
                    'c8': tf.Variable(kaiming([nouts[1]]))}


with tf.variable_scope("asimple") as scope:
    weightMatrix1 = [[] for i in range(nrk)]
    for i in range(nrk):
        weightMatrix1[i] = [[] for j in range(nrd + 1)]
        weightMatrix1[i][0] = {'in': tf.Variable(xavier([1, 1, nins[2], nfts[2]]))}
        for j in range(nrd):
            weightMatrix1[i][j + 1] = {'b1': tf.Variable(xavier([1, 1, nfts[2], nfts[2]])),
                                       'b2': tf.Variable(xavier([1, 1, nfts[2], nfts[2]])),
                                       'out': tf.Variable(xavier([1, 1, nfts[2], nouts[2]]))}

    biasMatrix1 = [[] for i in range(nrk)]
    for i in range(nrk):
        biasMatrix1[i] = [[] for j in range(nrd + 1)]
        biasMatrix1[i][0] = {'in': tf.Variable(kaiming([nfts[2]]))}
        for j in range(nrd):
            biasMatrix1[i][j + 1] = {'b1': tf.Variable(kaiming([nfts[2]])),
                                     'b2': tf.Variable(kaiming([nfts[2]])),
                                     'out': tf.Variable(kaiming([nouts[2]]))}

batch_size = nframe
batch_size_test = nframe

with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset('./dataset/pascal3d_aeroplane.tfrecords')
    dataset = dataset.map(_parse_function)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=5000)

    iterator = dataset.make_initializable_iterator()

    dataset_test = tf.data.TFRecordDataset('./dataset/pascal3d_aeroplane.tfrecords')
    dataset_test = dataset_test.map(_parse_function)

    dataset_test = dataset_test.repeat(2)
    dataset_test = dataset_test.batch(batch_size_test)

    iterator_test = dataset_test.make_initializable_iterator()

    _pts_2d, _pts_3d = iterator.get_next()
    _pts_2d = tf.cast(_pts_2d,tf.float32)
    _pts_3d = tf.cast(_pts_3d, tf.float32)

    _pts_2d = tf.reshape(_pts_2d, [-1, 2*npts])
    _pts_3d = tf.reshape(_pts_3d, [-1, 3*npts])

    _pts_2d_test, _pts_3d_test = iterator_test.get_next()
    _pts_2d_test = tf.cast(_pts_2d_test, tf.float32)
    _pts_3d_test = tf.cast(_pts_3d_test, tf.float32)

    _pts_2d_test = tf.reshape(_pts_2d_test, [-1, 2*npts])
    _pts_3d_test = tf.reshape(_pts_3d_test, [-1, 3*npts])

# training
in2dst = tf.reshape(_pts_2d, [nframe,1,1,2*npts])

out_2ds = [[] for i in range(nrk)]
for i in range(nrk):
    outs = CNN_shapes(in2dst, weightMatrix1[i], biasMatrix1[i], nrd)
    outs_list = tf.stack(outs, axis=1)
    outs_list = tf.divide(outs_list,
                        tf.tile(tf.sqrt(tf.reduce_sum(tf.square(outs_list), axis=4, keepdims=True)),
                                [1, 1, 1, 1, 3 * npts]))
    out_2ds[i] = tf.reshape(tf.transpose(outs_list,[1,0,2,3,4]),[nrd, 1,1,-1])
out_2ds_list = tf.stack(out_2ds, axis=0)

out_scale_t = tf.abs(CNN_1(in2dst, weightMatrix_s, biasMatrix_s))
out_scale_temp = tf.reshape(out_scale_t,[-1,nframe,nrk])
out_scale = out_scale_temp

out_2d_scale = tf.reduce_sum(tf.multiply(tf.tile(tf.reshape(tf.transpose(out_scale,[2,0,1]), [nrk, 1, 1, 1, nframe]), [1, nrd, 1, 1, 3 * npts]), out_2ds_list), axis=0)

rots_t = CNN_1(in2dst, weightMatrix_r, biasMatrix_r)
rots_dc = rotation_decomposition(tf.reshape(rots_t,[-1,nframe,3,3]))

tproj = tf.matmul(tf.transpose(tf.reshape(out_2d_scale, [nrd, nframe, 3, npts]),[0,1,3,2]),tf.tile(tf.reshape(rots_dc,[1,nframe,3,3]),[nrd,1,1,1]))
loss_proj = tf.reduce_mean(tf.square(tproj[:,:,:,0:2] - tf.tile(tf.transpose(tf.reshape(in2dst,[1,nframe,2,npts]),[0,1,3,2]),[nrd,1,1,1])))

shape_mtx = tf.reshape(tf.transpose(tf.reshape(out_2ds_list,[nrk, nrd,nframe,3,npts]),[0, 1, 4, 3,2]),[nrk, nrd,npts*3,nframe])
simshape = tf.svd(tf.matmul(tf.transpose(shape_mtx, [0, 1, 3, 2]), shape_mtx), compute_uv=False)
simshape = tf.reduce_sum(simshape, axis=2)
loss_lr = tf.reduce_mean(simshape)

reg_scale = tf.reduce_mean(tf.square(out_scale_temp))

# cost function
cost_all = (loss_proj * 800. + loss_lr * 80.) + reg_scale * 5.

# optimizers
optm_rotation = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_all, var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rotation'))
optm_others = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_all, var_list = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scale'),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='asimple')])

# test
in2dst_test = tf.reshape(_pts_2d_test,[nframe,1,1,2*npts])

out_2ds_test = [[] for i in range(nrk)]
for i in range(nrk):
    outs_test = CNN_shapes(in2dst_test, weightMatrix1[i], biasMatrix1[i], nrd)
    out_2ds_test_t = outs_test[nrd-1]
    out_2ds_test_t = tf.divide(out_2ds_test_t,
                        tf.tile(tf.sqrt(tf.reduce_sum(tf.square(out_2ds_test_t), axis=3, keepdims=True)), [1, 1, 1, 3 * npts]))
    out_2ds_test[i] = tf.reshape(out_2ds_test_t,[1,1,1,-1])

out_scale_test_t = tf.abs(CNN_1(in2dst_test, weightMatrix_s, biasMatrix_s))
out_scale_temp_test = tf.reshape(out_scale_test_t,[-1,nframe,nrk])
out_scale_test = out_scale_temp_test

out_2d_scale_test = tf.multiply(tf.tile(tf.reshape(out_scale_test[:,:,0],[-1,1,1,nframe]),[1,1,1,3*npts]), out_2ds_test[0])
for i in range(nrk-1):
    out_2d_scale_test = out_2d_scale_test + tf.multiply(tf.tile(tf.reshape(out_scale_test[:,:,i+1],[-1,1,1,nframe]),[1,1,1,3*npts]), out_2ds_test[i+1])

rots_test_t = CNN_1(tf.concat([in2dst_test],axis=3), weightMatrix_r, biasMatrix_r)
rots_dc_test = rotation_decomposition(tf.reshape(rots_test_t,[-1,nframe,3,3,]))

infer_rot_tests = [[] for i in range(nframe)]
for i in range(nframe):
    rotstn = rots_dc_test[:,i,:,:]

    infer_rot_test = tf.reshape(
        tf.transpose(tf.matmul(tf.transpose(tf.reshape(out_2d_scale_test[:,:,:,i*3*npts:(i+1)*3*npts], [-1, 3, npts]), [0, 2, 1]), rotstn), [0, 2, 1]),
        [-1, 1, 1, 3 * npts])
    infer_rot_tests[i] = infer_rot_test

## session
aconfig = tf.ConfigProto()
aconfig.gpu_options.allow_growth = True
aconfig.allow_soft_placement = True

sess = tf.Session(config=aconfig)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

# training
print ("Training start")
sess.run(iterator.initializer)
for epoch in range(training_epochs):
    total_batch = int(datalen / batch_size)

    for i in range(total_batch + 1):
        for j in range(25):
            sess.run([optm_rotation])
        sess.run([optm_others])

    print ("Training: [%d / %d]" % (epoch+1, training_epochs))

# for evaluation
sess.run(iterator_test.initializer)
saverdir = "./results/"
if not os.path.exists(saverdir):
    os.makedirs(saverdir)

test_batch = int((datalen) / batch_size_test)
for i in range(test_batch + 1):
    _infer_rot, _gt, = sess.run([infer_rot_tests, _pts_3d_test])
    savename = saverdir + str(i)
    sio.savemat(savename, mdict={'infer_rot': _infer_rot, 'gt': _gt})

print ("Training end")