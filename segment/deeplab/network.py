import tensorflow as tf
import tensorflow.contrib.layers as layers
slim = tf.contrib.slim
from resnet import resnet_v2, resnet_utils
class Network:
    def __init__(self , config):
        """
        初始化模型的配置  
        """
                
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)
        self.start_learning_rate =config['learning_rate']
        self.lambda_l2 = config['lambda_l2']
        # self.stddev = config.stddev
        self.batch_size = config['batch_size']
        # self.use_fp16 = config.use_fp16
        self.class_num = config['class_num']
        self.params_dir = config['saver_directory']


        self.img_height = config['img_height']//config['scale']
        self.img_width = config['img_width']//config['scale']

        self.is_train = config['is_train']
        self.output_stride = config['output_stride']
        self.decay = config['learning_rate_decay']
        self.decay_step = config['decay_step']

        self.images = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width, 3)
                )

        self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width))

        self.labels_by_classes = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width, self.class_num))

        self.pred_images = tf.placeholder(
                dtype = tf.float32,
                shape = (None, self.img_height, self.img_width, 3)
                )

        print("Deeplab:")
        print("Batch size: {}, Output stride: {}\nNumber of class: {}".format(self.batch_size , self.output_stride,
            self.class_num))
        # self.miou = tf.placeholder(tf.float32, shape=(), name="valid_miou")
        # self.valid_miou = tf.summary.scalar("valid_miou", self.miou )
    def loss(self):
        """
        损失函数  
        """
        return tf.add( tf.add_n(tf.get_collection('losses'))  , 
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) , name = "total_loss")

    def add_to_loss(self , predicts ):

        valid_indices = tf.where(tf.not_equal(self.labels , -1))

        valid_labels = tf.gather_nd(params = self.labels_by_classes, 
            indices=valid_indices)

        valid_logits = tf.gather_nd(params=predicts, 
            indices=valid_indices)


        cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits,
                                                                     labels=valid_labels)
        cross_entropy_tf = tf.reduce_mean(cross_entropies)
        tf.add_to_collection("losses", cross_entropy_tf)

    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        """
        将每一stage的最后一层加入损失函数 
        """

        flatten_vis = tf.reshape(self.vis_mask, [batch_size, -1])
        flatten_labels = tf.multiply( tf.reshape(labels, [batch_size, -1]) ,flatten_vis)
        flatten_predicts = tf.multiply(tf.reshape(predicts, [batch_size, -1]) , flatten_vis)
        # flatten_labels = tf.reshape(labels, [batch_size, -1])
        # flatten_predicts = tf.reshape(predicts, [batch_size, -1])
        # print(flatten_labels , flatten_predicts)
        with tf.name_scope(name) as scope:
            euclidean_loss = tf.sqrt(tf.reduce_sum(
              tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
            # print(euclidean_loss)
            euclidean_loss_mean = tf.reduce_mean(euclidean_loss,
                name='euclidean_loss_mean')

        tf.add_to_collection("losses", euclidean_loss_mean)

    def train_op(self, total_loss, global_step):
        """
        Optimizer
        """
        self._loss_summary(total_loss)

        self.learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                   self.decay_step, self.decay, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum = 0.9)
        grads = optimizer.compute_gradients(total_loss)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)    

        return apply_gradient_op

    ###存储与恢复参数checkpoint####
    def save(self, sess, saver, filename, global_step):
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print ("Save params at " + path)

    def restore(self, sess, saver, filename):
        print ("Restore from previous model: ", self.params_dir+filename)
        saver.restore(sess, self.params_dir+filename)
    ###存储loss 和 中间的图片进入log###
    def _loss_summary(self, loss):
        tf.summary.scalar(loss.op.name + "_raw", loss)

    def _mIOU_summary(self, predicts):
        p = tf.reshape(tf.argmax(predicts, axis=3) , [self.batch_size,-1])
        l = tf.reshape(self.labels, [self.batch_size,-1])
        miou,self.update_op = tf.metrics.mean_iou(p, l, num_classes=self.class_num)
        # tf.summary.scalar('miou', miou)
    def _image_summary(self, x, channels):
        x = tf.cast(x, tf.float32)
        def sub(batch, idx):  
            name = x.op.name
            if channels>1:
                tmp = x[batch, :, :, idx] * 255
                tmp = tf.expand_dims(tmp, axis = 2)
                tmp = tf.expand_dims(tmp, axis = 0)
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
            else:

                tmp = x[batch, :, :] * 255
                tmp = tf.expand_dims(tmp, axis = 2)
                tmp = tf.expand_dims(tmp, axis = 0)
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
        for idx in range(channels):
            sub(0, idx)

            
        # if (self.batch_size > 1):
        #   for idx in range(channels):
        #     # the first batch
        #     sub(0, idx)
        #     # the last batch
        #     sub(-1, idx)
        # else:
        #   for idx in range(channels):
        #     sub(0, idx)

    def _fm_summary(self, predicts):
      with tf.name_scope("fcn_summary") as scope:
          self._image_summary(self.labels, 1)
          tmp_predicts = tf.argmax(predicts , axis=3)
          self._image_summary(tmp_predicts, 1)


    @slim.add_arg_scope
    def atrous_spatial_pyramid_pooling(self,net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """

        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                               activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
            return net


    def deeplab_v3(self ):
        """
        # ImageNet mean statistics
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        """

        #~2000 expert project plumage images:
        _R_MEAN = 45.48
        _G_MEAN = 44.85
        _B_MEAN = 44.29

        # mean subtraction normalization
        inputs = self.images
        inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]

        is_train = self.is_train

        # inputs has shape - Original: [batch, 513, 513, 3]
        with slim.arg_scope(resnet_utils.resnet_arg_scope(self.lambda_l2, is_train)):
            # resnet = getattr(resnet_v2, args.resnet_model)
            _, end_points = resnet_v2.resnet_v2_50(inputs,
                                   self.class_num,
                                   is_training=is_train,
                                   global_pool=False,
                                   spatial_squeeze=False,
                                   output_stride=self.output_stride)

            with tf.variable_scope("DeepLab_v3"):

                # get block 4 feature outputs
                net = end_points['resnet_v2_50/block4']

                net = self.atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256)

                net = slim.conv2d(net, self.class_num, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

                size = tf.shape(inputs)[1:3]
                size = inputs.shape[1:3]
                # # resize the output logits to match the labels dimensions
                # #net = tf.image.resize_nearest_neighbor(net, size)
                net = tf.image.resize_bilinear(net, size)
                if is_train:
                    self.add_to_loss(net)
                    self._fm_summary(net)
                    self._mIOU_summary(net)
                return net


# def training_network(prediction , loss , train_op, sess,config):