import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class Pose_Estimation:
    def __init__(self , config,ori_img_width, ori_img_height, is_train = None):
        """
        Initialize the model with config and its hyperparameters
        """
        if is_train is not None:
            self.is_train = is_train
        else:
            self.is_train = config['is_train']
        self.is_grey = config['is_grey']
        self.network_name = config['network_name']

        self.cpu = '/cpu:0'     
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)

        self.batch_size = config["batch_size"]

        self.points_num = config["points_num"]
        self.params_dir = config["saver_directory"]

        self.scale = config['scale']
        self.output_stride = config['output_stride']
        self.img_height = ori_img_height // self.scale
        self.img_width = ori_img_width // self.scale

        self.fm_height = self.img_height//self.output_stride
        self.fm_width = self.img_width//self.output_stride 

        self.images = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.img_height, self.img_width, 3)
                )
        self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.fm_height, self.fm_width, self.points_num))
        self.coords = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.points_num * 2))
        self.vis_mask = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.fm_height, self.fm_width, self.points_num))

        self.dropout_rate = config['dropout_rate']
        self.lambda_l2 = config["l2"]

        self.point_names = config['point_names']
        self.nFeat = config['nfeats']
        ## configuration of hourglass
        if self.network_name == 'hourglass':
            self.nStack = _help_func_dict(config, 'nstacks')
            self.nLow = _help_func_dict(config, 'nlow')
            self.tiny = _help_func_dict(config, 'tiny')
            self.modif = False



        if self.is_train:
        # All learning rate decay is in `train_op(self, total_loss, global_step)`
            self.decay_restart = config["decay_restart"]
            if self.decay_restart is True:
                self.restart_decay_steps = config["first_decay_epoch"] * config["one_epoch_steps"]
                self.t_mul = _help_func_dict(config, 't_mul', 2.0)
                self.m_mul = _help_func_dict(config, 'm_mul', 1.0)            
            self.optimizer = _help_func_dict(config, 'optimizer', "adam")
            self.start_learning_rate =config["learning_rate"]
            self.exponential_decay_step = config["exponential_decay_epoch"] * config["one_epoch_steps"]
            self.learning_rate_decay = config["learning_rate_decay"]
            
                    ## Summary configuration
        self.weight_summary = _help_func_dict(config, 'weight_summary', False)
        self.filter_summary = _help_func_dict(config, 'filter_summary', False)
        self.result_summary = _help_func_dict(config, 'result_summary', False)

        print("\nInitialize the {} network.\n\tIs Training:{}\n\tInput shape: {}\n\tOutput shape: {}".format(self.network_name,
            self.is_train, self.images.shape.as_list(), self.labels.shape.as_list()))
        
        if self.is_train:
            print("#### configuration ######")
            print("Optimizer: {}\tStart Learning rate: {}\tdecay_restart: {}".format(self.optimizer,
             self.start_learning_rate, self.decay_restart) )

################### functions ##########
    def loss(self):
        """
        Loss function of all losses  
        """
        
        return tf.add( tf.add_n(tf.get_collection('losses')) , 
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name = "total_loss")

    
    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        """
        Euclidean loss of between predictions and labels.
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
        # add loss into summary
        self._loss_summary(total_loss)

        #####The learning rate decay method

        if self.decay_restart:
            # Cosine decay and restart
            # print("decayn restart: {}".format(self.restart_decay_steps))
            self.learning_rate = tf.train.cosine_decay_restarts(self.start_learning_rate, global_step,
             self.restart_decay_steps, t_mul = self.t_mul , m_mul = self.m_mul)
        else:
            # exponential_decay
            # print("expotineal decayn: {}".format(self.exponential_decay_step))
            self.learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                       self.exponential_decay_step, self.learning_rate_decay, staircase=True)

        ##### Select the optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        # grads = optimizer.compute_gradients(total_loss)
        # apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)  
          
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            apply_gradient_op = optimizer.minimize(total_loss, global_step)


        tf.summary.scalar("learning_rate", self.learning_rate, collections = ['train'])
        return apply_gradient_op

    ###存储与恢复参数checkpoint####
    def save(self, sess, saver, filename, global_step):
        """
        Goal: save network parameters to file in a folder.
        """
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print ("Save params at " + path)

    def restore(self, sess, saver, filename):
        """
        Goal: Restore parameters for network to file in a folder.
        """
        print ("Restore from previous model: ", self.params_dir+filename)
        saver.restore(sess, self.params_dir+filename)
    ###存储loss 和 中间的图片进入log###
    def _loss_summary(self, loss):
        """
        Save the loss to summary log
        """
        with tf.device(self.cpu):
            with tf.name_scope('train_loss'):
                tf.summary.scalar(loss.op.name + "_raw", loss,collections=['train'])

        # self.valid_loss_summary = tf.summary.scalar("Validation Loss", loss )

    def _image_summary(self, x, channels,is_train):
        def sub(batch, idx):
            name = x.op.name
            tmp = x[batch, :, :, idx] * 255
            tmp = tf.expand_dims(tmp, axis = 2)
            tmp = tf.expand_dims(tmp, axis = 0)
            if is_train:
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100,collections = ['train'])
            else:
                tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100,collections = ['valid'])
        if (self.batch_size > 1):
          for idx in range(channels):
            # the first batch
            sub(0, idx)
            # the last batch
            sub(-1, idx)
        else:
          for idx in range(channels):
            sub(0, idx)

    def _fm_summary(self, predicts, is_train = True):
      with tf.name_scope("fcn_summary") as scope:
          self._image_summary(self.labels, self.points_num,is_train)
          tmp_predicts = tf.nn.relu(predicts)
          self._image_summary(tmp_predicts, self.points_num,is_train)

    def _create_summary(self, pred_img):
        """
        Goals: Create summaries for training and validation

        e.g, loss for training and valdation.
        Accuracy
        Images:
        """
        self.point_acc = tf.placeholder(dtype = tf.float32,
         shape = (self.points_num,))
        self.valid_loss = tf.placeholder(dtype = tf.float32)

        self.ave_pts_diff = tf.placeholder(dtype = tf.float32)
        # self.point_pck = tf.placeholder(dtype = tf.float32,
        #  shape = (self.points_num,))        
        with tf.device(self.cpu):
            with tf.name_scope('train'):
                if self.result_summary:
                    self._fm_summary(pred_img)
                # tf.summary.scalar("Training loss", self.loss_value, collections = ['train'])
            tf.summary.scalar("Valid_loss", self.valid_loss, collections = ['valid'])
            tf.summary.scalar("Valid_ave_accuracy", self.ave_pts_diff, collections = ['valid'])
            with tf.name_scope('valid'):
                for i in range(self.points_num):
                     tf.summary.scalar("points/" + self.point_names[i], 
                        self.point_acc[i], collections = ['valid'])
        self.valid_summary = tf.summary.merge_all('valid')    
    
    def _weights_hist_summary(self):
        """
        Goal: add weights and bias histogram into tensorboard
        """
        if self.weight_summary:
            train_vars = tf.trainable_variables(scope="")

            weights = [w for w in train_vars if "weights" in w.name]
            for weight in weights:
                # print(weight.name)
                tf.summary.histogram(weight.name, weight, collections = ['train'])

            biases = [w for w in train_vars if "biases" in w.name]
            for bias in biases:
                tf.summary.histogram(bias.name, bias, collections = ['train'])

        # self.weights_summary = tf.summary.merge_all('weight') 

    def _mid_layers_summary(self, mid_layers):
        """
        Goal: Add output of mid layers into image summary
        """
        if self.filter_summary:
            cols = 16
            for mid in mid_layers:
                img = mid[0,...]
                img = tf.transpose(img, [2,0,1])

                rows = img.shape[0]//cols



                n, nrows, ncols = img.shape
                w = cols*ncols
                h = rows*nrows
                img = tf.reshape(img,[h//int(nrows), -1, nrows, ncols])
                img = tf.transpose(img, [0,2,1,3])
                img = tf.reshape(img, [h, w])   

                img = tf.expand_dims(img, axis = 2)
                img = tf.expand_dims(img, axis = 0)            
                #print(img.shape)
                tf.summary.image("mid/"+mid.name, img, max_outputs = 100,collections = ['train'])

            f1 = tf.trainable_variables(scope="")[0]
            f1 = tf.transpose(f1, [3,0,1,2])
            tf.summary.image("filter/"+f1.name, f1, max_outputs = 100,collections = ['train'])
                # for i in range(img.shape[-1]):
                #     img_mini =  img[0,:,:,i]
                #     img_mini = tf.expand_dims(img_mini, axis = 2)
                #     img_mini = tf.expand_dims(img_mini, axis = 0)
                #     tf.summary.image("mid/"+mid.name+"/{}".format(i), img_mini, max_outputs = 100,collections = ['train'])
    def inference_pose_vgg(self):

        # center_map = self.center_map
        lm_cnt = self.points_num
        # if self.is_train:
        #     image = self.images
        # else:
        #     image = self.pred_images
        if self.is_grey is True:
            self.images = tf.image.rgb_to_grayscale(self.images)

        image = self.images
        with tf.variable_scope('PoseNet'):
            # pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
            conv1_1 = layers.conv2d(image, 64, 3, 1, activation_fn=None, scope='conv1_1')
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
            conv1_2 = tf.nn.relu(conv1_2)
            pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
            conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None, scope='conv2_1')
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
            conv2_2 = tf.nn.relu(conv2_2)
            pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
            conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None, scope='conv3_1')
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
            conv3_2 = tf.nn.relu(conv3_2)
            conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3')
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
            conv3_4 = tf.nn.relu(conv3_4)
            pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
            conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1, activation_fn=None, scope='conv4_1')
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
            conv4_2 = tf.nn.relu(conv4_2)
            conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
            conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
            conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM')
            conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
            conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM')
            conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
            conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM')
            conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
            conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM')
            conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
            conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM')
            conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
            conv5_2_CPM = layers.conv2d(conv5_1_CPM, lm_cnt, 1, 1, activation_fn=None, scope='conv5_2_CPM')
            concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
            Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1, activation_fn=None, scope='Mconv1_stage2')
            Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
            Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1, activation_fn=None, scope='Mconv2_stage2')
            Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
            Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1, activation_fn=None, scope='Mconv3_stage2')
            Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
            Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1, activation_fn=None, scope='Mconv4_stage2')
            Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
            Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1, activation_fn=None, scope='Mconv5_stage2')
            Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
            Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1, activation_fn=None, scope='Mconv6_stage2')
            Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
            Mconv7_stage2 = layers.conv2d(Mconv6_stage2, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage2')
            concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
            Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1, activation_fn=None, scope='Mconv1_stage3')
            Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
            Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1, activation_fn=None, scope='Mconv2_stage3')
            Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
            Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1, activation_fn=None, scope='Mconv3_stage3')
            Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
            Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1, activation_fn=None, scope='Mconv4_stage3')
            Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
            Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1, activation_fn=None, scope='Mconv5_stage3')
            Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
            Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1, activation_fn=None, scope='Mconv6_stage3')
            Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
            Mconv7_stage3 = layers.conv2d(Mconv6_stage3, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage3')
            concat_stage4 = tf.concat(axis=3, values=[Mconv7_stage3, conv4_7_CPM])
            Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 7, 1, activation_fn=None, scope='Mconv1_stage4')
            Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
            Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 7, 1, activation_fn=None, scope='Mconv2_stage4')
            Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
            Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 7, 1, activation_fn=None, scope='Mconv3_stage4')
            Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
            Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 7, 1, activation_fn=None, scope='Mconv4_stage4')
            Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
            Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 128, 7, 1, activation_fn=None, scope='Mconv5_stage4')
            Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
            Mconv6_stage4 = layers.conv2d(Mconv5_stage4, 128, 1, 1, activation_fn=None, scope='Mconv6_stage4')
            Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
            Mconv7_stage4 = layers.conv2d(Mconv6_stage4, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage4')
            concat_stage5 = tf.concat(axis=3, values=[Mconv7_stage4, conv4_7_CPM])
            Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 7, 1, activation_fn=None, scope='Mconv1_stage5')
            Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
            Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 7, 1, activation_fn=None, scope='Mconv2_stage5')
            Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
            Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 7, 1, activation_fn=None, scope='Mconv3_stage5')
            Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
            Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 7, 1, activation_fn=None, scope='Mconv4_stage5')
            Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
            Mconv5_stage5 = layers.conv2d(Mconv4_stage5, 128, 7, 1, activation_fn=None, scope='Mconv5_stage5')
            Mconv5_stage5 = tf.nn.relu(Mconv5_stage5)
            Mconv6_stage5 = layers.conv2d(Mconv5_stage5, 128, 1, 1, activation_fn=None, scope='Mconv6_stage5')
            Mconv6_stage5 = tf.nn.relu(Mconv6_stage5)
            Mconv7_stage5 = layers.conv2d(Mconv6_stage5, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage5')
            concat_stage6 = tf.concat(axis=3, values=[Mconv7_stage5, conv4_7_CPM])
            Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 7, 1, activation_fn=None, scope='Mconv1_stage6')
            Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
            Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 7, 1, activation_fn=None, scope='Mconv2_stage6')
            Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
            Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 7, 1, activation_fn=None, scope='Mconv3_stage6')
            Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
            Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 7, 1, activation_fn=None, scope='Mconv4_stage6')
            Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
            Mconv5_stage6 = layers.conv2d(Mconv4_stage6, 128, 7, 1, activation_fn=None, scope='Mconv5_stage6')
            Mconv5_stage6 = tf.nn.relu(Mconv5_stage6)
            Mconv6_stage6 = layers.conv2d(Mconv5_stage6, 128, 1, 1, activation_fn=None, scope='Mconv6_stage6')
            Mconv6_stage6 = tf.nn.relu(Mconv6_stage6)
            Mconv7_stage6 = layers.conv2d(Mconv6_stage6, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage6')
            # print(conv5_2_CPM)
            # print(Mconv7_stage6)
            # print(self.labels)
            if self.is_train:
                self.add_to_euclidean_loss(self.batch_size, conv5_2_CPM, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage2, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage3, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage4, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage5, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage6, self.labels, 'st')
        # self.add_to_accuracy(Mconv7_stage6)
        self._create_summary(Mconv7_stage6)
        return Mconv7_stage6





    def cpm_vgg(self):

        # center_map = self.center_map
        is_train = self.is_train
        lm_cnt = self.points_num
        # if is_train:
        #     image = self.images
        # else:
        #     image = self.pred_images

        if self.is_grey is True:
            self.images = tf.image.rgb_to_grayscale(self.images)

        image = self.images
        with tf.variable_scope('CPM'):
            # print("lambda : {} keep prob: {} ".format(self.lambda_l2 , self.keep_prob))
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
            # pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
            conv1_1 = layers.conv2d(image, 64, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv1_1')
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, weights_regularizer=regularizer,activation_fn=None, scope='conv1_2')
            conv1_2 = tf.nn.relu(conv1_2)
            # return conv1_2
            # Pool
            pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
            conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv2_1')
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = layers.conv2d(conv2_1, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv2_2')
            conv2_2 = tf.nn.relu(conv2_2)
            pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
            conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_1')
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = layers.conv2d(conv3_1, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_2')
            conv3_2 = tf.nn.relu(conv3_2)
            conv3_3 = layers.conv2d(conv3_2, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_3')
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_4 = layers.conv2d(conv3_3, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_4')
            conv3_4 = tf.nn.relu(conv3_4)
            pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
            conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_1')
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = layers.conv2d(conv4_1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_2')
            conv4_2 = tf.nn.relu(conv4_2)
            ### add for different receptive field.
            conv4_3 = layers.conv2d(conv4_2, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_3')
            conv4_3 = tf.nn.relu(conv4_3) 
            
            if self.output_stride ==16:
                pool4_stage1 = layers.max_pool2d(conv4_3, 2, 2)
               
                conv5_1 = layers.conv2d(pool4_stage1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_1')
                conv5_1 = tf.nn.relu(conv5_1)
                conv5_2 = layers.conv2d(conv5_1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_2')
                conv5_2 = tf.nn.relu(conv5_2)
                conv5_3 = layers.conv2d(conv5_2, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_3')
                conv5_3 = tf.nn.relu(conv5_3) 
               
                conv4_3_CPM = layers.conv2d(conv5_3, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
            else:
                conv4_3_CPM = layers.conv2d(conv4_3, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')

            # conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
            conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
            conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_4_CPM')
            conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
            conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_5_CPM')
            conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
            conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_6_CPM')
            conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
            conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_7_CPM')
            conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
            conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_1_CPM')
            conv5_1_CPM = tf.nn.relu(conv5_1_CPM)

            # conv5_1_CPM = tf.nn.dropout(conv5_1_CPM, self.keep_prob)
            conv5_1_CPM = tf.layers.dropout(conv5_1_CPM,
             rate = self.dropout_rate, training = is_train)

            conv5_2_CPM = layers.conv2d(conv5_1_CPM, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_2_CPM')
            concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
            Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage2')
            Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
            Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage2')
            Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
            Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage2')
            Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
            Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage2')
            Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
            Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage2')
            Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
            Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage2')
            Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)

            # Mconv6_stage2 = tf.nn.dropout(Mconv6_stage2, self.keep_prob)
            Mconv6_stage2 = tf.layers.dropout(Mconv6_stage2,
             rate = self.dropout_rate, training = is_train)

            Mconv7_stage2 = layers.conv2d(Mconv6_stage2, lm_cnt, 1, 1, weights_regularizer=regularizer,activation_fn=None, scope='Mconv7_stage2')
            concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
            Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage3')
            Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
            Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage3')
            Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
            Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage3')
            Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
            Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage3')
            Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
            Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage3')
            Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
            Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage3')
            Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
            # Mconv6_stage3 = tf.nn.dropout(Mconv6_stage3, self.keep_prob)
            Mconv6_stage3 = tf.layers.dropout(Mconv6_stage3,
             rate = self.dropout_rate, training = is_train)
            Mconv7_stage3 = layers.conv2d(Mconv6_stage3, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage3')
            concat_stage4 = tf.concat(axis=3, values=[Mconv7_stage3, conv4_7_CPM])
            Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage4')
            Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
            Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage4')
            Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
            Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage4')
            Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
            Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage4')
            Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
            Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage4')
            Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
            Mconv6_stage4 = layers.conv2d(Mconv5_stage4, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage4')
            Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
            # Mconv6_stage4 = tf.nn.dropout(Mconv6_stage4, self.keep_prob)
            Mconv6_stage4 = tf.layers.dropout(Mconv6_stage4,
             rate = self.dropout_rate, training = is_train)

            Mconv7_stage4 = layers.conv2d(Mconv6_stage4, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage4')
            concat_stage5 = tf.concat(axis=3, values=[Mconv7_stage4, conv4_7_CPM])
            Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage5')
            Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
            Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage5')
            Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
            Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage5')
            Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
            Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage5')
            Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
            Mconv5_stage5 = layers.conv2d(Mconv4_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage5')
            Mconv5_stage5 = tf.nn.relu(Mconv5_stage5)
            Mconv6_stage5 = layers.conv2d(Mconv5_stage5, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage5')
            Mconv6_stage5 = tf.nn.relu(Mconv6_stage5)
            # Mconv6_stage5 = tf.nn.dropout(Mconv6_stage5, self.keep_prob)
            Mconv6_stage5 = tf.layers.dropout(Mconv6_stage5,
             rate = self.dropout_rate, training = is_train)

            Mconv7_stage5 = layers.conv2d(Mconv6_stage5, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage5')
            concat_stage6 = tf.concat(axis=3, values=[Mconv7_stage5, conv4_7_CPM])
            Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage6')
            Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
            Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 7, 1, weights_regularizer=regularizer,activation_fn=None, scope='Mconv2_stage6')
            Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
            Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage6')
            Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
            Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage6')
            Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
            Mconv5_stage6 = layers.conv2d(Mconv4_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage6')
            Mconv5_stage6 = tf.nn.relu(Mconv5_stage6)
            Mconv6_stage6 = layers.conv2d(Mconv5_stage6, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage6')
            Mconv6_stage6 = tf.nn.relu(Mconv6_stage6)
            # Mconv6_stage6 = tf.nn.dropout(Mconv6_stage6, self.keep_prob)
            Mconv6_stage6 = tf.layers.dropout(Mconv6_stage6,
             rate = self.dropout_rate, training = is_train)
            Mconv7_stage6 = layers.conv2d(Mconv6_stage6, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage6')
            # print(conv5_2_CPM)
            # print(Mconv7_stage6)
            # print(self.labels)
            if is_train:
                self.add_to_euclidean_loss(self.batch_size, conv5_2_CPM, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage2, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage3, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage4, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage5, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage6, self.labels, 'st')
        # self.add_to_accuracy(Mconv7_stage6)
        self._create_summary(Mconv7_stage6)

        self._mid_layers_summary([conv2_2,conv3_4,conv4_3,Mconv6_stage5])
        self._weights_hist_summary()
        return Mconv7_stage6

################### Stacked hourglass#####################

    def add_loss_stacked_hourglass(self, predicts):
        """
        Goal Add losses between predictions and labels

        params:
            predict shape: (batch, nStack, height, width, channel)
        self.labels shape: (batch, height, width, channel)
        self.vis_mask shape: (batch, height, width, channel)
        """
        stacked_visMaps = tf.stack([self.vis_mask] * self.nStack, axis = 1, name = "stacked_visMaps")
        stacked_labels = tf.stack([self.labels] * self.nStack, axis = 1, name = "stacked_labels")

        output_mask = tf.multiply(predicts , stacked_visMaps)
        # self.gtMaps_mask = tf.multiply(self.gtMaps , self.visMaps)


        with tf.name_scope("hourglass") as scope:
            loss = tf.reduce_mean(tf.square(tf.subtract(output_mask , stacked_labels)))


        tf.add_to_collection("losses", loss)


    def unblockshaped_withc(self,tensor, h, w):
        """
        input 
        Return an array of shape (h, w, c) where
        h * w = arr.size //  c

        If arr is of shape (n, nrows, ncols, c), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols,c  = tensor.shape

        tensor = tf.reshape(tensor,[h//int(nrows), -1, nrows, ncols, c])
        tensor = tf.transpose(tensor, [0,2,1,3,4])
        tensor = tf.reshape(tensor, [h, w, c]) 
        return tensor

    def _filter_summary_hg(self):
        """
        Goal: Add output of mid layers into image summary
        """
        if self.filter_summary:
            train_vars = tf.trainable_variables(scope="")
            weights = [w for w in train_vars if "weights" in w.name]
            f1 = weights[0]
            f1 = tf.transpose(f1, [3,0,1,2])

            tf.summary.image("filter/"+f1.name, f1, max_outputs = 100,collections = ['train'])
            # unblock the 4D to 3D
            cols = 16
            rows = f1.shape[0]//cols
            n, nrows, ncols, c = f1.shape
            w = cols*ncols
            h = rows*nrows
                
            f1 = self.unblockshaped_withc(f1, h, w) 

            f1 = tf.expand_dims(f1, axis = 0)      

            f1 = tf.image.resize_images(f1,  [f1.shape[0]*20, f1.shape[1]*20])
            tf.summary.image("filter/"+f1.name, f1, max_outputs = 100,collections = ['train'])


    def hourglass(self):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """

        if self.is_grey is True:
            self.images = tf.image.rgb_to_grayscale(self.images)

        images = self.images
        with tf.name_scope('hourglass'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(images, [[0,0],[2,2],[2,2],[0,0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters= 64, kernel_size = 6, strides = 2, name = 'conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut = 128, name = 'r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128
                if self.tiny:
                    r3 = self._residual(pool1, numOut=self.nFeat, name='r3')
                else:
                    r2 = self._residual(pool1, numOut= int(self.nFeat/2), name = 'r2')
                    r3 = self._residual(r2, numOut= self.nFeat, name = 'r3')
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack
            if self.tiny:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')

                        drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, name = 'll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.points_num, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.points_num, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], ll[0], r3], name = 'merge')
                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name= 'll')
                            if self.modif:
                                # TEST OF BATCH RELU
                                out[i] = self._conv_bn_relu(ll[i], self.points_num, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.points_num, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], ll[i], sum_[i-1]], name= 'merge')
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat,1,1, 'VALID', 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.points_num, 1,1, 'VALID', 'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.points_num, 1,1, 'VALID', 'out')
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
                else:
                    return tf.stack(out, axis= 1 , name = 'final_output')   
            else:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        # print(r3.shape)
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        # print("hr_glass 1")
                        # print(hg[0])
                        drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1,1, 'VALID', name = 'conv')
                        ll_[0] =  self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                        # print(ll_[0])
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.points_num, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.points_num, 1, 1, 'VALID', 'out')
                        #Yichen:Add result to out and sum array.
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')

                        sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                    
                    # Yichen : In the middle    
                    for i in range(1, self.nStack -1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name= 'conv')
                            ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                            if self.modif:
                                out[i] = self._conv_bn_relu(ll[i], self.points_num, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.points_num, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[0]], name= 'merge')

                    with tf.name_scope('stage_' + str(self.nStack -1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.is_train, name = 'dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat, 1, 1, 'VALID', 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.points_num, 1,1, 'VALID', 'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.points_num, 1,1, 'VALID', 'out')
                if self.modif:
                    #add losses and return the prediction.
                    predicts = tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
                else:
                    predicts = tf.stack(out, axis= 1 , name = 'final_output')
                self.add_loss_stacked_hourglass(predicts)

        self._create_summary(predicts[:,self.nStack-1,...])
        self._filter_summary_hg()
        return predicts[:,self.nStack-1,...]

                    
    def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs          : Input Tensor (Data Type : NHWC)
            filters     : Number of filters (channels)
            kernel_size : Size of kernel
            strides     : Stride
            pad             : Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name            : Name of the block
        Returns:
            conv            : Output Tensor (Convolved Input)
        """
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            if self.weight_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections = ['train'])
            return conv            

    def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs          : Input Tensor (Data Type : NHWC)
            filters     : Number of filters (channels)
            kernel_size : Size of kernel
            strides     : Stride
            pad             : Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name            : Name of the block
        Returns:
            norm            : Output Tensor
        """
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.is_train)
            if self.weight_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections = ['train'])
            return norm

    def _conv_block(self, inputs, numOut, name = 'conv_block'):
        """ Convolutional Block
        Args:
            inputs  : Input Tensor
            numOut  : Desired output number of channel
            name    : Name of the block
        Returns:
            conv_3  : Output Tensor
        """
        if self.tiny:
            with tf.name_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.is_train)
                pad = tf.pad(norm, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                return conv
        else:
            with tf.name_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.is_train)
                    conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.is_train)
                    pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                    conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.is_train)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                return conv_3


    def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
        """ Skip Layer
        Args:
            inputs  : Input Tensor
            numOut  : Desired output number of channel
            name    : Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
                return conv             
    
    def _residual(self, inputs, numOut, name = 'residual_block'):
        """ Residual Unit
        Args:
            inputs  : Input Tensor
            numOut  : Number of Output Features (channels)
            name    : Name of the block
        """
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.modif:
                return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
            else:
                return tf.add_n([convb, skipl], name = 'res_block')
    
    def _hourglass(self, inputs, n, numOut, name = 'hourglass'):
        """ Hourglass Module
        Args:
            inputs  : Input Tensor
            n       : Number of downsampling step
            numOut  : Number of Output Features (channels)
            name    : Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name = 'up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
            low_1= self._residual(low_, numOut, name = 'low_1')
            
            if n > 0:
                low_2 = self._hourglass(low_1, n-1, numOut, name = 'low_2')
            else:
                low_2 = self._residual(low_1, numOut, name = 'low_2')
                
            low_3 = self._residual(low_2, numOut, name = 'low_3')
            # print("low3 ", low_3.shape)

            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(up_1)[1:3], name = 'upsampling')
            # print(up_1.shape, up_2.shape)
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2,up_1]), name='out_hg')
            else:
                return tf.add_n([up_2,up_1], name='out_hg')



def _help_func_dict(config,key, default_value = None):
    if key in config:
        return config[key]
    else:
        return default_value