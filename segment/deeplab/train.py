from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys
import os
import network
import metrics
from datetime import date


dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config , save_config



print('--Parsing Config File')
config_name = 'config_contour.cfg'

params = process_config(os.path.join(dirname, config_name))
save_config(os.path.join(dirname, config_name) , params['saver_directory'])


tf.reset_default_graph()
model = network.Network(params)
predict = model.deeplab_v3()

loss = model.loss()
train_op = model.train_op(loss, model.global_step)


##reading data###

train_csv = pd.read_csv(params['train_file'])



valid_csv = pd.read_csv(params['valid_file'])

if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)


# train_csv = train_csv[0:5]
# valid_csv = valid_csv[0:5]

print("Read training data ....")
train_data = data_input.plumage_data_input(train_csv,batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'])
print("Read valid data ....")
valid_data = data_input.plumage_data_input(valid_csv,batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'])


train_data_size = train_data.df_size
# print(params)
total_steps = (params['nepochs'] * train_data_size) //params['batch_size']



param_dir = params['saver_directory']
logdir = params['log_dir']
restore_file = params['restore_param_file']
save_filename = "{}_{}".format(str(date.today()) ,params['name'])
initialize = params['init']

summary_steps = total_steps // params['summary_interval']
valid_steps = total_steps // params['valid_interval']
saver_steps = total_steps // params['saver_interval']

print('Total steps: {}\nSum steps: {}, Valid steps: {}, Save steps: {}'.format(total_steps,
 summary_steps,valid_steps,saver_steps))

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    if os.listdir(param_dir) == [] or initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        sess.run(init_op)
        saver.restore(sess, restore_file)
#         model.restore(sess, saver, restore_file)
    sess.run(tf.local_variables_initializer())
    merged = tf.summary.merge_all()
#     logdir = config.logdir

    writer = tf.summary.FileWriter(logdir, sess.graph)
    valid_loss_ph = tf.placeholder(tf.float32, shape=(), name="valid_loss")
    loss_valid = tf.summary.scalar('loss_valid',valid_loss_ph ) 
    valid_acc_ph = tf.placeholder(tf.float32, shape=(), name="valid_acc")
    acc_valid = tf.summary.scalar('m_iou',valid_acc_ph ) 

    
    
    
    for i in range(total_steps):
        #训练阶段
        #以batch_size随机选择数据 迭代max_iteration次 
        tmp_global_step = model.global_step.eval()
        x_train,y_train = train_data.get_next_batch()
        feed_dict = {
                    model.images: x_train,
                    model.labels: np.argmax(y_train,axis = 3),
                    model.labels_by_classes: y_train
                    }
        sess.run(train_op, feed_dict=feed_dict)
        ###### Write training detail#####
        if (i+1) % summary_steps == 0:
            print("{} steps Loss: {}".format(i+1,sess.run(loss, feed_dict=feed_dict)))
            lear = model.learning_rate.eval()
#             print("\tGlobal steps and learning rates: {}  {}".format(tmp_global_step,lear))

            result_train=sess.run(predict, feed_dict=feed_dict)
            result_with_label = np.argmax(result_train,axis=3)
            gt_with_label = np.argmax(y_train,axis = 3)
            acc = np.sum(result_with_label == gt_with_label) / gt_with_label.size
            print("train pixel accuracy: ", acc)
            train_iou = metrics.segs_eval(result_with_label,gt_with_label,mode="miou")
            print("train iou :", train_iou)
            summary,_ = sess.run([merged,model.update_op], feed_dict=feed_dict)    
            writer.add_summary(summary, tmp_global_step)
            
        if (i+1) % valid_steps ==0:
            #write the validation result
            acc_list = np.array([])
            loss_list = np.array([])
            for i_df_valid in np.arange(0,valid_csv.shape[0],params["batch_size"]):
                x,y_valid = valid_data.get_next_batch_no_random()
                feed_dict = {
                    model.images: x,
                    model.labels: np.argmax(y_valid,axis = 3),
                    model.labels_by_classes: y_valid
                    }            
                _loss = sess.run(loss, feed_dict=feed_dict)

                result_mini = sess.run(predict, feed_dict=feed_dict)
                acc_iou = metrics.segs_eval(np.argmax(result_mini,axis = 3),np.argmax(y_valid,axis = 3),mode="miou")
                
                acc = np.sum(np.argmax(result_mini,axis=3) == np.argmax(y_valid,axis=3)) / np.argmax(y_valid,axis=3).size
                print("valid pixekofl accuracy: ", acc)
                
                acc_list = np.append(acc_list,acc_iou)
                loss_list = np.append(loss_list,_loss)

            print("\t VALIDATION {} steps: average acc and loss : {}  {}".format(i+1,np.mean(acc_list),np.mean(loss_list)))
#             miou = metrics.segs_eval(result_with_label,gt_with_label,mode="miou")
            writer.add_summary(sess.run(loss_valid, feed_dict={valid_loss_ph: np.mean(loss_list)}) , tmp_global_step)  
            writer.add_summary(sess.run(acc_valid, feed_dict={valid_acc_ph: np.mean(acc_list)}) , tmp_global_step) 
            
        if (i + 1) % saver_steps == 0:
            #Write the parameters
            tmp_global_step = model.global_step.eval()
            epochs = (tmp_global_step*params["batch_size"])//train_data_size
            model.save(sess, saver, save_filename,epochs)
    