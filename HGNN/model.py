import tensorflow as tf
import numpy as np
import json
from network import GNN
from datetime import *
import os
from dataloader_100 import load_num_category, load_train_data, load_train_size, load_fitb_data, load_test_size, \
	load_auc_data

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

import warnings
warnings.filterwarnings("ignore")

ftrain = open('/storage/home/hcocice1/sarora320/wstm/NGNN/data/train_no_dup_new_100.json', 'r', encoding='utf-8')
train_outfit_list = json.load(ftrain)
ftest = open('/storage/home/hcocice1/sarora320/wstm/NGNN/data/test_no_dup_new_100.json', 'r', encoding='utf-8')
test_outfit_list = json.load(ftest)


def cm_hyGCN(batch_size, hidden_size, n_steps, learning_rate, num_category, opt, i, beta):
	hidden_stdv = np.sqrt(1. / (hidden_size))
	if i == 0:
		with tf.compat.v1.variable_scope("cm_ocphn", reuse=None):
			category_w = tf.compat.v1.get_variable(name='gnn/w/conf_image', shape=[120, 2048],
			                             initializer=tf.random_normal_initializer(hidden_stdv))
			w_att0 = tf.compat.v1.get_variable(name='gnn/w/w_att0', shape=[2 * hidden_size, 1],
			                         initializer=tf.random_normal_initializer(hidden_stdv))
			w_att1 = tf.compat.v1.get_variable(name='gnn/w/w_att1', shape=[2 * hidden_size, 1],
			                         initializer=tf.random_normal_initializer(hidden_stdv))
	else:
		with tf.compat.v1.variable_scope("cm_ocphn"):
			tf.compat.v1.get_variable_scope().reuse_variables()

	image_pos = tf.compat.v1.placeholder(tf.float32, [batch_size, num_category, 2048])
	image_neg = tf.compat.v1.placeholder(tf.float32, [batch_size, num_category, 2048])

	with tf.compat.v1.variable_scope('gcn', reuse=None):
		image_state_pos, Se_pos, Ie_pos = GNN('image', image_pos, batch_size, hidden_size, n_steps, num_category,
		                                      category_w)
		tf.compat.v1.get_variable_scope().reuse_variables()
		image_state_neg, Se_neg, Ie_neg = GNN('image', image_neg, batch_size, hidden_size, n_steps, num_category,
		                                      category_w)

	for i in range(batch_size):
		image_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_pos[i], w_att0), [num_category, 1]))
		image_score_pos = tf.reshape(tf.matmul(image_state_pos[i], w_att1), [num_category, 1])
		image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
		image_Se_pos = tf.reshape(image_conf_pos[Se_pos[i][0]] * image_score_pos[Se_pos[i][0]], [1])
		image_Ie_pos = tf.reshape(image_conf_pos[Ie_pos[i][0]] * image_score_pos[Ie_pos[i][0]], [1])
		score_pos = image_Se_pos + image_Ie_pos

		image_conf_neg = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_neg[i], w_att0), [num_category, 1]))  # [1,120]
		image_score_neg = tf.reshape(tf.matmul(image_state_neg[i], w_att1), [num_category, 1])  # [120,1]
		image_score_neg = tf.maximum(0.01 * image_score_neg, image_score_neg)  # [120,1]
		image_Se_neg = tf.reshape(image_conf_neg[Se_neg[i][0]] * image_score_neg[Se_neg[i][0]], [1])
		image_Ie_neg = tf.reshape(image_conf_neg[Ie_neg[i][0]] * image_score_neg[Ie_neg[i][0]], [1])
		score_neg = image_Se_neg + image_Ie_neg
		if i == 0:
			s_pos = score_pos
			s_neg = score_neg
		else:
			s_pos = tf.concat([s_pos, score_pos], 0)
			s_neg = tf.concat([s_neg, score_neg], 0)

	s_pos = tf.reshape(s_pos, [batch_size, 1])
	s_neg = tf.reshape(s_neg, [batch_size, 1])

	s_pos_mean = tf.reduce_mean(s_pos)
	s_neg_mean = tf.reduce_mean(s_neg)

	cost_parameter = 0.
	num_parameter = 0.

	for variable in tf.compat.v1.trainable_variables():
# 		print(variable)
		cost_parameter += tf.keras.regularizers.l2(0.1)(variable)
# 		print(cost_parameter)
		num_parameter += 1
# 		print(num_parameter)

	print("Number of parameters: {}".format(num_parameter))    
	cost_parameter /= num_parameter

    

	score = tf.nn.sigmoid(s_pos - s_neg)
	score_mean = tf.reduce_mean(score)
	cost = -score_mean + cost_parameter

	if opt == 'Adam':
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
	if opt == 'Momentum':
		optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
	if opt == 'RMSProp':
		optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(cost)
	# optimizer1 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_acc)
	if opt == 'Adadelta':
		optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate).minimize(cost)
	print("Optimizer used: {}".format(opt))
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'

	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
	with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
		if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
			init = tf.compat.v1.initialize_all_variables()
		else:
			init = tf.compat.v1.global_variables_initializer()
		sess.run(init)

		best_accurancy = 0.
		best_auc = 0.
		best_epoch = 0
		saver = tf.compat.v1.train.Saver()

		train_size, train_size_ = load_train_size()
		print('train_size is %d' % train_size_)
		train_batch = int(train_size_ / batch_size)
		print('train_batch is %d' % train_batch)

# 		for epoch in range(80):
		for epoch in range(15):
			test_interval = 1000
			no_count = 0.
			c_all = 0.
			score_all = 0.
			dis_pos_all = 0.
			dis_neg_all = 0.
			for i in range(train_batch):
				train_image_pos, train_image_neg, size_ = load_train_data(i, batch_size, train_outfit_list)

				if size_ >= batch_size:
					image_pos_ = train_image_pos[0:batch_size]
					image_neg_ = train_image_neg[0:batch_size]
					_, c, score, dis_pos_, dis_neg_ = sess.run(
						[optimizer, cost, score_mean, s_pos_mean, s_neg_mean],
						feed_dict={image_pos: image_pos_,
						           image_neg: image_neg_})
					c_all += c
					score_all += score
					dis_pos_all += dis_pos_
					dis_neg_all += dis_neg_
					if i % test_interval == 0:
						print('now batch: %d, total batch: %d' % (i, train_batch))

						c_average = c_all / (i + 1)
						score_average = score_all / (i + 1)
						dis_pos_average = dis_pos_all / (i + 1)
						dis_neg_average = dis_neg_all / (i + 1)

						test_size_fitb = load_test_size()
						batches = int((test_size_fitb * 4) / batch_size)
						right = 0.
						for ii in range(batches):
							image = load_fitb_data(ii, batch_size, test_outfit_list)
							answer = sess.run([s_pos], feed_dict={image_pos: image})

							answer = np.asarray(answer[0])
							for j in range(int(batch_size / 4)):
								a = []
								for k in range(j * 4, (j + 1) * 4):
									a.append(answer[k][0])
								if np.argmax(a) == 0:  # np.argmax(a)返回a中最大值的索引
									right += 1.

						accurancy = float(right / test_size_fitb)
						if accurancy > best_accurancy:
							best_accurancy = accurancy
							best_epoch = epoch

						####### AUC #######
						test_size_auc = load_test_size()
						batches = int((test_size_auc * 2) / batch_size)
						right = 0.

						pos_score = []
						neg_score = []
						for ii in range(batches):
							image = load_auc_data(ii, batch_size, test_outfit_list)
							answer = sess.run([s_pos], feed_dict={image_pos: image})
							answer = np.asarray(answer[0])
							for jj in range(batch_size):
								if jj % 2 == 0:
									pos_score.append(answer[jj][0])
								else:
									neg_score.append(answer[jj][0])
						test_size = len(pos_score)
						for i in range(test_size):
							for j in range(test_size):
								if pos_score[i] > neg_score[j]:
									right += 1
								elif pos_score[i] == neg_score[j]:
									right += 0.5
								else:
									right += 0
						auc = float(right / (test_size * test_size))

						if auc > best_auc:
							best_auc = auc
							best_epoch = epoch
							saver.save(sess, "trained_model/hygvn.ckpt")

						print('now():' + str(datetime.now()))
						print("Train Epoch:", '%d' % epoch,
						      "accurancy:", "{:.9f}".format(accurancy), "auc:", "{:.9f}".format(auc))
						print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
						      "Best auc: %f" % best_auc,
						      "Best epoch: %d" % best_epoch)
						print("batch_size: %d, hidden_size: %d,  n_steps: %d, learning_rate: %f" %
						      (batch_size, hidden_size, n_steps, learning_rate))

				else:
					no_count += 1

			c_average = c_all / train_batch
			score_average = score_all / train_batch
			dis_pos_average = dis_pos_all / train_batch
			dis_neg_average = dis_neg_all / train_batch

			print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
			      "Best auc: %f" % best_auc,
			      "Best epoch: %d" % best_epoch)
			print("batch_size: %d, hidden_size: %d, n_steps: %d, learning_rate: %f" % (
				batch_size, hidden_size, n_steps, learning_rate))

			batches = int((test_size_fitb * 4) / batch_size)
			right = 0.

			for i in range(batches):
				image = load_fitb_data(i, batch_size, test_outfit_list)
				answer = sess.run([s_pos], feed_dict={image_pos: image})
				answer = np.asarray(answer[0])

				for j in range(int(batch_size / 4)):
					a = []
					for k in range(j * 4, (j + 1) * 4):
						a.append(answer[k][0])
					if np.argmax(a) == 0:
						right += 1.

			accurancy = float(right / test_size_fitb)

			batches = int((test_size_auc * 2) / batch_size)
			right = 0.
			pos_score = []
			neg_score = []
			for i in range(batches):
				image = load_auc_data(i, batch_size, test_outfit_list)
				answer = sess.run([s_pos], feed_dict={image_pos: image})  # answer 就是8个正样本和8个负样本的值  是[16,1]
				answer = np.asarray(answer[0])  # 转换为array
				for jj in range(batch_size):
					if jj % 2 == 0:
						pos_score.append(answer[jj][0])
					else:
						neg_score.append(answer[jj][0])
			test_size = len(pos_score)
			for i in range(test_size):
				for j in range(test_size):
					if pos_score[i] > neg_score[j]:
						right += 1
					elif pos_score[i] == neg_score[j]:
						right += 0.5
					else:
						right += 0
			auc = float(right / (test_size * test_size))

			if auc > best_auc:
				best_auc = auc
				best_epoch = epoch

			if accurancy > best_accurancy:
				best_accurancy = accurancy
				best_epoch = epoch
				# saver.save(sess, "modal_final/hygcn.ckpt")

			print("Test Epoch:", '%d' % epoch, "accuracy:", "{:.9f}".format(accurancy), "auc:",
			      "{:.9f}".format(auc))

			print('now():' + str(datetime.now()))
			print("batch_size: %d, hidden_size: %d, n_steps: %d, learning_rate: %f" % (
				batch_size, hidden_size, n_steps, learning_rate))
			print("Epoch:", '%d' % epoch, "finished", "Best accurancy: %f" % best_accurancy,
			      "Best auc: %f" % best_auc,
			      "Best epoch: %d" % best_epoch)

	return best_accurancy


if __name__ == '__main__':
	num_category = load_num_category()
	best_accurancy = 0.
	i = 0
	batch_size = 16
	hidden_size = 16
	n_steps = 1
	learning_rate = 0.0001
	opt = "Adam"
	beta = 0.2
	accurancy = cm_hyGCN(batch_size, hidden_size, n_steps, learning_rate, num_category, opt, i, beta)
	print("best parameter is batch_size, hidden_size,  n_steps, learning_rate, optimizer:%d ,%d , %d, %f, %s" % (
		batch_size, hidden_size, n_steps, learning_rate, opt))
