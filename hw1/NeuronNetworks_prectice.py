import tensorflow as tf
import numpy as np

# hyper paremeters
num_neurons_lay1 = 800
num_neurons_lay2 = 400
num_neurons_lay3 = 200

def add_layer(inputs, in_size, out_size, activation_fn = None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_fn is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_fn(Wx_plus_b)
	return outputs

def build_train_NN(n_x, n_y, x_data, y_data):
	xs = tf.placeholder(tf.float32, [None, n_x])
	ys = tf.placeholder(tf.float32, [None, n_y])
	lay1 = add_layer(xs, n_x, num_neurons_lay1, activation_fn = tf.nn.relu)
	lay2 = add_layer(lay1, num_neurons_lay1, num_neurons_lay2, activation_fn = tf.nn.relu)
	lay3 = add_layer(lay2, num_neurons_lay2, num_neurons_lay3, activation_fn = tf.nn.relu)
	prediction = add_layer(lay3, num_neurons_lay3, n_y,activation_fn = None)
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)))
	train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for _ in range(1000):
		sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
		print sess.run(loss, feed_dict = {xs:x_data, ys:y_data})
		prediction_value = sess.run(prediction, feed_dict = {xs:x_data, ys:y_data})