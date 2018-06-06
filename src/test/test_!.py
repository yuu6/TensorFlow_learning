# -*- coding:utf-8 -*-
"""
@Time:2018/6/1 22:45
@Author:yuhongchao
"""
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

# 构造一个线性模型
#

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
y = W * x_data +b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图
sess = tf.Session()
sess.run(init)

#拟合平面
for step in range(201):
	sess.run(train)
	if step % 20 ==0:
		print(step,sess.run(W),sess.run(b))