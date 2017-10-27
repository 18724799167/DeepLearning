# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:14:50 2017

@author: Administrator
"""

import tensorflow as tf

def test1():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))