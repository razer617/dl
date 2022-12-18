
import numpy as np
import tensorflow as tf
print("Matrix Multiplication uising tensorflow")

x=tf.constant([1,2,3,4,5,6],shape=[2,3])
y=tf.constant([7,8,9,10,11,12],shape=[3,2])
print("x=",x)
print("y=",y)
z=tf.matmul(x,y)
print("Matrix Multiplication= ",z)

ematrix=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,name="Matrix A")
print("matrix A={}\n\n".format(ematrix))
evalue,evector=tf.linalg.eigh(ematrix)
print("Eigen Vectors:{}\n\n Eigen Value: {} \n".format(evalue,evector))
