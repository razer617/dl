
import numpy as np
x=np.matrix([[1,2,3],[2,3,7],[2,4,5]])
y=np.matrix([[4,5,6],[2,4,8],[4,3,7]])
print ("matrix1: ",x)
print ("matrix2: ",y)
print(x.shape)
print(y.shape)
a=np.add(x,y)
m=np.matmul(x,y)
g=np.transpose(x)
h=np.transpose(y)
print("addition is :",a)
print(a.shape)
print("Multiplication is: ",m)
print(m.shape)
print("Transpose of matrix1:",g)
print(g.shape)
print("Transpose of matrix2:",h)
print(h.shape)
