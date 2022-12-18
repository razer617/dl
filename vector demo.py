
import numpy as np
#declaring vector
x=[1,2,3,]
y=[4,5,6]
print(x)
print(y)
print(type(x))
#this dosent give the vector addition
print(x+y)
#vector addition using numpy
z=np.add(x,y)
print(type(z))
#vector Cross Product
mul=np.cross(x,y)
print(mul)
