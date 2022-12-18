
import numpy as np
from numpy import generic
def isscalar(num):
 if isinstance(num,generic):
     return True
 else:
     return False
print(np.isscalar(3.1))
print(np.isscalar("abc"))
print(np.isscalar([1,2,3]))
