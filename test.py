import numpy as np
mat = np.array([[0,1,2],
               [0,0,3],
               [7,0,0]])
print(np.where(mat == 0))
print(mat[np.where(mat == 0)])