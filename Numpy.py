#Numpy is the python library used for working with arrays and matrices.
import numpy as np
#Creating a Id array
id_array=np.array([1,2,3,4,5])
print("Id Array:",id_array)
#Definning number of dimensions using arrays
import numpy as np
np.array([1,2,3,4,5],ndmin=2)
print("Array with 2 dimensions:",np.array([1,2,3,4,5] , ndmin=2))
#Accessing Array with Elements
import numpy as np
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)
import numpy as np
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr[1, 1, 0])
#dimension of arrays means how many dimensions this array as 
import numpy as np
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr[1, 1, 0])