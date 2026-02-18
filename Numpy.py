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
#how to get the second element from the numpy array
import numpy as np
arr=np.array([1,2,3,4,5])
print(arr[1])
#How to Access the 2d array
import numpy as np
arr=np.array([[4,5,6],[9,8,7],[6,5,4]])
print(arr[1,2])
#slicing of array
import numpy as np
arr=np.array([1,2,3,4,5])
print(arr[1:4])
print(arr[:2])
print(arr[2:])
#copy and view of array
import numpy as np
arr=np.array([1,2,3,4,5])
arr_copy=arr.copy()
arr_view=arr.view()
#how to print the shape of an array
#shape tells how many rows and coluns are present in the array
import numpy as np
arr=np.array([[1,2,3],[4,5,6]])
print(arr.shape)
#reshape of array
#iif we want to change the shape of an array we can use the reshape mehthod 
#reshape takes new shape as argument and returns the new array with the 
import numpy as np
arr=np.array([1,2,3,4,5,6])
arr_reshaped=arr.reshape(2,3)
print(arr_reshaped)
