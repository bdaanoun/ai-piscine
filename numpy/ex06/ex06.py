import numpy as np

D2_arr = np.ones((9,9), dtype=np.int8)
print(D2_arr)

D2_arr[1:8, 1:8]=0
D2_arr[2:7, 2:7] = 1
D2_arr[3:6, 3:6] = 0
D2_arr[4, 4] = 1
print(D2_arr)

array_1 = np.array([1,2,3,4,5], dtype=np.int8)
array_2 = np.array([1,2,3], dtype=np.int8)
# print(array_2*array_1)

reshaped_arr = array_1.reshape(5,1)
res= reshaped_arr * array_2

print(res)