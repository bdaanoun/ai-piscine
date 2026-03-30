import numpy as n
mn_1_50 = n.arange(1,51)
print(mn_1_50)

mn_51_100 = n.arange(51, 101)
print(mn_51_100)

array = n.concatenate([mn_1_50, mn_51_100])
print(array)

reshape_arr = array.reshape(10, 10)
print(reshape_arr)