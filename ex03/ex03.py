import numpy as np

hunderd_arr = np.arange(1,101)
# print(hunderd_arr)

odd_ints = hunderd_arr[::2]
print(odd_ints)

reversed_even_nums = hunderd_arr[-1::-2]
print(reversed_even_nums)


hunderd_arr[1::3] = 0
print(hunderd_arr)