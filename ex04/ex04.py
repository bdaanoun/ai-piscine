import numpy as np

rng = np.random.default_rng(seed=888)
# heights = rng.normal(loc=1.51, scale=0.0741, size=2)
one_dem=rng.standard_normal(100)
# print(one_dem)

two_dem = rng.integers(1,10, endpoint=True, size=(8, 8))
# print(two_dem)

three_dem = rng.integers(1,17, endpoint=True, size=(4,2,5))
print(three_dem)
