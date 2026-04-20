import numpy as np

generator = np.random.default_rng(123)
grades = np.round(generator.uniform(low = 0.0, high = 10.0, size = (10, 2)))
grades[[1,2,5,7], [0,0,0,0]] = np.nan
print(grades)

third_grade = np.where(np.isnan(grades[:,0]) , grades[0:, 1], grades[:, 0])
# res = np.column_stack([grades, third_grade])
res = np.concatenate([ grades,third_grade.reshape(10, 1)], axis=1)
print(res)
