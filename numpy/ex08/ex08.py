import numpy as np
data = np.genfromtxt("data/winequality-red.csv", delimiter=";", skip_header=1, dtype=np.float32)
print(data.nbytes)

np.set_printoptions(suppress=True)

rows = data[[1, 6, 11]]
rows = rows[~np.isnan(rows).any(axis=1)]
print(rows)

alcohol = bool(np.any(rows[:, 10]>20))
print(alcohol)

avg_alcohol = np.nanmean(data[:, 10])
print(avg_alcohol)

ph_static = data[:, 8]

th_25 = np.percentile(ph_static, 25)
th_50 = np.percentile(ph_static, 50)
th_75 = np.percentile(ph_static, 75)
th_min = np.nanmin(ph_static)
th_max = np.nanmax(ph_static)
th_mean = np.nanmean(ph_static)

print(f"25 percentile:  {np.percentile(ph_static, 25):.2f}")
print(f"50 percentile:  {np.percentile(ph_static, 50):.2f}")
print(f"75 percentile:  {np.percentile(ph_static, 75):.2f}")
print(f"mean:  {np.nanmean(ph_static):.2f}")
print(f"min:  {np.nanmin(ph_static):.2f}")
print(f"max:  {np.nanmax(ph_static):.2f}\n")



p20 = np.percentile(data[:, 9], 20)

mask = data[:, 9] <= p20

selec_quality = data[mask, 11]

average_quality = np.mean(selec_quality)
print(np.round(average_quality, 1))


best_qu =  np.max(data[:, 11])
worst_qu =  np.min(data[:, 11])

best_bool = data[:, 11] == best_qu
worst_bool = data[:, 11] == worst_qu

best_mean = np.mean(data[best_bool], axis=0)
worst_mean = np.mean(data[worst_bool], axis=0)
print(best_mean)
print(worst_mean)
