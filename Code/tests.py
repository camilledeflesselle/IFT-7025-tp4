fold_indices = np.arange(150)
valid_indices = np.array_split(fold_indices, 10)

i = 1

idx = valid_indices[i_fold]

print(idx)

id_train = valid_indices[i_fold]