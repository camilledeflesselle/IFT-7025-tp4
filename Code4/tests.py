fold_indices = np.arange(150)
valid_indices = np.array_split(fold_indices, 10)

i = 1

idx = valid_indices[i_fold]

print(idx)

id_train = valid_indices[i_fold]

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
for i in np.unique(test_labels):
    true = [label == i for label in test_labels]
    pred = [label == i for label in predictions]
    print("Matrice de confusion pour classe {} : {}".format(i, confusion_matrix(true, pred)))
    p, r, f1_scor, _ = precision_recall_fscore_support(true, pred, average='binary', pos_label=True)
    print("  * Précision =", p)
    print("  * Rappel =", r)
    print("  * F1-score =", f1_scor)