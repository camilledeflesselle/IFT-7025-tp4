import numpy as np

def confusion_matrix_binary(true_labels, predicted_labels, positive = 0, print_va = False):
    true = [label == positive for label in true_labels]
    pred = [label == positive for label in predicted_labels]
    vp, fp, fn, vn = 0, 0, 0, 0

    for i in range(len(pred)):
        if true[i] :
            if pred[i] : vp +=1
            else :  fn +=1
        elif pred[i] : fp +=1
        else : vn +=1
    conf = np.array([[vp, fn],
                     [fp, vn]])
    if print_va :
        print("  * Matrice de confusion :", conf)
        print('   Vrais positifs : {} | Faux négatifs : {} \n   Faux positifs : {} | Vrais négatifs : {} '.format(vp, fn, fp, vn))

    return conf

def precision(fp, vp):
    return vp/(vp+fp)

def rappel(fn, vp):
     return vp/(vp+fn)

def F1_score(p, r):
    return 2 * (p*r) / (p+r)

def exactitude(vn, fp, fn, vp):
    return (vn + vp)/(vn + fp + fn + vp)

def show_metrics(true_labels, predicted_labels):
    for i in np.unique(true_labels):
        print("\nClasse positive :", i)
        conf = confusion_matrix_binary(true_labels, predicted_labels, i, True)
        vp, fn = conf[0]
        fp, vn = conf[1]
        print("  * Exactitude =", exactitude(vn, fp, fn, vp))
        p = precision(fp, vp)
        r = rappel(fn, vp)
        print("  * Précision =", p)
        print("  * Rappel =", r)
        print("  * F1-score =", F1_score(p, r))