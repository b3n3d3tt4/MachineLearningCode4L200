import numpy as np
from sklearn  import metrics
from scipy.optimize import linear_sum_assignment # --> algoritmo ungherese
from sklearn.preprocessing import label_binarize

def cluster_accuracy(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    # definition of the dimension of the contingency matrix
    # --> the +1 is needed because labels start from 0
    D = max(y_pred.max(), y_true.max()) + 1 
    
    # creation of the contingency matrix
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    # finds the permutation that maximises the number of correct samples  with the best correspondence cluster --> class
    # --> linear_sum_assignment as a default minimise so to maximise we minimise max-w or in modern scipy we use maximise=True
    row_ind, col_ind = linear_sum_assignment(w, maximize = True)
    
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size



def cluster_purity(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
        
    return metrics.accuracy_score(y_pred_voted, y_true)


def roc_auc(y_true, q_pred, n_classes):
    
    if n_classes == 2:  # binary ROC AUC
        auc = max(metrics.roc_auc_score(y_true, q_pred[:, 1]), metrics.roc_auc_score(y_true, q_pred[:, 0]))
        
    else:  # micro-averaged ROC AUC (multiclass)
        fpr, tpr, _ = metrics.roc_curve(label_binarize(y_true, classes=np.unique(y_true)).ravel(), q_pred.ravel())
        auc = metrics.auc(fpr, tpr)
        
    return auc