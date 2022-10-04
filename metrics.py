import numpy as np

def average_precision(scores, target, max_k = None):
    
    assert scores.shape == target.shape, "The input and targets do not have the same shape"
    assert scores.ndim == 1, "The input has dimension {}, but expected it to be 1D".format(scores.shape)

    # sort examples
    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k == None:
        max_k = len(indices)
    
    # Computes prec@i
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break
        
    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i


def micro_f1(Ng, Np, Nc):
    mF1 = (2 * np.sum(Nc)) / (np.sum(Np) + np.sum(Ng))

    return mF1

def macro_f1(Ng, Np, Nc):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F1_k = (2 * precision_k * recall_k)/(precision_k + recall_k)

    F1_k[np.isnan(F1_k)] = 0

    MF1 = np.sum(F1_k)/n_class

    return precision_k, recall_k, F1_k, MF1

def overall_metrics(Ng, Np, Nc):
    OP = np.sum(Nc) / np.sum(Np)
    OR = np.sum(Nc) / np.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)

    return OP, OR, OF1

def per_class_metrics(Ng, Np, Nc):
    n_class = len(Ng)
    CP = np.sum(Nc / Np) / n_class
    CR = np.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)

    return CP, CR, CF1

def mean_average_precision(ap):
    return np.mean(ap)

def exact_match_accuracy(scores, targets, threshold = 0.5):
    n_examples, n_class = scores.shape

    binary_mat = np.equal(targets, (scores >= threshold))
    row_sums = binary_mat.sum(axis=1)

    perfect_match = np.zeros(row_sums.shape)
    perfect_match[row_sums == n_class] = 1

    EMAcc = np.sum(perfect_match) / n_examples

    return EMAcc

def class_weighted_f2(Ng, Np, Nc, weights, threshold=0.5):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F2_k = (5 * precision_k * recall_k)/(4*precision_k + recall_k)

    F2_k[np.isnan(F2_k)] = 0

    ciwF2 = F2_k * weights 
    ciwF2 = np.sum(ciwF2) / np.sum(weights)

    return ciwF2, F2_k


def evaluation(scores, targets, weights, threshold = 0.5):

    assert scores.shape == targets.shape, "The input and targets do not have the same size: Input: {} - Targets: {}".format(scores.shape, targets.shape)

    _, n_class = scores.shape

    # Arrays to hold binary classification information, size n_class +1 to also hold the implicit normal class
    Nc = np.zeros(n_class+1) # Nc = Number of Correct Predictions  - True positives
    Np = np.zeros(n_class+1) # Np = Total number of Predictions    - True positives + False Positives
    Ng = np.zeros(n_class+1) # Ng = Total number of Ground Truth occurences

    # False Positives = Np - Nc
    # False Negatives = Ng - Nc
    # True Positives = Nc
    # True Negatives = n_examples - Np + (Ng - Nc)

    # Array to hold the average precision metric. only size n_class, since it is not possible to calculate for the implicit normal class
    ap = np.zeros(n_class)
    

    for k in range(n_class):
        tmp_scores = scores[:, k]
        tmp_targets = targets[:, k]
        tmp_targets[tmp_targets == -1] = 0 # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss

        Ng[k] = np.sum(tmp_targets == 1)
        Np[k] = np.sum(tmp_scores >= threshold) # when >= 0 for the raw input, the sigmoid value will be >= 0.5
        Nc[k] = np.sum(tmp_targets * (tmp_scores >= threshold))

        ap[k] = average_precision(tmp_scores, tmp_targets)
        #print("the Ng is {}".format(Ng))

    # Get values for "implict" normal class
    tmp_scores = np.sum(scores >= threshold, axis=1)
    tmp_scores[tmp_scores > 0] = 1
    tmp_scores = np.abs(tmp_scores - 1)
    
    tmp_targets = targets.copy()
    tmp_targets[targets == -1] = 0 # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
    tmp_targets = np.sum(tmp_targets, axis=1)
    tmp_targets[tmp_targets > 0] = 1
    tmp_targets = np.abs(tmp_targets - 1)

    Ng[-1] = np.sum(tmp_targets == 1)
    Np[-1] = np.sum(tmp_scores >= threshold)
    Nc[-1] = np.sum(tmp_targets * (tmp_scores >= threshold))



    # If Np is 0 for any class, set to 1 to avoid division with 0
    Np[Np == 0] = 1

    # Overall Precision, Recall and F1
    OP, OR, OF1 = overall_metrics(Ng, Np, Nc)

    # Per-Class Precision, Recall and F1
    CP, CR, CF1 = per_class_metrics(Ng, Np, Nc)

    # Macro F1
    precision_k, recall_k, F1_k, MF1 = macro_f1(Ng, Np, Nc)

    # Micro F1
    mF1 = micro_f1(Ng, Np, Nc)

    # Zero-One exact match accuracy
    EMAcc = exact_match_accuracy(scores, targets)

    # Mean Average Precision (mAP)
    mAP = mean_average_precision(ap)



    F2, F2_k, = class_weighted_f2(Ng[:-1], Np[:-1], Nc[:-1], weights)

    F2_normal = (5 * precision_k[-1] * recall_k[-1])/(4*precision_k[-1] + recall_k[-1])

    new_metrics = {"F2": F2,
                   "F2_class": list(F2_k) + [F2_normal],
                   "F1_Normal": F1_k[-1]}

    main_metrics = {"OP": OP,
                    "OR": OR,
                    "OF1": OF1,
                    "CP": CP,
                    "CR": CR,
                    "CF1": CF1,
                    "MF1": MF1,
                    "mF1": mF1,
                    "EMAcc": EMAcc,
                    "mAP": mAP}

    auxillery_metrics = {"P_class": list(precision_k),
                         "R_class": list(recall_k),
                         "F1_class": list(F1_k),
                         "AP": list(ap),
                         "Np": list(Np),
                         "Nc": list(Nc),
                         "Ng": list(Ng)}

    return new_metrics, main_metrics, auxillery_metrics