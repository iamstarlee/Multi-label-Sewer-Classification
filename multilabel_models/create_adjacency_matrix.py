import os
import pandas as pd
import numpy as np
import argparse


Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]


def normalize_adjacency_matrix(adj):
    D = adj.sum(1)
    D_power = D ** (-0.5)
    D_power = np.diag(D_power)

    adj_norm = np.matmul(np.matmul(D_power, adj), D_power)
    return adj_norm

def MLGCN_adjacency_preprocessing(adjacency_matrix, class_count, tau = 0.4, neighbour_weight = 0.2):
    adj = adjacency_matrix
    classes = class_count
    num_classes = len(classes)

    classes = class_count[:, np.newaxis] # Reshape to (num_classes, 1)
    adj = adj / classes # Row normalize - Equation 6

    # Binarize adjacency matrix - Equation 7
    adj[adj < tau] = 0
    adj[adj >= tau] = 1

    adj_binary = adj

    # Re-weight adjacency matrix - Equation 8
    adj = adj * neighbour_weight / (adj.sum(0, keepdims=True) + 1e-6)
    adj = adj + np.identity(num_classes, np.int)*(1-neighbour_weight)

    return adj, adj_binary


def KSSNET_adjacency_preprocessing(stat_adj, knowledge_adj = None, lambd = 0.4, tau = 0.02, eta = 0.4):

    # Equation 2
    if knowledge_adj is None or lambd == 1.0:
        stat_adj_norm = normalize_adjacency_matrix(stat_adj)
        adj = stat_adj_norm
    elif stat_adj is None or lambd == 0.0:
        knowledge_adj_norm = normalize_adjacency_matrix(knowledge_adj)
        adj = knowledge_adj_norm
    else:
        stat_adj_norm = normalize_adjacency_matrix(stat_adj)
        knowledge_adj_norm  = normalize_adjacency_matrix(knowledge_adj)
        adj = lambd * stat_adj_norm + (1-lambd)*knowledge_adj_norm

    adj[adj < tau] = 0  # Equation 3

    adj_KS = eta*adj + np.identity(adj.shape[0])*(1-eta) #Equation 4

    return adj_KS


def create_adjacency_matrix(annRoot, split):


    LabelNames = Labels.copy()
    LabelNames.remove("VA")
    LabelNames.remove("ND")

    num_classes = len(LabelNames)

    gtPath = os.path.join(annRoot, "SewerML_{}.csv".format(split))
    gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = LabelNames)

    adjacency_matrix = np.zeros((num_classes, num_classes))
    class_sum = np.zeros((num_classes), dtype=np.int)

    for idx_1 in range(num_classes):
        defect_1 = Labels[idx_1]

        subdf1 = gt[gt[defect_1] == 1]
        class_sum[idx_1] = len(subdf1)
        for idx_2 in range(idx_1+1, num_classes):
            defect_2 = Labels[idx_2]

            subdf2 = subdf1[subdf1[defect_2] == 1]

            adjacency_matrix[idx_1, idx_2] = adjacency_matrix[idx_2, idx_1] = len(subdf2)

    return adjacency_matrix, class_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, default="../annotations")
    parser.add_argument('--outputPath', type=str, default="../adjacency_matrices")
    parser.add_argument('--split', type=str, default = "Train")
    parser.add_argument('--mlgcn_tau', type=float, default = 0.4)
    parser.add_argument('--mlgcn_neighbour_weight', type=float, default = 0.2)
    parser.add_argument('--kssnet_lambda', type=float, default = 0.4)
    parser.add_argument('--kssnet_tau', type=float, default = 0.02)
    parser.add_argument('--kssnet_eta', type=float, default = 0.4)
    args = parser.parse_args()

    args = vars(args)

    outputPath = args["outputPath"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)


    adj, class_sum = create_adjacency_matrix(args["inputPath"], args["split"])
    adj_reweighted_mlgcn, adj_mlgcnn_binary = MLGCN_adjacency_preprocessing(adj, class_sum, args["mlgcn_tau"], args["mlgcn_neighbour_weight"])
    adj_ks = KSSNET_adjacency_preprocessing(adj_reweighted_mlgcn, None, args["kssnet_lambda"], args["kssnet_tau"], args["kssnet_eta"])

    np.save(os.path.join(outputPath, "adj.npy"), adj)
    np.save(os.path.join(outputPath, "adj_binary_mlgcn.npy"), adj_mlgcnn_binary)
    np.save(os.path.join(outputPath, "adj_reweighted_mlgcn.npy"), adj_reweighted_mlgcn)
    np.save(os.path.join(outputPath, "adj_ks.npy"), adj_ks)