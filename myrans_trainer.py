import os
import numpy as np
import pandas as pd
import time as time 
import argparse
import pickle
from sklearn.ensemble import ExtraTreesClassifier
import multiprocessing as mp
import time


def findListIndecies(files, imgPaths, id):
    _start = time.time()

    valid_files_index = [i for i, item in enumerate(files) if item in imgPaths]


    _end = time.time()
    print("ID {} took: {} seconds".format(id, _end-_start))
    return [id, valid_files_index]



Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
def loadData(path, split, stage):

    gist = np.load(os.path.join(path, "{}_Data.npy".format(split)))
    files = np.load(os.path.join(path, "{}_Filename.npy".format(split)))

    if stage == "binary":
        gt = pd.read_csv(os.path.join(".", "annotations", "SewerML_{}.csv".format(split)), sep=",", encoding="utf-8", usecols = ["Filename", "Defect"])
        imgPaths = gt["Filename"].values
        labels =  gt["Defect"].values.reshape(imgPaths.shape[0],)

        pos_count = len(labels[labels == 1])
        neg_count = labels.shape[0] - pos_count
        class_weights = np.asarray([neg_count/pos_count])

    elif stage == "defect" or stage == "e2e":
        gt = pd.read_csv(os.path.join(".", "annotations", "SewerML_{}.csv".format(split)), sep=",", encoding="utf-8", usecols = Labels + ["Filename", "Defect"])

        if stage == "defect":
            gt = gt[gt["Defect"] == 1]

            
        imgPaths = gt["Filename"].values
        labels = gt[Labels].values

        num_classes = len(Labels)
        data_len = labels.shape[0]
        class_weights = []

        for defect in range(num_classes):
            pos_count = len(labels[labels[:,defect] == 1])
            neg_count = data_len - pos_count

            class_weight = neg_count/pos_count if pos_count > 0 else 0
            class_weights.append(np.asarray([class_weight]))

    if stage == "defect":
        if not os.path.isfile(os.path.join(path, "{}_Data_Subset.npy".format(split))):
            # valid_files_index = [i for i, item in enumerate(files) if item in set(imgPaths)]

            MAX_NR_CORES = min(8, mp.cpu_count())
            NR_CORES = np.minimum(MAX_NR_CORES, len(imgPaths))
            filenamesLen = len(imgPaths)
            splits = [0]
            splits.extend([int(idx/NR_CORES*filenamesLen) for idx in range(1,NR_CORES+1)])
            arguments = []

            for core, idx in enumerate(range(1, NR_CORES+1)):
                imgPathSSplit = imgPaths[splits[idx-1]:splits[idx]]
                arguments.append({"files": files, "imgPaths": set(imgPathSSplit), "id":core+1})

            try:
                p = mp.Pool(NR_CORES)
                print("Evaluating on {} cpu cores".format(NR_CORES))
                processes = [p.apply_async(findListIndecies, kwds=inp) for inp in arguments]
                results = [p.get() for p in processes]
                p.close()
                p.join()

            except:
                p.close()
                p.terminate()
                raise Exception("Evaluation failed")
            
            results = sorted(results, key=lambda x: x[0])
            valid_files_index = []
            list(map(valid_files_index.extend, [x[1] for x in results]))

            gist = gist[valid_files_index]
            files = files[valid_files_index]
            np.save(os.path.join(path, "{}_Data_Subset.npy".format(split)), gist)
            np.save(os.path.join(path, "{}_Filename_Subset.npy".format(split)), files)
        else:
            gist = np.load(os.path.join(path, "{}_Data_Subset.npy".format(split)))
            files = np.load(os.path.join(path, "{}_Filename_Subset.npy".format(split)))

    return gist, labels, files, class_weights


def safe_log(x, eps=1e-10): 
    result = np.where(x > eps, x, -10) 
    np.log(result, out=result, where=result > 0)
    return result

def lossFunc(output, target, weights):
    output = np.asarray(output)

    target = target.reshape(target.shape[0], -1)

    if output.ndim == 2:
        output = output.reshape(-1, output.shape[0], output.shape[1])

    loss = 0
    for cls_idx in range(target.shape[1]):
        logOutput = safe_log(output[cls_idx, :, 0])
        logOutputInverse = safe_log(output[cls_idx, :, 1])

        binaryMask = target[:,cls_idx].astype(np.bool)
        invBinaryMask = ~binaryMask
        posLoss = np.sum(weights[cls_idx] * logOutput[binaryMask])
        negLoss = np.sum(logOutputInverse[invBinaryMask])

        loss += -(posLoss+negLoss)

    loss = loss / target.shape[1]
    loss = loss / target.shape[0]

    return loss


    


def main(args):
    stage = args["stage"]
    output_dir = args["output_dir"]
    gist_dir = args["gist_features"]

    dataDir = gist_dir
    outputDir = os.path.join(output_dir, stage)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    gistTrain, labelsTrain, filesTrain, class_weights = loadData(dataDir, "Train", stage)
    gistVal, labelsVal, filesVal, _ = loadData(dataDir, "Val", stage)
    

    if stage == "binary":
        class_weights_dict = {1:class_weights[0]}
    else:
        class_weights_dict = [{0:1, 1:cw} for cw in class_weights]

    model = ExtraTreesClassifier
    results = {}
    best_val_loss = np.inf
    best_mode_loss = None

    start_time = time.time()
    for n_estimators in [10, 100, 250]:
        for max_depth in [10, 20, 30]:
            for max_features in ["sqrt", "log2", 512//3]:

                forest = model(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1, random_state=0, class_weight=class_weights_dict, min_samples_leaf = 1)
                

                forest.fit(gistTrain, labelsTrain)

                trainProb = forest.predict_proba(gistTrain)
                valProb = forest.predict_proba(gistVal)

                trainLoss = lossFunc(trainProb, labelsTrain, class_weights)
                valLoss = lossFunc(valProb, labelsVal, class_weights)
                
                tag = "{}_{}_{}".format(n_estimators, max_depth, max_features)
                
                if valLoss < best_val_loss:
                    best_val_loss = valLoss
                    best_mode_loss = "{}_{}_{}".format(n_estimators, max_depth, max_features)
                    
                    with open(os.path.join(outputDir, "best_model_loss.pkl"), 'wb') as fid:
                        pickle.dump(forest, fid) 

                with open(os.path.join(outputDir, "{}.pkl".format(tag)), 'wb') as fid:
                        pickle.dump(forest, fid) 

                results[tag] = {"ID": tag, "Trees": n_estimators, "Depth": max_depth, "Features": max_features, "TrainLoss": trainLoss, "ValLoss": valLoss}
                print(tag, trainLoss, valLoss)
                

    end_time = time.time()

    print("\nTime spent: {}".format(end_time-start_time))

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.sort_values(["Trees", "Depth", "Features"])
    print(df)

    df.to_csv(os.path.join(outputDir, "searchResults.csv"), sep=",", index=False)

    with open(os.path.join(outputDir, "bestModels.txt"), "w") as f:
        f.write("Best Loss Model: {}: TrnLoss {} - ValLoss {}\n".format(best_mode_loss, results[best_mode_loss]["TrainLoss"], results[best_mode_loss]["ValLoss"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--stage", default = "binary", choices=["binary", "defect", "e2e"])
    ap.add_argument("--output_dir", type=str, default="./myransModels")
    ap.add_argument("--gist_dir", type=str, default = "./GistFeatures")

    args = vars(ap.parse_args())
    main(args)