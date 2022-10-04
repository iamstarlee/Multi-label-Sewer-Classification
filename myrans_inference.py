import os
import numpy as np
import pandas as pd
import time as time 
import argparse
import pickle

from myrans_trainer import loadData


def main(args):
    modelPath = args["model_path"]
    dataDir = args["gist_dir"]
    output_dir_sub = args["output_dir"]
    split = args["split"]

    if "binary" in modelPath:
        stage = "binary"
        dataStage = "binary"
        Labels = ["Defect"]
    else:
        Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]

        if "e2e" in modelPath:
            stage = "e2e"
        else:
            stage = "defect"
        
        dataStage = "e2e"


    outputDir = os.path.join(output_dir_sub, stage)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    
    gistFeats, _, fileNames, _ = loadData(dataDir, split, dataStage)
    
    with open(os.path.join(modelPath), 'rb') as fid:
        forest = pickle.load(fid) 

    start_time = time.time()

    prob = forest.predict_proba(gistFeats)

    if isinstance(prob, list):
        prob = np.array(prob)
    if prob.ndim == 2:
        prob = prob.reshape(-1, prob.shape[0], prob.shape[1])


    expname = os.path.basename(os.path.normpath(modelPath))
    
    sigmoid_dict = {}
    sigmoid_dict["Filename"] = fileNames
    for idx, header in enumerate(Labels):
        sigmoid_dict[header] = prob[idx, :, 0]

    df_out = pd.DataFrame(sigmoid_dict)
    df_out.to_csv(os.path.join(outputDir, "Myrans", "Myrans_{}_{}_sigmoid.csv".format(expname, split.lower())), sep=",", index=False)

    end_time = time.time()

    print("\nTime spent: {}".format(end_time-start_time))

    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, default="./pretrained_models")
    ap.add_argument("--gist_dir", type=str, default="./GISTFeatures")
    ap.add_argument("--output_dir", type=str, default="./results")
    ap.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])

    args = vars(ap.parse_args())
    main(args)