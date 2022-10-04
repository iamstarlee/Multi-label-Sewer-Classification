import os
import pandas as pd
import numpy as np
import argparse

Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]


def combineTwoStage(args):
    strategy = args["strategy"]

    stageOnePath = args["stageOnePath"]
    stageTwoPath = args["stageTwoPath"]

    outputPath = args["outputPath"]
    outputName = args["outputName"]
    split = args["split"]

    stageOne = pd.read_csv(stageOnePath, sep=",")
    stageTwo = pd.read_csv(stageTwoPath, sep=",")

    stageOne = stageOne.sort_values(by=["Filename"]).reset_index(drop=True)
    stageTwo = stageTwo.sort_values(by=["Filename"]).reset_index(drop=True)

    defectScores = stageOne["Defect"].values.reshape(-1, 1)
    classScores = stageTwo[Labels].values
    filenames = stageTwo["Filename"].values


    if strategy == "multiply":
        newClassScores = defectScores * classScores
    elif strategy == "replace":
        defectScoreMatrix = np.repeat(defectScores, classScores.shape[-1], axis=1)
        binaryMat = defectScoreMatrix < 0.5
        newClassScores = classScores.copy()
        np.putmask(newClassScores, binaryMat, defectScoreMatrix)


    finalDict = {}
    finalDict["Filename"] = list(filenames)
    for idx, label in enumerate(Labels):
        finalDict[label] = newClassScores[:, idx]


    finalDF = pd.DataFrame(finalDict)
    finalDF.to_csv(os.path.join(outputPath, "{}_{}_sigmoid.csv".format(outputName, split)), sep=",", index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "./results")
    parser.add_argument("--output_mame", type=str)
    parser.add_argument("--strategy", type=str, default = "replace", choices=["replace", "multiply"])
    parser.add_argument("--split", type=str, default = "test", choices=["train", "val", "test"])
    parser.add_argument("--stage_one_path", type=str)
    parser.add_argument("--stage_two_path", type=str)

    args = vars(parser.parse_args())

    combineTwoStage(args)