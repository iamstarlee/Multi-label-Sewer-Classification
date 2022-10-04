import os
import json
import argparse
import pandas as pd
from metrics import evaluation


LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
Labels = list(LabelWeightDict.keys())
LabelWeights = list(LabelWeightDict.values())

def calculateResults(args):
    scorePath = args["score_path"]
    targetPath = args["gt_path"]

    outputPath = args["output_path"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    split = args["split"]

    targetSplitpath = os.path.join(targetPath, "{}13.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",")
    targetsDf = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)
    targets = targetsDf[Labels].values

    for subdir, dirs, files in os.walk(scorePath):
        
        for scoreFile in files:
            item = os.path.splitext(scoreFile)
            
            if split.lower() not in item[0]:
                continue
            # if "e2e" not in item[0] and "twostage" not in item[0] and "defect" not in item[0]:
            #     continue
            if "sigmoid" not in item[0]:
                continue
            if "binary" in item[0]:
                continue
            if item[1] != ".csv":  # if os.path.splitext(scoreFile)[-1] != ".csv":
                continue
            

            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scoresDf = scoresDf.sort_values(by=["Filename"]).reset_index(drop=True)
            
            scores = scoresDf[Labels].values
            
            new, main, auxillary = evaluation(scores, targets, LabelWeights)

            outputName = "{}_{}".format(split, scoreFile)
            if split.lower() == "test":
                outputName = outputName[:len(outputName) - len("_test_sigmoid.csv")]
            elif split.lower() == "val":
                outputName = outputName[:len(outputName) - len("_val_sigmoid.csv")]
            elif split.lower() == "train":
                outputName = outputName[:len(outputName) - len("_train_sigmoid.csv")]


            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump({"Labels": Labels, "LabelWeights": LabelWeights, "New": new, "Main": main, "Auxillary": auxillary}, fp)

            newString = "{:.2f}   {:.2f} ".format(new["F2"]*100,  auxillary["F1_class"][-1]*100)

            aveargeString = "{:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}".format(main["mF1"]*100, main["MF1"]*100, main["OF1"]*100, main["OP"]*100, main["OR"]*100, main["CF1"]*100, main["CP"]*100, main["CR"]*100, main["EMAcc"]*100, main["mAP"]*100)
            
            classF1String = "   ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]])
            classF2String = "   ".join(["{:.2f}".format(x*100) for x in new["F2_class"]])
            classPString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]])
            classRString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]])
            classAPString = "   ".join(["{:.2f}".format(x*100) for x in auxillary["AP"]])

            with open(os.path.join(outputPath,'{}_latex.txt'.format(outputName)), "w") as text_file:
                text_file.write("New metrics: " + newString + "\n")
                text_file.write("ML main metrics: " + aveargeString + "\n")
                text_file.write("Class F1: " + classF1String + "\n")
                text_file.write("Class F2: " + classF2String + "\n")
                text_file.write("Class Precision: " + classPString + "\n")
                text_file.write("Class Recall: " + classRString + "\n")
                text_file.write("Class AP: " + classAPString + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "D:\\Research\\3-code\\sewer-ml\\output")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str, default = "D:\\Research\\3-code\\sewer-ml\\results")
    parser.add_argument("--gt_path", type=str, default = "D:\\Research\\3-code\\sewer-ml\\annotations")

    args = vars(parser.parse_args())

    calculateResults(args)