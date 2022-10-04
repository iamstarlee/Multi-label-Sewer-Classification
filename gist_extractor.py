import os
import cv2
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import time

import gist

def extractGISTWorker(files, dirPath, id):
    _start = time.time()

    dataArr = None
    filenames = []
    # print("the files are {} , the dirPath is {}, and the id is {}".format(files, dirPath, id))
    files = sorted(files)
    for idx, imgFile in enumerate(files):

        if idx % 2000 == 0:
            print(id, idx, len(files))

        imgPath = os.path.join(dirPath, imgFile)
        print('the imgPath is {}'.format(imgPath))
        img = cv2.imread(imgPath)
        imgGray = np.mean(img, axis=2)
        img[:,:,0] = imgGray
        img[:,:,1] = imgGray
        img[:,:,2] = imgGray
        
        img = cv2.resize(img, (128, 128)).astype(np.uint8)

        descriptor = gist.extract(img, nblocks=4, orientations_per_scale=(8, 8, 8, 8))

        filenames.append(imgFile)
        if dataArr is None:
            dataArr = descriptor[:512]
        else:
            dataArr = np.vstack((dataArr, descriptor[:512]))

    _end = time.time()
    print("ID {} took: {} seconds".format(id, _end-_start))
    return dataArr, filenames


def main(args):
    dataDir = args["data_root"]
    annDir = args["ann_root"]
    outputDir = args["output_dir"]
    cores = args["cores"]

    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    MULTIPROCESSING = True
    MAX_NR_CORES = min(cores, mp.cpu_count())
    if MAX_NR_CORES == 1:
        MULTIPROCESSING = False
    results = []


    for split in ["Train", "Val", "Test"]:
        gtPath = os.path.join(annDir, "{}13.csv".format(split))
        gtDF = pd.read_csv(gtPath, encoding="utf-8")

        filenames = gtDF["Filename"].values.tolist()

        # set number of core for mutliprocessing
        if MULTIPROCESSING: 
            NR_CORES = np.minimum(MAX_NR_CORES, len(filenames))
            filenamesLen = len(filenames)
            splits = [0]
            splits.extend([int(idx/NR_CORES*filenamesLen) for idx in range(1,NR_CORES+1)])
            arguments = []

            for core, idx in enumerate(range(1, NR_CORES+1)):
                filesSplit = filenames[splits[idx-1]:splits[idx]]
                arguments.append({"files": filesSplit, "dirPath": dataDir, "id":core+1})

            try:
                p = mp.Pool(NR_CORES)
                print("Evaluating on {} cpu cores".format(NR_CORES))
                processes = [p.apply_async(extractGISTWorker, kwds=inp) for inp in arguments]
                results = [p.get() for p in processes]
                p.close()
                p.join()

            except:
                p.close()
                p.terminate()
                raise Exception("Evaluation failed")
        else:
            arguments = [{"files": filesSplit, "dirPath": dataDir, "id":1}]
            results = [extractGISTWorker(**inp) for inp in arguments]
    
        filenames = []
        data = [x[0] for x in results]
        list(map(filenames.extend, [x[1] for x in results]))



        dataArr = np.vstack(data)

        np.save(os.path.join(outputDir, split+"_Data.npy"), dataArr)
        np.save(os.path.join(outputDir, split+"_Filename.npy"), filenames)

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_root", default ="D:\\Research\\3-code\\sewer-ml\\Data\\train13")
    ap.add_argument("-a", "--ann_root", default ="D:\\Research\\3-code\\sewer-ml\\annotations")
    ap.add_argument("-o", "--output_dir", help="Path to where the output ground truth CSV file should be saved", default ="D:\\Research\\3-code\\sewer-ml\\GistFeatures")
    ap.add_argument("-co", "--cores", help="Amount of CPU cores to use", default = 4, type = int)

    args = vars(ap.parse_args())
    main(args)