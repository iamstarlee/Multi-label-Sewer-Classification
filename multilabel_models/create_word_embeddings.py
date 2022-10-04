import os
import numpy as np
import argparse



def one_hot_embedding(num_classes):
    return np.identity(num_classes)


def glove_embedding(num_classes):
    raise ValueError("GloVe embeddings not yet implemented")
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputPath', type=str, default = "../word_embeddings")
    parser.add_argument('--numClasses', type=int, default=17)
    args = parser.parse_args()

    args = vars(args)

    outputPath = args["outputPath"]
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)


    num_classes = args["numClasses"]

    one_hot = one_hot_embedding(num_classes)
    np.save(os.path.join(outputPath, "one_hot.npy"), one_hot)
    glove = glove_embedding(num_classes)