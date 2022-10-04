import os
from argparse import ArgumentParser
from inference import run_inference



def iterateResultsDirs(args):
    log_input_path = args["log_input"]
    base_output_dir = args["results_output"]

    for subdir, dirs, _ in os.walk(log_input_path):

        if len(dirs) == 0:
            model_subdirs = subdir[len(log_input_path):]
            model_subdirs = model_subdirs.replace("/", "|")
            model_subdirs = model_subdirs.replace("\\", "|")
            model_subdirs = model_subdirs.split("|")
            model_subdirs = [x for x in model_subdirs if len(x)]

            model = model_subdirs[0]
            version = model_subdirs[1]

            model_version = "{}_{}".format(model, version)

            output_dir = os.path.join(base_output_dir, model_version)


            args_dict = {"ann_root": args["ann_root"],
                        "data_root": args["data_root"],
                        "batch_size": args["batch_size"],
                        "workers": args["workers"],
                        "split": args["split"],
                        "model_path": subdir,
                        "results_output": output_dir,
                        "model_version": model_version}

            try:
                run_inference(args_dict)
            except ValueError as e:
                print("\t"+str(e))
            print()



if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations")
    parser.add_argument('--data_root', type=str, default="D:\\Documents\\VS2022Projects\\sewer-ml\\Data\\train13")
    parser.add_argument('--batch_size', type=int, default=16, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument("--results_output", type=str, default="D:\\Documents\\VS2022Projects\\sewer-ml\\output")
    parser.add_argument("--log_input", type=str, default="D:\\Documents\\VS2022Projects\\sewer-ml\\logs\\xie2019_binary\\binary-version_1")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])

    args = vars(parser.parse_args())

    iterateResultsDirs(args)