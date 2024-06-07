# Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark

This repository is the official implementation of [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Haurum_Sewer-ML_A_Multi-Label_Sewer_Defect_Classification_Dataset_and_Benchmark_CVPR_2021_paper.pdf) and [Supplementary](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Haurum_Sewer-ML_A_Multi-Label_CVPR_2021_supplemental.pdf).


The Sewer-ML project page can be found [here](https://vap.aau.dk/sewer-ml/).

## Requirements

The anaconda environments used to run the code are included in the anaconda_environment folder.

The main packages are listed below:

- Pytorch 1.6.0
- Torchvision 0.7.0
- Pytorch-Lightning 0.9.0
- Pandas >= 1.1.1
- Numpy >= 1.15
- Scikit-Learn 0.19.2
- lear-gist-python
- inplace-abn


The Sewer-ML dataset can be accessed after filling out this [Google Form](https://forms.gle/hBaPtoweZumZAi4u9).
The Sewer-ML dataset is licensed under the Creative Commons BY-NC-SA 4.0 license.


## Training

The models can be trained in four different "training modes", based on the type of classifier is wanted:
- training_mode = e2e: An end-to-end classifier is trained on all the data with multi-label annotations.
- training_mode = binary: A binary classifier is trained on all the data with binary defect annotations.
- training_mode = defect: A defect classifier is trained on just the data with defects occuring, with multi-label annotations.
- training_mode = binaryrelevance: A binary classifier is trained on the data with just the defect denoted in the "br_defect" argument.


### CNN classifiers

When training the images are normalized with the following mean and standard deviation, found using the calculate_normalization.py script.:
- mean = [0.523, 0.453, 0.345]
- std = [0.210, 0.199, 0.154]

Two examples of this would be training the binary classifier by Xie et al. and a e2e model using the TResNet-L architecture:

```
python lightning_trainer.py --precision 16 --batch_size 128 --max_epochs 90 --progress_bar_refresh_rate 500 --gpus 2 --distributed_backend ddp  --model xie2019_binary  --training_mode binary --log_save_interval 1000 --row_log_interval 100 --ann_root <path_to_annotations> --data_root <path_to_data> --log_save_dir <path_to_model_logs>
```

```
python lightning_trainer.py --precision 16 --batch_size 128 --max_epochs 90 --progress_bar_refresh_rate 500 --gpus 2 --distributed_backend ddp  --model tresnet_l  --training_mode e2e --log_save_interval 1000 --row_log_interval 100 --ann_root <path_to_annotations> --data_root <path_to_data> --log_save_dir <path_to_model_logs>
```


### Extra trees classifier

To train an Extra Tree classifier on the data, first the GIST features needs to be extracted.
This is done using the gist_extractor.py script, utilizing the [Lear-GIST-Python package](https://github.com/whitphx/lear-gist-python).

```
python gist_extractor.py --data_root <path_to_data> --ann_root <path_to_annotations> --output_dir <path_to_gist_features>
```

When calling the myrans_trainer.py script, a grid search is performed over the number of trees, the max depth of the trees and maximum number of features used.

```
python myrans_trainer.py --stage <training_mode> --output_dir <path_to_model_output> --gist_dir <path_to_gist_features>
```

The "stage" argument must be supplied indicating whether it is trained on binary data, defect data, or all the data.



## Evaluation

To evaluate a set of models on the validation set of the Sewer-ML dataset, first the raw predictions for each image should be generated, which is subsequently compared to the ground truth. The raw predictions should be probabilities.

When the predictions have been obtained the performance of the model can be determined using the calculate_results.py script.

```
python calculate_results.py --output_path <path_to_metric_results> --split <dataset_split_to_use> --score_path <path_to_predictions> --gt_path <path_to_annotations>
```


### CNN classifiers

The validation prediction of the CNN classifiers when trained using the lighting_trainer script can be obtained the iterate_results_dir.py script. The script iterates over a directory contain a subdirectory per trained model.

```
python iterate_results_dir.py --ann_root <path_to_annotations> --data_root <path_to_data> --results_output <path_to_results> --log_input <path_to_model_logs> --split <dataset_split_to_use>
```

If a single model needs to be evaluted this can be done using the inference.py script. Additionally, if a specific set of weights needs to be used, this can be done by setting the --best_weights flag. Otherwise it is expected that there is a last.ckpt file which points to the best performing model weights.

```
python inference.py --ann_root <path_to_annotations> --data_root <path_to_data> --results_output <path_to_results> --model_path <path_to_models> --split <dataset_split_to_use>
```


### Extra trees classifier

The extra trees classifiers can be evaluated using the myrans_inference.py script.

```
python myrans_inference.py --model_path <path_to_models> --gist_dir <path_to_gist_features> --output_dir <path_to_results> --split <dataset_split_to_use>
```


## Combining binary and defect/e2e results
Some methods utilize a two-stage approach.

The results of the two trained networks can be combined using the combine_two_stage.py script. The results can be combined in two ways: "replace" and "multiply". For all our results the replace method was used.

The replace method uses the defect probability of the first stage for all defects if a normal pipe is detected. Otherwise the second stage defect probabilities are used.

The multiply method multiplies the defect probability of the first stage with the individual defect probabilites fro mthe second stage.

```
python combineTwoStage.py --outputName <output_filepath> --stage_one_path <stage_one_filepath> --stage_two_path <stage_two_filepathh> --split <dataset_split_to_use> --strategy <how_to_merge_predictions>
```


## Pre-trained Models

You can download pretrained models here:

- [Model Repository](https://sciencedata.dk/shared/sewerml_cvpr2021_models) trained on Sewer-ML using the parameters described in the paper.

For the CNNs two model versions are provided: Weights which can be used in pure Pytorch and weights which are compatible with the Pytorch Lightning setup.

Each model weight file consists of a dict with the model state_dict and the most important model hyper_parameters:
- model
- num_classes
- training_mode
- br_defect

## Results
We compared six method from the sewer defect classification domain and size from the general multi-label classification domain. Some methods did not converge and results are not reported for these. The methods are evalauted using the F2-CIW and F1-Normal metrics. Details can be found in the paper.

### Sewer Defect Classification

| Model name         | F2-CIW (Val) | F1-Normal (Val) | F2-CIW (Tst) | F1-Normal (Tst) |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| Xie et al.   |     48.57%      |      91.08%    |     48.34%      |    90.62%       |
| Chen et al.   |     42.03%      |      3.96%    |     41.74%      |    3.59%       |
| Hassan et al.   |     13.14%      |      0.00%    |     12.94%      |    0.00%       |
| Myrans et al.   |     4.01%      |      26.03%    |    4.11%      |    27.48%       |
| ResNet-101   |     53.26%      |      79.55%    |    53.21%      |    78.57%       |
| KSSNet   |     54.42%      |      80.60%    |     54.55%      |    79.29%       |
| TResNet-M   |     53.83%      |      81.23%    |     54.79%      |    79.91%       |
| TResNet-L  |     54.63%      |      81.22%    |     53.75%      |    79,88%       |
| TResNet-XL   |     54.42%      |      81.81%    |    54.24%      |    80.42%       |
| Benchmark (Xie + TResNet-L)   |     55.36%      |      91.32%    |     55.11%      |    90.94%       |


A live leaderboard can be found at the associated [Codalab Challenge](https://competitions.codalab.org/competitions/32705).


## Code Credits

Parts of the code builds upon prior work:

- The GraphConvolutional layer is obtained from the ML-GCN authors implementation, an adaption of Thomas Kipf's original implementation. Found at: [https://github.com/tkipf/pygcn](https://github.com/tkipf/pygcn)
- Code to calculate the dataset mean and standard deviation was found at: [https://gist.github.com/pmeier/f5e05285cd5987027a98854a5d155e27](https://gist.github.com/pmeier/f5e05285cd5987027a98854a5d155e27)
- The TResNet model code comes from the Author's repository: [https://github.com/Alibaba-MIIL/TResNet](https://github.com/Alibaba-MIIL/TResNet])
- The evlauation code is partly based on the evaluation code from the ML-GCN authors: [https://github.com/Megvii-Nanjing/ML-GCN/](https://github.com/Megvii-Nanjing/ML-GCN/)
- The KSSNET Code was inspired by the Author's spatiotemporal model implementation: [https://github.com/mathkey/mssnet](https://github.com/mathkey/mssnet)
- The GIST feature descriptors are extracted using the Lear-GIST-Python package: [https://github.com/whitphx/lear-gist-python](https://github.com/whitphx/lear-gist-python)


## Contributing

The Code is licensed under an MIT License, with exceptions of the TResNet, ML-GCN, and KSSNET code which follows the license of the original authors.

The Sewer-ML Dataset follows the Creative Commons Attribute-NonCommerical-ShareAlike 4.0 (CC BY-NC-SA 4.0) International license.



## Bibtex
```bibtex
@InProceedings{Haurum_2021_CVPR,
    author    = {Haurum, Joakim Bruslund and Moeslund, Thomas B.},
    title     = {Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13456-13467}
}
```

