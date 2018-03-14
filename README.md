# A Mixed-Scale Dense Convolutional Neural Network for image analysis

This repository is a simplified implementation of the paper: [A mixed-scale dense convolutional neural network for image analysis](http://www.pnas.org/content/early/2017/12/21/1715832114), for image segmentation. The implementation uses the [PyTorch](https://pytorch.org) framework.

> Note: Work in progress. Any contribution is appreciated.


The files are organized as follows:

```
.
├── config.py (contains the configuration class which handles reading config for experiments)
├── data
│   └── README.md
├── data_handler.py (data handler class for making and modifying datasets)
├── experiment
│   └── cfg.yml (template and default config file [DO NOT REMOVE])
├── main.py (main file that runs the model)
├── model.py (model architecture specifications)
├── README.md
└── utils.py (utility functions used throughout the code)

```

# Creating an Experiment

To create a new experiment, create the following directory structure.

```
.
├── Annotations (Stores annotations for the generated segments from the model [TODO])
├── cfg.yml (configuration file for the experiment)
├── checkpoints (Saves model checkpoints for selective use)
└── output (stores the generated segmented images)
    └── training (images generated during the training process)
```

## To run an experiment

The `main.py` script is used to run the experiment. To train the model, without using a pretrained checkpoint, to write the images in experiment directory, run the following command:

```
python main.py --exp_dir=<EXP_DIR> --cfg=<CONFIG_PATH> --nopretrained --write_images --train
```

To just run the model you have trained, update the config file with the path to the latest checkpoint and run the following command:

```
python main.py --exp_dir=<EXP_DIR> --cfg=<CONFIG_PATH> --pretrained --write_images --train
```

# Dataset

The structure used when creating the dataset is as follows:

```.
├── README.md (contains any information about the dataset)
├── top (contains the RGB images)
└── gt (ground truth data)
```
This structure is to be used for all dataset creation and adaptation pruposes. Change congif file and `config.py` for different dataset.



## Acknowledgements

Thanks to the authors of the Paper: [A mixed-scale dense convolutional neural network for image analysis](http://www.pnas.org/content/early/2017/12/21/1715832114) (Pelt, D. M. *et. al.*)