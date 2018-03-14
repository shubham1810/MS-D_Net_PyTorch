# Mixed-Scale Dense Convolutional Network

The files are organized as follows:

```
.
├── config.py (contains the configuration class which handles reading config for experiments)
├── data_handler.py (data handler class for making and modifying datasets)
├── experiment
│   └── cfg.yml (template and default config file [DO NOT REMOVE])
├── main.py (main file that runs the model)
├── model.py (model architecture specifications)
├── README.md
├── scripts
│   └── create_experiment.sh (script to create a new experiment)
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

The `main.py` script is used to run the experiment.
The command to run an experiment in vizualization mode is as follows:
Assuming, don't save output, use pretrained model, don't import VGG weights.

```
python main.py --exp_dir=<EXP_DIR> --cfg=<CONFIG_PATH> --pretrained --write_images=True --train
```

# Dataset

The structure used when creating the dataset is as follows:

```.
├── README.md (contains any information about the dataset)
├── top (contains the RGB images)
└── gt (ground truth data)
```
This structure is to be used for all dataset creation and adaptation pruposes. Change congif file and `config.py` for different dataset.
