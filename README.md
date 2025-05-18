# Settlement and Electricity Detection

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-indigo.svg?style=for-the-badge&logo=PyTorch%20Lightning)](https://lightning.ai/docs/pytorch/stable/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![WeightsAndBiases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](http://wandb.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://github.com/cs175cv-w2024/final-project-team-tsvt/blob/main/LICENSE)

</div>

## Authors
<div align="center">

[<img src="author-cards/Author_Paul.png" width="22%">](https://github.com/trantripau1)
[<img src="author-cards/Author_Wilson.png" width="22%">](https://github.com/thewontonsoup)
[<img src="author-cards/Author_Luis.png" width="22%">](https://github.com/MrSpookyAngel)
[<img src="author-cards/Author_Emmanuel.png" width="22%">](https://github.com/Emmanuel-7orres)

</div>

## The Data
** The following description is taken directly from the IEEE GRSS 2021 Challenge [website](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/).

The IEEE GRSS 2021 ESD dataset is composed of 98 tiles of 800×800 pixels, distributed respectively across the training, validation and test sets as follows: 60, 19, and 19 tiles. Each tile includes 98 channels from the below listed satellite images. Please note that all the images have been resampled to a Ground Sampling Distance (GSD) of 10 m. Thus each tile corresponds to a 64km2 area.

## Prerequisite
- Python 3.10.11: Install from [Python's official website](https://www.python.org/)
- CUDA Toolkit 11.8.0: Install from [NVIDIA’s official website](https://developer.nvidia.com/cuda-toolkit-archive)

## Setting Up virtual environment

To make sure you download all the packages to begin this homework assignment we will utilize a Python virtual environment which is an isolated environment that allows you to run the homework with its own dependencies and libraries independent of other Python projects you may be working on. Here's how to set it up:

1. Create a virtual environment:
   
   `python3 -m venv venv`

2. Activate the virtual environment:
   * On macOS and Linux:
  
        `source venv/bin/activate`

   * On Windows:
  
        `venv\Scripts\activate`

3. (Optional) We suggest calling removing any problematic cached pip packages:

   `pip cache purge`

4. Install the required packages:

    `pip install -r requirements.txt`

## Directions

Clone this repository e.g. `git clone git@github.com:cs175cv-w2024/final-project-team-tsvt.git`

Please download and unzip the `dfc2021_dse_train.zip` saving the `Train` directory into the `data/raw` directory.
The zip file is available at the following [url](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view?usp=sharing).

To train your own model:
- Use the default values of the ESDConfig in the provided scripts/train.py or configure the provided scripts/train.yml to the desired model and parameters
- Run the script by entering `python -m scripts.train`
- Supported Models: `SegmentationCNN`, `UNet`, `FCNResnetTransfer`, `UNetPlusPlus`
- An example for running the `Segmentation CNN` architecture for 5 max epochs you would enter:
  
`python -m scripts.train --model_type=SegmentationCNN --max_epochs=5`



For running hyperparameter sweeps:
- Use the default values of the ESDConfig in the provided scripts/train.py and/or configure the provided scripts/train.yml to the desired model and parameters
- Run the script by entering `python -m scripts.train_sweeps`

To evaluate your model:
- Use the default values of the EvalConfig in the provided scripts/evaluate.py or configure the provided scripts/evaluate.yml to the desired model and parameters
- Run the script by entering `python -m scripts.evaluate`

Note: You may be prompted for a Weights & Balances account, enter “3” to skip login, otherwise enter “1” or “2” to create and connect an account

Our models can be found [here](https://drive.google.com/drive/folders/1EHMJ0rXebZ8dB1j0Cs60rkAhq8sESRjo?usp=drive_link).






