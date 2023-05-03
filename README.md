# Training and saving models with CML on a dedicated AWS EC2 runner

The files in this repository provide an example on how to use a dedicated runner on AWS to train and export a machine learning model. It accompanies [this blog post](https://dvc.org/blog/CML-runners-saving-models-1), which contains a full guide on how to achieve this.

## Contents
This repository contains the following files:

- `requirements.txt`: the packages necessary for training our model.
- `get_data.py`: script that generates sample data to train a model on.
- `train.py`: script that trains a model on the generated data and exports that model to a binary file, along with a confusion matrix and some metrics.
- `.github/workflows/train-test.yaml`: example workflow that provisions an AWS EC2 instance to run `train.py` and export the resulting model to a DVC repo.

## How to install and run
- Clone this repository.
- `python3 -m venv venv`
- `pip install -r requirements.txt`
- `dvc init`
- `aws configure`
- Add `PERSONAL_ACCESS_TOKEN`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to the secrets of your repo.
