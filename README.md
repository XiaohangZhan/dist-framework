## A prototype for distributed training with PyTorch.

### Note
This repo will not be maintained.

### Features

With this framework, you get:

* High extensibility: customize your algorithm for any purpose.
* High-efficiency distributed training, validation, evaluation, feature extraction.

### Requirements

* PyTorch >= 0.4.1
* Others:

    ```sh
    pip install -r requirements.txt
    ```

### Usage

* For example, train Cifar-10 with resnet20 in 14 minutes, get 92.59% accuracy.

    ```sh
    cd dist-framework
    mkdir data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    cd ..
    sh experiments/classification/Cifar/resnet20/train.sh # train, don't forget to open tensorboard for visualization
    sh experiments/classification/Cifar/resnet20/resume.sh $ITER # resume from iteration $ITER
    sh experiments/classification/Cifar/resnet20/validate.sh $ITER # offline validation
    sh experiments/classification/Cifar/resnet20/evaluate.sh $ITER # offline evaluation
    sh experiments/classification/Cifar/resnet20/extract.sh $ITER # feature extraction
    ```

### Extensibility

* You need to write your own Dataset in `dataset.py` and your algorithm under `models` (refer to `models/classification.py`), and design your config file. That't it!

### Note

* Please use `sh scripts/kill.sh` to kill.
