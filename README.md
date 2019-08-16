## A generic framework for distributed training with PyTorch.

### Features

With this framework, your can:

* High extensibility: customize your algorithm for any purpose.
* High-efficiency distributed training, validation, evaluation, feature extraction.
* Warmlup learning rate in multiple steps.
* 

### Requirements

* PyTorch >= 0.4.1
* Others:

    ```
    pip install -r requirements.txt
    ```

### Train

    ```
    sh experiments/ImageNet/resnet50/train.sh
    ```

### Note

    * Please use `sh kill.sh` to kill.
