## A generic framework for distributed training with PyTorch.

### Features

With this framework, your can:

* High extensibility: customize your algorithm for any purpose.
* High-efficiency distributed training, validation, evaluation, feature extraction.
* Warmlup learning rate in multiple steps.
*

### Algorithms implemented

1. Image classification: Cifar, ImageNet, Face
2. Autoencoder, VAE (coming soon)
3. GAN, CGAN (coming soon)
4. Self-sup algorithms: colorization, conditional-motion-propagation (coming soon)
5. Semantic Segmentation (coming soon)

### Requirements

* PyTorch >= 0.4.1
* Others:

    ```
    pip install -r requirements.txt
    ```

### Usage

    * For example, Cifar resnet18

    ```
    sh experiments/Cifar/resnet18/train.sh # train, don't forget to open tensorboard for visualization
    sh experiments/Cifar/resnet18/resume.sh $ITER # resume from iteration $ITER
    sh experiments/Cifar/resnet18/validate.sh $ITER # offline validation
    sh experiments/Cifar/resnet18/evaluate.sh $ITER # offline evaluation
    sh experiments/Cifar/resnet18/extract.sh $ITER # feature extraction
    ```

### Extensibility

    * You need to write your own Dataset in `dataset.py` and your algorithm under `models` (refer to `models/classification.py`), and design your config file. That't it!

### Note

    * Please use `sh scripts/kill.sh` to kill.
