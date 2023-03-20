# ResNet20 with LSQ quantization and noise-aware training
A ResNet20 with LSQ quantization and noise aware training for in-memory computing based on RRAM

the models are based on ResNet and trained on CIFAR-10 dataset.

You can train and test the model as well as see the weight distribution using matplotlib in Python.

* configuration.py is used for setting the basic configurations of the network like learning rate(lr) and noise
* NoiseScheme.py is used for adding noise to the network to simulate mapping to the real hardware computing chips.  

* **please run the model using resnet20.py**
