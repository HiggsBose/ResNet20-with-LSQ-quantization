# ResNet20 with LSQ quantization and noise-aware training
A ResNet20 with LSQ quantization and noise aware training for in-memory computing based on RRAM.

The network is based on Kaiming He's paper, and the quantization shceme is LSQ-Net proposed by Steven K. Esser and et al. from IBM. It can be found on arXiv:1902.08153. 

The models are based on ResNet and trained on CIFAR-10 dataset.

You can train and test the model as well as see the weight distribution using matplotlib in Python.

* configuration.py is used for setting the basic configurations of the network like learning rate(lr) and noise
* NoiseScheme.py is used for adding noise to the network to simulate mapping to the real hardware computing chips.  
* QuantizationScheme.py is used for defining quantization method of weights and activations of the network.
* train_test.py defines the training and testing method of the network.
* Model.py defines the structure of the network
* BuildingBlock.py is the basic building blcok of ResNet

* **please run the model using resnet20.py**
