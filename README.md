# ResNet for FashionMNIST
基于ResNet实现对FashionMNIST的分类，
实现了一个类似ResNet18的结构，但其中不同的卷积层的channel数是随便调的，第一层卷积也没有使用7x7的卷积核，而是采用了3x3的卷积核.

本模型在FashionMNIST测试集上达到了92.68%的准确度

下面介绍各个文件意义：
* **ResNet.py**: 主函数，**运行model直接run此文件**，可以选择重新开始训练一个新模型或是继续训练已有的model.pt中预训练的模型，也可以注释掉
```train_test.train(model, training_set, testing_set)```只看测试结果
* **Model.py**: 定义网络的类，**网络的结构**可以在此查看
* **ResidualBlock.py**: 网络的组成单元，比如单个的Residual层，单个卷积层，单个Linear层
* **configuration.py**: 设置一些超参数，比如用GPU还是CPU跑model（自动设置），比如训练epoch总数，learning rate等等
* **Dataset.py**: 准备数据集，加载数据集并分为训练集与测试集，在训练集上，进行一系列data augmentation以增强模型的泛化能力
* **train_test.py**: 训练以及测试网络的函数，loss采用CrossEntropy，optim即为SGD，learning rate并未使用scheduler进行动态调整
* **save_load.py**: 定义存下以及加载model的函数，可以每次加载预训练好的model进行测试，或继续训练进行fine tune
