'''

ResNet18 for clarification of FashionMNIST

Created by Zelun Pan
04/04/2023

'''
import torch
import configuration as cfg
import ResidualBlock
import configuration
import train_test
import Model
import Dataset
import save_load


torch.manual_seed(3407)
# model = Model.model().to(cfg.device)
model = save_load.load('model.pt').to(cfg.device)
training_set, testing_set = Dataset.load_data()
test_acc_init = train_test.test(model, testing_set)
train_test.train(model, training_set, testing_set)
test_acc = train_test.test(model, testing_set)
print('Test accuracy is: {:.4}%'.format(test_acc))

if test_acc_init < test_acc:
    save_load.save(model)





