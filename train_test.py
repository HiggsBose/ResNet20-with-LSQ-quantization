import torch
from torch import nn, optim
import configuration as cfg
import operator

def test_net(model, test_set, accuracy_list=[], require_vis=False):
    '''
    test the network's performance on the test set

    :param model: the model to be evaluated
    :param test_set: the test dataset
    :return: the inference accuracy of the model on the test dataset
    '''

    model.eval()
    test_correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            img, label = data
            img, label = img.cuda(), label.cuda()
            if require_vis:
                model.require_vis = require_vis
                output, list = model(img)
            else:
                output = model(img)
            _, prediction = torch.max(output.data, 1)
            total += label.size(0)
            test_correct += (prediction == label).sum().item()
    print('Test acc: {:.4}%'.format(100 * test_correct / total))
    accuracy_list.append(100 * test_correct / total)
    acc = 100 * test_correct / total
    if require_vis:
        return acc, list
    else:
        return acc


def train_net(model, train_set, writer, test_set=None, evaluation=False):
    '''
    train the model on the training set (evaluate the performance on test set if eval is True
    if evaluation is True, then the test_set must be given

    :param model: the model to be trained
    :param train_set: the training dataset
    :param test_set: the test dataset
    :param evaluation: choose whether to evaluate the model or not
    :return: the trained model
    '''
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    cost = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': weight_p, 'weight_decay': cfg.config["weight_decay"]},
                           {'params': bias_p, 'weight_decay': 0}], lr=cfg.config["lr"], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1, last_epoch=-1)

    loss_list = []
    accuracy_list = []
    best_acc = 0
    iterations = 0
    model_param = []
    for epoch in range(cfg.config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        print('Epoch {}/{}'.format(epoch+1, cfg.config["num_epochs"]))
        print('-'*30)

        total_train = 0
        i = 0
        for data in train_set:
            img, label = data
            img, label = img.to(cfg.config["device"]), label.to(cfg.config["device"])
            # print(img)
            output = model(img)
            loss = cost(output, label)
            train_loss += loss.item()
            _, prediction = torch.max(output, 1) # predict the largest value's tag
            total_train += label.size(0)
            num_correct = (prediction == label).sum()
            train_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            iterations += 1
            writer.add_scalar(tag='training loss', scalar_value=loss, global_step=iterations)
            if i % 100 == 0:
                print('[%d, %5d] training_loss: %f' % (epoch + 1, i, train_loss / 100))
                loss_list.append(train_loss / 100)
                train_loss = 0.0
        print('Train acc:{:.4}%'.format(100*train_acc/total_train))
        writer.add_scalar(tag='training accuracy', scalar_value=train_acc/total_train, global_step=epoch)

        scheduler.step()

        if evaluation:
            acc = test_net(model, test_set, accuracy_list)
            writer.add_scalar(tag='test accuracy', scalar_value=acc, global_step=epoch)
            if acc > best_acc:
                best_acc = acc
                best_acc_loc = epoch + 1

            print('test best acc:{}% at epoch {}'.format(best_acc, best_acc_loc))

    return model