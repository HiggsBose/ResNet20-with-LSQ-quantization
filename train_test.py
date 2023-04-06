import torch
import configuration as cfg


def train(model, training_set, testing_set):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(cfg.num_epoch):
        iteration = 0
        train_correct = 0
        total = 0
        for item in training_set:
            iteration += 1
            feature, label = item
            feature = feature.to(cfg.device)
            label = label.to(cfg.device)
            total += label.size(0)
            pred = model.forward(feature)
            prediction = torch.argmax(pred, dim=1)
            train_correct += (prediction == label).sum()

            loss = loss_function(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}, Iteration {},  loss is {}'.format(epoch, iteration, loss))

        training_acc = 100 * train_correct / total
        print('Epoch: {}, training accuracy: {:.4}%'.format(epoch, training_acc))

    return None


def test(model, testing_set):
    model.eval()
    with torch.no_grad():
        total_count = 0
        correct = 0
        for item in testing_set:
            total_count += 1
            feature, label = item
            feature = feature.to(cfg.device)
            label = label.to(cfg.device)
            pred = model(feature)
            pred_result = torch.argmax(pred)
            if pred_result == label:
                correct += 1

        return 100 * correct / total_count
