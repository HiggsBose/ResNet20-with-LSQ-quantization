from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import configuration as cfg


def load_data(augmentation=True):
    '''
    load the training dataset and the test dataset, perform data augmentation by default

    :param augmentation: True by default, choose whether use augmentation or not
    :return: training set loader train_loader & test set loader test_loader
    '''

    # download & prepare dataset
    if augmentation:
        train_tf = transforms.Compose(      # define data preprocess transformations
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )   # use transformations to generate new images to enlarge the training dataset and to avoid overfitting
    else:
        train_tf = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )

    test_tf = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)]
    )

    train_set = datasets.CIFAR10('./data', train=True, transform=train_tf, download=True)
    test_set = datasets.CIFAR10('./data', train=False, transform=test_tf, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.config["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=cfg.config["batch_size"], shuffle=False)

    return train_loader, test_loader