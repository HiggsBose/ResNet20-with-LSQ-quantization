import torchvision.datasets
import torchvision as TV
from torch.utils.data import DataLoader
import configuration as cfg

def load_data():
    # define data augmentation to prevent over fitting
    train_tf = TV.transforms.Compose([TV.transforms.RandomCrop(28, padding=4),
                                     TV.transforms.RandomHorizontalFlip(p=0.5),
                                     TV.transforms.RandomVerticalFlip(p=0.5),
                                     TV.transforms.ToTensor(),
                                     TV.transforms.Normalize(mean=0.5, std=0.5)])
    test_tf = TV.transforms.Compose([TV.transforms.ToTensor(),
                                     TV.transforms.Normalize(mean=0.5, std=0.5)])

    training = torchvision.datasets.FashionMNIST('./data', train=True, transform=train_tf, download=True)
    testing = torchvision.datasets.FashionMNIST('./data', train=False, transform=test_tf, download=True)
    training_set = DataLoader(training, batch_size=cfg.batch_size, shuffle=True)
    testing_set = DataLoader(testing, batch_size=1, shuffle=True)

    return training_set, testing_set
