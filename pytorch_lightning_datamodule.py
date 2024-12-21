import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

class AdvancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, dataset_name='MNIST'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        # Define transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download data if needed
        if self.dataset_name == 'MNIST':
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)
        elif self.dataset_name == 'CIFAR10':
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split data into train, validation, and test sets
        if self.dataset_name == 'MNIST':
            dataset = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        elif self.dataset_name == 'CIFAR10':
            dataset = CIFAR10(self.data_dir, train=True, transform=self.transform_cifar)
            self.cifar_train, self.cifar_val = random_split(dataset, [45000, 5000])
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform_cifar)

    def train_dataloader(self):
        if self.dataset_name == 'MNIST':
            return DataLoader(self.mnist_train, batch_size=self.batch_size)
        elif self.dataset_name == 'CIFAR10':
            return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.dataset_name == 'MNIST':
            return DataLoader(self.mnist_val, batch_size=self.batch_size)
        elif self.dataset_name == 'CIFAR10':
            return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.dataset_name == 'MNIST':
            return DataLoader(self.mnist_test, batch_size=self.batch_size)
        elif self.dataset_name == 'CIFAR10':
            return DataLoader(self.cifar_test, batch_size=self.batch_size)

# Example usage
data_module = AdvancedDataModule(dataset_name='CIFAR10')
data_module.prepare_data()
data_module.setup('fit')
train_loader = data_module.train_dataloader()
