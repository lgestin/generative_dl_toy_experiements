import torch
import torchvision


class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torchvision.datasets.MNIST(root=data_path)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torchvision.transforms.PILToTensor()(x)
        x = 2 * x / 255 - 1
        x = MNIST.normalize(x)
        return x, y

    @staticmethod
    def normalize(x):
        return (x + 0.8) / 0.3

    @staticmethod
    def unnormalize(x):
        return x * 0.3 - 0.8

    def __len__(self):
        return len(self.data)
