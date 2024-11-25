import os 


import torchvision 
from torch.utils.data import DataLoader 
from torchvision.transforms import Compose, Lambda, ToTensor


def download_data():
    os.makedirs('data/mnist', exist_ok=True)
    mnist = torchvision.datasets.MNIST('data/mnist', download=True)
    print(f"MNIST dataset downloaded with size {len(mnist)} and tensor shape {ToTensor()(mnist[0][0]).shape}")



def get_data_loader(batch_size:int):
    transform = Compose([ToTensor(), Lambda(lambda x:(x-0.5)*2)])
    dataset = torchvision.datasets.MNIST(root = './data/mnist', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_mnist_tensor_shape():
    return (1,28,28)

if __name__ == '__main__':
    download_data()
