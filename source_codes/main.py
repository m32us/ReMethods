from data_transformations import mnist_transform
import torchvision.transforms as transforms

from data_loaders import mnist_dataloader

from loss_funcs import CrossEntropyLoss

from models import VGG11

from trainers import Trainer, MPTrainer
from testers import Tester

import numpy as np
from loggers import set_logger

set_logger(data_name='mnist', save_path='./loggers/log')

def tile_image(image):
    '''duplicate along channel axis'''
    return image.repeat(3,1,1)

transform=[
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: tile_image(x))
        ]

train_dataset, test_dataset = mnist_dataloader.get_dataset(
    './datasets', transform=mnist_transform(lst_trans_operations=transform))

print('Train data set:', len(train_dataset))
print('Test data set:', len(test_dataset))

train_dataloader, valid_dataloader, test_dataloader = mnist_dataloader.loader(
    train_dataset, test_dataset)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

loss_func = CrossEntropyLoss()

model = VGG11(n_classes=10).cuda()

print(model)

print("Total number of parameters =", np.sum(
    [np.prod(parameter.shape) for parameter in model.parameters()]))

trainer = MPTrainer(model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                    train_epochs=25, valid_epochs=2, learning_rate=0.001, loss_func=loss_func, optimization_method='adam')

# model, losses, accuracies = trainer.run()

# trainer.save_model('saved_models/vgg11_mnist.model')

model_loaded = trainer.load_model('saved_models/vgg11_mnist.model')

tester = Tester(model=model_loaded,
                test_dataloader=test_dataloader, use_gpu=True)

tester.run()
