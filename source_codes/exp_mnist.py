from data_transformations import mnist_transform
import torchvision.transforms as transforms
from data_loaders import mnist_dataloader
from loss_funcs import CrossEntropyLoss
from models import VGG11
from trainers import Trainer, MPTrainer
from testers import Tester
import numpy as np

import matplotlib.pyplot as plt
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
model

print("Total number of parameters =", np.sum(
    [np.prod(parameter.shape) for parameter in model.parameters()]))

trainer = Trainer(model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                  train_epochs=20, valid_epochs=2, learning_rate=0.001, loss_func=loss_func, optimization_method='Adam')


trainer.load_model('saved_models/vgg_mnist.model')


import torch
import time

from models import LRPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_relevance_scores(
    x: torch.tensor, r: torch.tensor, name: str,) -> None:
    """Plots results from layer-wise relevance propagation next to original image.
    Method currently accepts only a batch size of one.
    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
    """
    max_fig_size = 20

    _, _, img_height, img_width = x.shape
    max_dim = max(img_height, img_width)
    fig_height, fig_width = (
        max_fig_size * img_height / max_dim,
        max_fig_size * img_width / max_dim,
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))

    x = x[0].squeeze().permute(1, 2, 0).detach().cpu()
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    axes[0].imshow(x)
    axes[0].set_axis_off()

    r_min = r.min()
    r_max = r.max()
    r = (r - r_min) / (r_max - r_min)
    axes[1].imshow(r, cmap="afmhot")
    axes[1].set_axis_off()

    fig.tight_layout()
    plt.show()


lrp_model = LRPModel(model=model, top_k=0.02)

for i, (x, y) in enumerate(test_dataloader):
    x = x.to(device)
    # y = y.to(device)  # here not used as method is unsupervised.

    t0 = time.time()
    r = lrp_model.forward(x)
    print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

    plot_relevance_scores(x=x, r=r, name=str(i))