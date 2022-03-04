from gc import callbacks
import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import collect_env

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning.callbacks.progress import TQDMProgressBar 

from datetime import datetime

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=256, num_workers=10)

    # information that python/python asks for
    print("****************************")
    collect_env.main()
    print("****************************", flush=True)

    # Initialize a trainer
    trainer = Trainer(

        callbacks=[TQDMProgressBar(refresh_rate=1)],
        gpus=[0,1],
        max_epochs=100, # about 2 minutes and 10 seconds on 1 A6000 ie: change gpu=[0] and comment out strategy='ddp'
        accelerator="gpu",
        strategy='ddp'
    )



    # Train the model 
    print("start = ", datetime.now(), flush=True)
    trainer.fit(mnist_model, train_loader)
    print("end = ", datetime.now(), flush=True)
    

if __name__ == '__main__':
    main()
