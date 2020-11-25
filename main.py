import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import ADataset, AugmixDataset, data_transforms_pre_train, data_transforms_train, data_transforms_val

## We resample a validation dataset

train_val_data = torch.utils.data.ConcatDataset([datasets.ImageFolder(data_folder + '/train_images',
                         transform=None),
                         datasets.ImageFolder(data_folder + '/val_images', transform=None)
                  ])
train_dataset_length = int(len(train_val_data)*0.9)
lengths = [train_dataset_length,len(train_val_data)-train_dataset_length]
train_data, val_data = torch.utils.data.random_split(train_val_data,lengths,generator=torch.Generator().manual_seed(42))

train_dataset = AugmixDataset(train_data, transform = [data_transforms_pre_train,data_transforms_train])
val_dataset = ADataset(val_data, transform = data_transforms_val)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, 
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, 
    shuffle=False)

from model import VitNetAug
# Creation and training of the model
model = VitNetAug(args.lr,args.momentum)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.experiment,
    filename='vit-finetuning-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min')
trainer = pl.Trainer(callbacks=[checkpoint_callback],gpus=1,log_every_n_steps=args.log_interval, max_epochs = args.epochs)

trainer.fit(model, train_loader,val_loader)