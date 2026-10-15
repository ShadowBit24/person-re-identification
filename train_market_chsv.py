# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model_chs import ft_net, ft_net_dense, PCB
from random_erasing import RandomErasing
import json
import copy
import time
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard Integration

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='dataset/market_rename',type=str, help='training dir path')  # Change this to your dataset path
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--num_epochs', default=25, type=int, help='Number of epochs to train the model')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

if not os.path.exists("./model/"):
    os.makedirs("./model/")
    
    
######################################################################
# Custom Dataset Class
# --------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, hist_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and histograms.
            transform (callable, optional): Optional transform to be applied on an image.
            hist_transform (callable, optional): Optional transform to be applied on a histogram.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.hist_transform = hist_transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_file in os.listdir(cls_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(cls_dir, img_file)
                    hist_file = img_file.replace('.jpg', '_histogram.npy')
                    hist_path = os.path.join(cls_dir, hist_file)
                    samples.append((img_path, hist_path, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, hist_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        
        # Load and process histogram
        try:
            histogram = np.load(hist_path).astype(np.float32)
        except FileNotFoundError:
            print(f"Warning: Histogram file not found for {img_path}. Using zeros.")
            histogram = np.zeros((768,), dtype=np.float32)
        
        histogram = torch.tensor(histogram, dtype=torch.float)
        
        # Apply histogram transformations
        if self.hist_transform:
            histogram = self.hist_transform(histogram)

        return image, histogram, label

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    # Ensure all elements are moved to CPU and converted to NumPy arrays
    train_losses_np = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in train_losses]
    val_losses_np = [loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in val_losses]
    train_accs_np = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_accs]
    val_accs_np = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in val_accs]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses_np)), train_losses_np, label='Training Loss')
    plt.plot(range(len(val_losses_np)), val_losses_np, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs_np)), train_accs_np, label='Training Accuracy')
    plt.plot(range(len(val_accs_np)), val_accs_np, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()



######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
     train_all = '_all'

# CustomDataset instances for train and val data
train_dataset = CustomDataset(
    root_dir=os.path.join(data_dir, 'train' + train_all),
    transform=data_transforms['train'],
    hist_transform=None  # Modify this if you need to transform histograms
)

val_dataset = CustomDataset(
    root_dir=os.path.join(data_dir, 'val'),
    transform=data_transforms['val'],
    hist_transform=None  # Modify this if you need to transform histograms
)

# Data loaders
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)
}

# Print contents of train and val directories
train_dir = os.path.join(data_dir, 'train' + train_all)
val_dir = os.path.join(data_dir, 'val')

#Remove '#' in the following 4 lines to print out the contents of the directories
#print("Contents of train directory:")
#print(os.listdir(train_dir))

#print("Contents of val directory:")
#print(os.listdir(val_dir))

dataset_sizes = {x: len(train_dataset) if x == 'train' else len(val_dataset) for x in ['train', 'val']}
class_names = train_dataset.classes

use_gpu = torch.cuda.is_available()

inputs, histograms, classes = next(iter(dataloaders['train']))

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    writer = SummaryWriter(f'runs/{name}')  # For TensorBoard
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()

    # Metrics storage
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, histograms, labels in dataloaders[phase]:
                inputs, histograms, labels = inputs.to(device), histograms.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, histograms)
                    loss = sum(criterion(output, labels) for output in outputs)
                    _, preds = torch.max(outputs[0], 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Training complete in {time.time() - since // 60}m {time.time() - since % 60}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    save_network(model, 'last')

    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    writer.close()  # Close the TensorBoard writer

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names))
else:
    model = ft_net(len(class_names))

if opt.PCB:
    model = PCB(len(class_names))

print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, opt.num_epochs)

