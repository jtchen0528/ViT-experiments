# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import argparse

import models.model_utils as model_utils

from datautil.dataset import CreateDataset

# %%

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", help="epoch", type=int, default=100)
parser.add_argument("-bs", "--batch_size",
                    help="batch size", type=int, default=32)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate", type=float, default=2e-5)
parser.add_argument("-train_dir", "---train_dir", help="train data directory",
                    type=str, default='dataset/tiny-imagenet-200/train')
parser.add_argument("-test_dir", "---test_dir", help="test data directory",
                    type=str, default='dataset/tiny-imagenet-200/val')
parser.add_argument("-m", "--model_name",
                    help="vit model type", type=str, default='vit')
parser.add_argument("-img_size", "--img_size",
                    help="image size", type=int, default=64)
parser.add_argument("-patch_size", "--patch_size",
                    help="patch size", type=int, default=8)
parser.add_argument("-gpu", "--gpu_id", help="gpu id", type=int, default=0)

args = parser.parse_args()

# %%
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
TRAIN_DS_PATH = args.train_dir
TEST_DS_PATH = args.test_dir
MODEL_NAME = args.model_name
IMG_SIZE = args.img_size
PATCH_SIZE = args.patch_size
GPU_ID = args.gpu_id
CUDA = torch.cuda.is_available()
# %%

transform = transforms.Compose(
    [transforms.ToTensor()])

train_ds = CreateDataset(folder=TRAIN_DS_PATH, transform=transform)

classes = train_ds.classes

test_ds = CreateDataset(folder=TEST_DS_PATH, transform=transform)
# %%
# Define Model

model = model_utils.create_model(
    model_name=MODEL_NAME, img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=len(classes))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

if CUDA:
    model.to(GPU_ID)

# %%
print("Number of train samples: ", len(train_ds))
print("Number of test samples: ", len(test_ds))

train_loader = data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=True)

# %%
best_eval = 0
os.makedirs('checkpoints/' + MODEL_NAME, exist_ok=True)
train_loss_list = []
test_loss_list = []
acc_top1_list = []
acc_top5_list = []
# Train the model
for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(train_loader):
        # Change input array into list with each batch being one element
        x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
        # Remove unecessary dimension
        for index, array in enumerate(x):
            x[index] = np.squeeze(array)
        # Send to GPU if available
        x = torch.tensor(x)
        y = torch.tensor(y)
        if torch.cuda.is_available():
            x, y = x.to(GPU_ID), y.to(GPU_ID)
        b_x = Variable(x)   # batch x (image)
        b_y = Variable(y)   # batch y (target)
        # Feed through model
        output = model(b_x)
        # Calculate loss
        loss = loss_func(output, b_y)
        train_loss_list.append(round(loss.item(), 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            # Get the next batch for testing purposes
            test = next(iter(test_loader))
            test_x = test[0]
            # Reshape and get feature matrices as needed
            test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
            for index, array in enumerate(test_x):
                test_x[index] = np.squeeze(array)
            # Send to appropirate computing device
            test_x = torch.tensor(test_x)
            test_y = torch.tensor(test[1])
            if torch.cuda.is_available():
                test_x = test_x.to(GPU_ID)
                test_y = test_y.to(GPU_ID)
            # Get output (+ respective class) and compare to target
            test_output = model(test_x)
            loss = loss_func(test_output, test_y)
            test_loss_list.append(round(loss.item(), 2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_output_top1 = test_output.argmax(1)
            test_output_top5_val, test_output_top5 = test_output.topk(5, dim=1, largest=True, sorted=True)

            # Calculate Accuracy

            accuracy_top1 = (test_output_top1 ==
                             test_y).sum().item() / BATCH_SIZE
            accuracy_top5 = 0
            for i in range(BATCH_SIZE):
                test_output_top5_list = test_output_top5[i].tolist()
                test_y_item = test_y[i].item()
                print(test_y_item, test_output_top5_list)
                if (test_output_top5_list.index(test_y_item) >= 0):
                    accuracy_top5 += 1
            accuracy_top5 = accuracy_top5 / BATCH_SIZE
            acc_top1_list.append(round(accuracy_top1, 2))
            acc_top1_list.append(round(accuracy_top5, 2))
            print('Epoch: ', epoch, '| train loss: %.4f' %
                  loss, '| top1 accuracy: %.2f' % accuracy_top1, '| top5 accuracy: %2f' % accuracy_top5, end="\r", flush=True)
            if accuracy_top1 > best_eval:
                best_eval = accuracy_top1
                torch.save(model.state_dict(), "checkpoints/{}/best_val_im{}_p{}_lr{}.pth".format(
                    args.model_name, args.img_size, args.patch_size, args.learning_rate))
    loss_file = open("checkpoints/{}/loss.txt", 'w')
    acc_file = open("checkpoints/{}/acc.txt", 'w')
    loss_file.write(
        {
            'Train Loss': train_loss_list,
            'Test Loss': test_loss_list,
        }
    )
    acc_file.write(
        {
            'Accuracy Top1': acc_top1_list,
            'Accuracy Top5': acc_top5_list,
        }
    )
    os.remove("checkpoints/{}/ep{}_im{}_p{}_lr{}.pth".format(args.model_name,
              epoch, args.img_size, args.patch_size, args.learning_rate))
    torch.save(model.state_dict(), "checkpoints/{}/ep{}_im{}_p{}_lr{}.pth".format(
        args.model_name, epoch + 1, args.img_size, args.patch_size, args.learning_rate))

# %%
