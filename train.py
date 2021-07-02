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

import models.model_utils as model_utils

from datautil.dataset import CreateDataset
# %%
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
TRAIN_DS_PATH = 'data/tiny-imagenet-200/train/'
TEST_DS_PATH = 'data/tiny-imagenet-200/val/'
MODEL_NAME = 'vit'
IMG_SIZE = 64
PATCH_SIZE = 8
GPU_ID = 0
CUDA = torch.cuda.is_available()
# %%

transform = transforms.Compose(
    [transforms.ToTensor()])

train_ds = torchvision.datasets.ImageFolder(
    TRAIN_DS_PATH, transform=transform)

classes = train_ds.classes
classes_with_id = train_ds.class_to_idx

eval_ds = CreateDataset(folder='data/tiny-imagenet-200/val/',
                        transform=transform, classes=classes)
test_ds = CreateDataset(folder=TEST_DS_PATH,
                        transform=transform, classes=classes)
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_output = test_output.argmax(1)
            # Calculate Accuracy
            print(test_output[:10])
            print(test_y[:10])
            accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
            print('Epoch: ', epoch, '| train loss: %.4f' %
                  loss, '| test accuracy: %.2f' % accuracy)
            if accuracy > best_eval:
                best_eval = accuracy
                torch.save(model.state_dict(), "checkpoints/vit.pth")
