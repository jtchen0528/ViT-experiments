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
import argparse
import json

import models.model_utils as model_utils

from datautil.dataset import CreateDataset

# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "-mode", "--mode", help="mode", type=str, default='TRAIN')
parser.add_argument("-pth", "--model_path",
                    help="load previous model folder", type=str, default='')

parser.add_argument("-e", "--epoch", help="epoch", type=int, default=100)
parser.add_argument("-bs", "--batch_size",
                    help="batch size", type=int, default=32)
parser.add_argument("-lr", "--learning_rate",
                    help="learning rate", type=float, default=2e-5)
parser.add_argument("-train_dir", "---train_dir", help="train data directory",
                    type=str, default='dataset/tiny-imagenet-200/train')
parser.add_argument("-test_dir", "---test_dir", help="test data directory",
                    type=str, default='dataset/tiny-imagenet-200/val')
parser.add_argument("-eval_dir", "---eval_dir", help="evaluation data directory",
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
MODE = args.mode
MODEL_PATH = args.model_path
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
TRAIN_DS_PATH = args.train_dir
TEST_DS_PATH = args.test_dir
EVAL_DS_PATH = args.eval_dir
MODEL_NAME = args.model_name
IMG_SIZE = args.img_size
PATCH_SIZE = args.patch_size
GPU_ID = args.gpu_id
CUDA = torch.cuda.is_available()
# %%

transform = transforms.Compose(
    [transforms.ToTensor()])
if MODE == 'TRAIN':
    train_ds = CreateDataset(folder=TRAIN_DS_PATH, transform=transform)

    classes = train_ds.classes

    test_ds = CreateDataset(folder=TEST_DS_PATH, transform=transform)
elif MODE == 'EVAL':
    eval_ds = CreateDataset(folder=EVAL_DS_PATH, transform=transform)
    classes = eval_ds.classes

# %%
# Define Model

train_loss_list = []
test_loss_list = []
acc_top1_list = []
acc_top5_list = []

model = model_utils.create_model(
    model_name=MODEL_NAME, img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=len(classes))

if MODE == 'CONTINUE' or MODE == 'EVAL':
    model_path_files = os.listdir(MODEL_PATH)
    for f in model_path_files:
        if f[:2] == 'ep':
            model.load_state_dict(torch.load(MODEL_PATH + '/' + f))
            CURR_EPOCH = int(f.split('.')[0][2:])
        elif f == 'loss.txt':
            loss_file = json.load(open(MODEL_PATH + '/' + f, 'r'))
            train_loss_list = loss_file['Train Loss']
            test_loss_list = loss_file['Test Loss']
        elif f == 'acc.txt':
            acc_file = json.load(open(MODEL_PATH + '/' + f, 'r'))
            acc_top1_list = acc_file['Accuracy Top1']
            acc_top5_list = acc_file['Accuracy Top5']
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

if CUDA:
    model.to(GPU_ID)

# %%


# %%
if MODE == 'TRAIN' or MODE == 'CONTINUE':

    print("Number of train samples: ", len(train_ds))
    print("Number of test samples: ", len(test_ds))

    train_loader = data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_eval = 0
    os.makedirs('checkpoints/' + MODEL_NAME, exist_ok=True)
    os.makedirs('checkpoints/{}/im{}_p{}_lr{}'.format(
        args.model_name, args.img_size, args.patch_size, args.learning_rate), exist_ok=True)

    # Train the model
    for epoch in range(EPOCHS):
        if epoch == 0 and MODE == 'CONTINUE':
            epoch = CURR_EPOCH
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
                if MODEL_NAME.split('_')[0] == 'Dino':
                    model.update_moving_average()
                test_output_top1 = test_output.argmax(1)
                test_output_top5_val, test_output_top5 = test_output.topk(
                    5, dim=1, largest=True, sorted=True)

                # Calculate Accuracy

                accuracy_top1 = (test_output_top1 ==
                                 test_y).sum().item() / BATCH_SIZE
                accuracy_top5 = 0
                for i in range(BATCH_SIZE):
                    test_output_top5_list = test_output_top5[i].tolist()
                    test_y_item = test_y[i].item()
                    if (test_y_item in test_output_top5_list):
                        accuracy_top5 += 1
                accuracy_top5 = accuracy_top5 / BATCH_SIZE
                acc_top1_list.append(round(accuracy_top1, 2))
                acc_top5_list.append(round(accuracy_top5, 2))
                print('Epoch: ', epoch, '| train loss: %.4f' %
                      loss, '| top1 accuracy: %.2f' % accuracy_top1, '| top5 accuracy: %.2f' % accuracy_top5, end="\r", flush=True)
                if accuracy_top1 > best_eval:
                    best_eval = accuracy_top1
                    torch.save(model.state_dict(), "checkpoints/{}/im{}_p{}_lr{}/best_val.pth".format(
                        args.model_name, args.img_size, args.patch_size, args.learning_rate))

        loss_file = open("checkpoints/{}/im{}_p{}_lr{}/loss.txt".format(
            args.model_name, args.img_size, args.patch_size, args.learning_rate), 'w')
        acc_file = open("checkpoints/{}/im{}_p{}_lr{}/acc.txt".format(args.model_name,
                        args.img_size, args.patch_size, args.learning_rate), 'w')
        loss_file.write(json.dumps({
            'Train Loss': train_loss_list,
            'Test Loss': test_loss_list,
        }))
        acc_file.write(json.dumps({
            'Accuracy Top1': acc_top1_list,
            'Accuracy Top5': acc_top5_list,
        }))

        if epoch != 0:
            os.remove("checkpoints/{}/im{}_p{}_lr{}/ep{}.pth".format(args.model_name,
                                                                     args.img_size, args.patch_size, args.learning_rate, epoch))
        torch.save(model.state_dict(), "checkpoints/{}/im{}_p{}_lr{}/ep{}.pth".format(
            args.model_name, args.img_size, args.patch_size, args.learning_rate, epoch + 1))
        print('Epoch: ', epoch, '| train loss: %.4f' %
              loss, '| top1 accuracy: %.2f' % accuracy_top1, '| top5 accuracy: %.2f' % accuracy_top5)

elif MODE == 'EVAL':
    eval_loader = data.DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=True)

    acc_top1_list_val = []
    acc_top5_list_val = []

    for step, (x, y) in enumerate(eval_loader):
        # Change input array into list with each batch being one element
        try:
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
            test_output_top1 = output.argmax(1)
            test_output_top5_val, test_output_top5 = output.topk(
                5, dim=1, largest=True, sorted=True)

            # Calculate Accuracy

            accuracy_top1 = (test_output_top1 ==
                            b_y).sum().item() / BATCH_SIZE
            accuracy_top5 = 0
            for i in range(BATCH_SIZE):
                test_output_top5_list = test_output_top5[i].tolist()
                test_y_item = b_y[i].item()
                if (test_y_item in test_output_top5_list):
                    accuracy_top5 += 1
            accuracy_top5 = accuracy_top5 / BATCH_SIZE
            acc_top1_list_val.append(accuracy_top1)
            acc_top5_list_val.append(accuracy_top5)

            print('top1 accuracy: %.2f' % accuracy_top1, '| top5 accuracy: %.2f' % accuracy_top5, end="\r", flush=True)
        except:
            pass

    print('Total top1 accuracy: %.2f' % (sum(acc_top1_list_val) / len(acc_top1_list_val)), '| Total top5 accuracy: %.2f' % (sum(acc_top5_list_val) / len(acc_top5_list_val)))

    fig, ax = plt.subplots()
    ax.plot(acc_top1_list)
    ax.set(xlabel='epoch', ylabel='Top1 Acc',
        title='{} Top1 Accuracy by epochs'.format(MODEL_PATH))
    fig.savefig(MODEL_PATH + '/acc_top1.png')
    fig, ax = plt.subplots()
    ax.plot(acc_top5_list)
    ax.set(xlabel='epoch', ylabel='Top5 Acc',
        title='{} Top5 Accuracy by epochs'.format(MODEL_NAME))
    fig.savefig(MODEL_PATH + '/acc_top5.png')
    fig, ax = plt.subplots()
    ax.plot(train_loss_list)
    ax.set(xlabel='epoch', ylabel='Training Loss',
        title='{} Training Loss by epochs'.format(MODEL_NAME))
    fig.savefig(MODEL_PATH + '/train_loss.png')
    fig, ax = plt.subplots()
    ax.plot(test_loss_list)
    ax.set(xlabel='epoch', ylabel='Testing Loss',
        title='{} Testing Loss by epochs'.format(MODEL_NAME))
    fig.savefig(MODEL_PATH + '/test_loss.png')