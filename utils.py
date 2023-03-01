import os
import csv
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from datetime import datetime
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
#!pip install --upgrade --force-reinstall --no-deps kaggle


from dataloaders import *

def train(args, model, device, train_samples, optimizer, criterion, epoch):
    model.train()
    scaler = GradScaler()
    for i in range(len(train_samples)):
        X, Y = train_samples[i]
        train_items = LibriItems(X, Y, context=args['context'])
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=args['batch_size'], shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, dev_samples):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(dev_samples)):
            X, Y = dev_samples[i]

            test_items = LibriItems(X, Y, context=args['context'])
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)

            for data, true_y in test_loader:
                data = data.float().to(device)
                true_y = true_y.long().to(device)                
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)

                pred_y_list.extend(pred_y.tolist())
                true_y_list.extend(true_y.tolist())

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    return train_accuracy




def evaluate(args, model, device, eval_samples):
  model.eval()
  eval_y_list = []
  with torch.no_grad():
        for i in range(len(eval_samples)):
            X = eval_samples[i]

            eval_items = LibriItemsEval(X, context=args['context'])
            eval_loader = torch.utils.data.DataLoader(eval_items, batch_size=args['batch_size'], shuffle=False)

            for data in eval_loader:
                data = data.float().to(device)              
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)

                eval_y_list.extend(pred_y.tolist())

  return eval_y_list


def eval_and_submit(args, model):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  eval_samples = LibriSamplesEval(data_path = args['LIBRI_PATH'], shuffle=False, partition="test-clean")
  test_predictions = evaluate(args, model, device, eval_samples)
  to_csv_list=[]
  for i, val in enumerate(test_predictions):
    to_csv_list.append([i, val])
  to_csv = pd.DataFrame(to_csv_list, columns=['Id, Label'])
  to_csv.to_csv("Submission.csv", index=False)