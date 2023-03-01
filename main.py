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
from models import *
from utils import *


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args['model_name_path']
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if args['continue_from_model']:
      print("continuing Model")
      checkpoint = torch.load(model_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      model.to(device)
    else:
      model = Network().to(device)
    #model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, eps=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100", csvpath=None)#"train_filenames_subset_2048_v2.csv")
    dev_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="dev-clean")
    
    for epoch in range(1, args['epoch'] + 1):
        if epoch>1:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(device)
        train(args, model, device, train_samples, optimizer, criterion, epoch)
        test_acc = test(args, model, device, dev_samples)
        scheduler.step(test_acc)
        print('Dev accuracy ', test_acc)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
        
    eval_and_submit(args, model)


    

if __name__ == '__main__':
    args = {
        'batch_size': 2048,
        'context': 32,
        'log_interval': 200,
        'LIBRI_PATH': '/content/hw1p2_student_data',
        'model_name_path': '/content/hw1p2_student_data/models/model',
        'lr': 0.1,
        'epoch': 11,
        'continue_from_model':False
    }
    main(args)