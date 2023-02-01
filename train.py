from VDSR_model import VDSR
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataset
import loop
import matplotlib.pyplot as plt
import numpy as np

def doTrain():
    # get dataloader
    train_dataloader, val_dataloader = dataset.getDataset()

    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # define model
    model = VDSR().to(device)

    # define loss function
    loss_fn = nn.MSELoss()

    # define optimizer
    learning_rate = 0.1
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # define tensorboard Logger
    writer = SummaryWriter()

    # define epoch
    max_epoch = 50

    for epoch in range(1, max_epoch + 1):
        # train
        print(f"EPOCH: {epoch} \n\n")
        model.train()
        loop.train_loop(model, train_dataloader, loss_fn, optimizer, writer, epoch)

        # valid
        model.eval()
        with torch.no_grad():
            loop.val_loop(model, val_dataloader, writer, epoch)

    writer.close()
    torch.save(model.state_dict(),"models/VDSR_model_state_dict.pt")
    torch.save(model,"models/VDSR_model.pt")

if __name__ == "__main__":
    doTrain()