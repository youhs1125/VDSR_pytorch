import torch
import dataset
import loop
import numpy as np
import utils

def doTest():


    # get dataloader
    test_dataloader = dataset.getDataset(isTrain=False)


    # load modelData
    vdsr = torch.load("models/VDSR_model.pt")
    vdsr.cpu()
    srcnn = torch.load("models/SRCNN_model.pt")
    srcnn.cpu()

    preds = loop.test_loop(vdsr,test_dataloader)
    srcnns = loop.test_loop(srcnn,test_dataloader)

    origins, bicubic = loop.getOrgBicubic(test_dataloader)

    utils.comparePSNR(origins,bicubic,preds,srcnns)


if __name__ == "__main__":
    doTest()