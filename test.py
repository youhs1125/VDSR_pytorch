import torch
import dataset
import loop
import numpy as np
import utils

def doTest():


    # get dataloader
    origin, ds = dataset.getTestData()


    # load modelData
    vdsr = torch.load("models/VDSR_model.pt")
    vdsr.cpu()
    srcnn = torch.load("models/SRCNN_model.pt")
    srcnn.cpu()

    preds = loop.test_loop(vdsr,ds)
    srcnns = loop.test_loop(srcnn,ds)

    bicubic = loop.getBicubic(ds)

    utils.comparePSNR(origin,bicubic,preds,srcnns)


if __name__ == "__main__":
    doTest()