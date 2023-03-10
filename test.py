import torch
import dataset
import loop
import numpy as np
import utils
from SRCNN_model import SRCNN

def doTest():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get dataloader
    origin, ds = dataset.getTestData()


    # load modelData
    vdsr = torch.load("models/VDSR_model.pt")
    vdsr.to(device)

    srcnn = SRCNN().to(device)
    srcnn.load_state_dict(torch.load("models/SRCNN_model_state_dict.pt"))

    preds = loop.test_loop(vdsr,ds)
    srcnns = loop.test_loop(srcnn,ds)

    bicubic = loop.getBicubic(ds)

    utils.comparePSNR(origin,bicubic,preds,srcnns)


if __name__ == "__main__":
    doTest()