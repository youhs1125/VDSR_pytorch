from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
from utils import adjust_learning_rate

# define train_loop

def train_loop(model, dataloader, loss_fn, optimizer, writer, epoch):
    size = len(dataloader.dataset)
    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = adjust_learning_rate(optimizer, epoch-1)
    print("learning_rate = ",lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    cur_loss = 0.0
    for iter, (X, y) in enumerate(tqdm(dataloader, position=0, leave=True, desc="train")):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), 0.4)

        optimizer.step()
        cur_loss += loss.item()

        if iter % 10 == 0:
            train_loss = cur_loss / 10
            num_pred = pred.cpu().detach().numpy()
            num_y = y.cpu().numpy()
            train_PSNR = cv2.PSNR(num_pred*255, num_y*255)

            print(f"{iter}-train_loss: {train_loss} PSNR: {train_PSNR}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("PSNR/train", train_PSNR, epoch)

            cur_loss = 0


# define validation_loop

def val_loop(model, dataloader, writer, epoch):
    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_PSNR = 0.0
    for iter, (X, y) in enumerate(tqdm(dataloader, position=0, leave=True, desc="validation")):
        pred = model(X.to(device))
        num_pred = pred.cpu().numpy()
        num_y = y.numpy()
        val_PSNR += cv2.PSNR(num_pred*255, num_y*255)

    val_PSNR /= len(dataloader)
    print(f"RSNR: {val_PSNR}")

    writer.add_scalar("PSNR/val", val_PSNR, epoch)

# define test_loop

def test_loop(model, ds):
    pred = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        for img in ds:
            temp = img.reshape(img.shape[-1],img.shape[0],img.shape[1])
            input = torch.FloatTensor(temp)
            pred.append(model(input.to(device)).cpu().numpy())

    return pred

def getBicubic(ds):
    bicubic = []

    for img in ds:
        bicubic.append(cv2.resize(img,dsize=(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC))


    return bicubic
