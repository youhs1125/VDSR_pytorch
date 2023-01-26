from tqdm import tqdm
import torch
import torch.nn as nn
import cv2

# define train_loop

def train_loop(model, dataloader, loss_fn, optimizer, writer, epoch):
    size = len(dataloader.dataset)
    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            train_loss = cur_loss / 100
            num_pred = pred.cpu().detach().numpy()
            num_y = y.cpu().numpy()
            train_PSNR = cv2.PSNR(num_pred, num_y)

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
        val_PSNR += cv2.PSNR(num_pred, num_y)

    val_PSNR /= len(dataloader)
    print(f"RSNR: {val_PSNR}")

    writer.add_scalar("PSNR/val", val_PSNR, epoch)

# define test_loop

def test_loop(model, dataloader):
    pred = []

    model.eval()
    with torch.no_grad():
        for (X,y) in (tqdm(dataloader,position=0, leave=True, desc="test")):
            pred.append(model(X).numpy())

    return pred

def getOrgBicubic(dataloader):
    origin = []
    bicubic = []

    with torch.no_grad():
        for (X,y) in (tqdm(dataloader,position=0, leave=True, desc="test")):
            origin.append(y.numpy().squeeze())
            temp = X.numpy()
            temp = temp.reshape(temp.shape[2],temp.shape[2],3)
            temp = cv2.resize(temp,dsize=(temp.shape[1],temp.shape[1]),interpolation=cv2.INTER_CUBIC)
            bicubic.append(temp)

    return origin,bicubic
