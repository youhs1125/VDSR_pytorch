import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculatePSNR(sr, hr, scale = 2):
    diff = (sr - hr)/256
    shave = scale
    diff[:,:,0] = diff[:,:,0]*65.738/256
    diff[:,:,1] = diff[:,:,1] * 129.057/256
    diff[:,:,2] = diff[:,:,2] * 25.064/256

    diff = np.sum(diff, axis=2)

    valid = diff[shave:-shave, shave:-shave]
    mse = np.mean(valid**2)

    return -10 * np.log10(mse)

def adjust_learning_rate(optimizer,epoch):
    lr = 0.1 * (0.1**(epoch//10))
    return lr

def backChannel(img):
    # print(img.shape)
    np_transpose = np.ascontiguousarray(img.transpose((1, 2, 0)))
    return np_transpose
def comparePSNR(origins, bicubic, preds1, preds2 = None, preds3 = None):
    # compare predicts with bicubic
    mPSNR = 0
    bPSNR = 0
    sPSNR = 0
    for i in range(len(preds1)):
        pred_num = backChannel(preds1[i])
        pred_num = pred_num * 255

        pred_num2 = backChannel(preds2[i])
        srcnn = pred_num2*255
        testi = backChannel(origins[i]*255)
        bicubici = bicubic[i]*255

        # print(pred_num.shape, testi.shape, bicubici.shape, srcnn.shape)
        predPSNR = calculatePSNR(pred_num, testi)
        bicubicPSNR = calculatePSNR(bicubici, testi)
        srcnnPSNR = calculatePSNR(srcnn,testi)
        # print("MODEL --- PSNR: ", predPSNR)
        # print("BICUBIC - PSNR: ", bicubicPSNR)

        mPSNR += predPSNR
        bPSNR += bicubicPSNR
        sPSNR += srcnnPSNR

        if i == 0:
            if preds2 == None:
                print(f"PSNR(of images below) VDSR : {predPSNR} Bicubic Interpolation: {bicubicPSNR}")
                fig, axes = plt.subplots(1, 3, figsize=(20, 36))
            else:
                print(f"PSNR(of images below) VDSR : {predPSNR} Bicubic Interpolation: {bicubicPSNR} SRCNN: {srcnnPSNR}")
                fig, axes = plt.subplots(1,4, figsize = (20,48))

            pred_num /= 255.0
            bicubici /= 255.0
            testi /= 255.0
            srcnn /= 255.0

            pred_num = np.clip(pred_num, 0.0, 1.0)
            bicubici = np.clip(bicubici, 0.0, 1.0)
            srcnn = np.clip(srcnn,0.0,1.0)


            axes[0].imshow(testi)
            axes[0].set_title("original")
            axes[1].imshow(bicubici)
            axes[1].set_title("Bicubic Interpolation")
            axes[2].imshow(pred_num)
            axes[2].set_title("VDSR")

            if preds2 != None:
                axes[3].imshow(srcnn)
                axes[3].set_title("SRCNN")
            plt.show()
    if preds2 == None:
        print(f"PSNR(avg) VDSR: {mPSNR / len(preds1)},  Bicubic Interpolation: {bPSNR / len(preds1)}")
    else:
        print(f"PSNR(avg) VDSR: {mPSNR / len(preds1)},  Bicubic Interpolation: {bPSNR / len(preds1)},   SRCNN: {sPSNR / len(preds1)}")