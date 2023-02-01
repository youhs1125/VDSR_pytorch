import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer,epoch):
    lr = 0.1 * (0.1**(epoch//10))
    return lr

def changeColorChannel(input):
    output = input.reshape(input.shape[1], input.shape[2], input.shape[0])
    return output
def comparePSNR(origins, bicubic, preds1, preds2 = None, preds3 = None):
    # compare predicts with bicubic
    mPSNR = 0
    bPSNR = 0
    sPSNR = 0
    for i in range(len(preds1)):
        pred_num = changeColorChannel(preds1[i])
        pred_num = pred_num * 255

        pred_num2 = changeColorChannel(preds2[i])
        srcnn = pred_num2*255
        testi = origins[i]*255
        bicubici = bicubic[i]*255

        # print(pred_num.shape, testi.shape, bicubici.shape, srcnn.shape)
        predPSNR = cv2.PSNR(pred_num, testi)
        bicubicPSNR = cv2.PSNR(bicubici, testi)
        srcnnPSNR = cv2.PSNR(srcnn,testi)
        # print("MODEL --- PSNR: ", predPSNR)
        # print("BICUBIC - PSNR: ", bicubicPSNR)

        mPSNR += predPSNR
        bPSNR += bicubicPSNR
        sPSNR += srcnnPSNR

        if i == 0:
            if preds2 == None:
                print(f"PSNR(of images below) VDSR : {predPSNR} Bicubic Interpolation: {bicubicPSNR}")
                fig, axes = plt.subplots(1, 3, figsize=(20, 36))
                print(f"PSNR(avg) VDSR: {mPSNR / len(preds1)},  Bicubic Interpolation: {bPSNR / len(preds1)}")
            else:
                print(f"PSNR(of images below) VDSR : {predPSNR} Bicubic Interpolation: {bicubicPSNR} SRCNN: {srcnnPSNR}")
                fig, axes = plt.subplots(1,4, figsize = (20,48))
                print(
                    f"PSNR(avg) VDSR: {mPSNR / len(preds1)},  Bicubic Interpolation: {bPSNR / len(preds1)},   SRCNN: {sPSNR / len(preds1)}")

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