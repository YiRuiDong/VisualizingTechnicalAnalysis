import pandas as pd
import timm
import torch
import PIL.ImageEnhance as ImageEnhance
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, \
    EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from Data_generate import Data_generator
from DrawLib import DrawOHLCTechnical
from PIL import Image
import cv2
import re
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2, in_chans=3)
    model_name = 'swin5BOLL+RSI'
    model.load_state_dict(torch.load(f'./checkpoint/{model_name}_params_complete.pt'))
    window = int(re.findall('[0-9]+', model_name)[0])
    model.eval()
    model = model.cuda()

    d_g = Data_generator('data_us/tech/', market='us')
    d_g.read_data(699, flag='test')
    stock_no = d_g.df['TICKER'][0]

    print(stock_no)

    idx = 115
    data_zip = d_g.sep_OHLC_Techs_Label(idx, window, 5, require_actual_ret=True, require_true_date=True)

    print(d_g.df.iloc[idx:idx + window])
    ret = data_zip['label_true']
    date = data_zip['date']

    start = pd.to_datetime(date.iloc[0, 0]).strftime("%b_%d_%Y")
    end = pd.to_datetime(date.iloc[-1, 0]).strftime("%b_%d_%Y")

    print(end)

    DrawOHLCTechnical(224, data_zip['OHLC'], './temp/', f'{stock_no}_{ret}.bmp', BOLL=data_zip['BOLL'],
                      RSI=data_zip['RSI'])

    with Image.open(f'./temp/{stock_no}_{ret}.bmp', 'r') as img:
        img_t = transforms.Resize((224, 224))(img)
        img.close()
        img_tensor = transforms.ToTensor()(img_t).to('cuda')
    x = torch.reshape(img_tensor, (1, 3, 224, 224))
    y = model(x).detach().cpu().numpy()
    print('predict cls:', np.argmax(y), 'score:', np.max(y))
    print('ground truth:', ret)
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layers[-1].blocks[-1].norm2],
                          reshape_transform=reshape_transform)
    grad_cam = cam(input_tensor=x, aug_smooth=True, eigen_smooth=True)

    grad_cam = grad_cam / np.max(grad_cam)

    cv2.imshow('img,jpg', grad_cam[0])
    cv2.waitKey(0)

    kernel = np.ones((2, 2), np.float32)

    # grad_cam[0] = cv2.GaussianBlur(grad_cam[0], ((224//window)+1 if (224//window)%2==0 else (224//window), 11), 0)
    # cv2.imshow('img,jpg', grad_cam[0])
    # cv2.waitKey(0)

    # grad_cam[0] = cv2.erode(grad_cam[0],kernel,iterations=5)
    # cv2.imshow('img,jpg', grad_cam[0])
    # cv2.waitKey(0)

    # _,grad_cam[0] = cv2.threshold(grad_cam[0],0.15,1,cv2.THRESH_TOZERO)
    # cv2.imshow('img,jpg', grad_cam[0])
    # cv2.waitKey(0)

    dst = np.zeros_like(grad_cam[0])
    grad_cam[0] = cv2.normalize(grad_cam[0], dst, 0, 1, cv2.NORM_MINMAX)

    # cv2.imshow('img,jpg', grad_cam[0])
    # cv2.waitKey(0)

    rgb_img = np.float32(cv2.resize(cv2.imread(f'./temp/{stock_no}_{ret}.bmp'), dsize=(224, 224))) / 255
    # grad_cam[0] = 1 - grad_cam[0]
    visualization = show_cam_on_image(np.zeros_like(rgb_img), grad_cam[0], use_rgb=False, colormap=cv2.COLORMAP_HOT)
    # visualization = cv2.resize(visualization, dsize=(3 * window, 224))
    cv2.imwrite(f'./temp/{stock_no}_{ret}_cam.jpg', visualization)
