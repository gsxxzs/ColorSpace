import os
import cv2
import numpy as np
from torch import Tensor,FloatTensor
from tqdm import tqdm
import torch


# def hsv2rgb(img):
#     RGB = torch.zeros(img.shape)
#     h, s, v = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#     H = img.shape[0]
#     W = img.shape[1]
#     h1 = h // 60
#     f = h / 60 - h1
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
#     # r, g, b = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
#     r, g, b = FloatTensor(H, W), FloatTensor(H, W), FloatTensor(H, W)
#     for i in range(0, H):
#         for j in range(0, W):
#             if h1[i, j] == 0:
#                 r[i, j], g[i, j], b[i, j] = v[i, j], t[i, j], p[i, j]
#             elif h1[i, j] == 1:
#                 r[i, j], g[i, j], b[i, j] = q[i, j], v[i, j], p[i, j]
#             elif h1[i, j] == 2:
#                 r[i, j], g[i, j], b[i, j] = p[i, j], v[i, j], t[i, j]
#             elif h1[i, j] == 3:
#                 r[i, j], g[i, j], b[i, j] = p[i, j], q[i, j], v[i, j]
#             elif h1[i, j] == 4:
#                 r[i, j], g[i, j], b[i, j] = t[i, j], p[i, j], v[i, j]
#             else:
#                 r[i, j], g[i, j], b[i, j] = v[i, j], p[i, j], q[i, j]
#             # r[i, j], g[i, j], b[i, j] = int(r[i, j] * 255), int(g[i, j] * 255), int(b[i, j] * 255)
#         RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2] = r, g, b
#     return RGB
#
#
# def rgb2hsv(img):
#
#     HSV = torch.zeros(img.shape)
#     h = img.shape[0]
#     w = img.shape[1]
#     H = FloatTensor(h, w)
#     S = FloatTensor(h, w)
#     V = FloatTensor(h, w)
#     r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#     for i in range(0, h):
#         for j in range(0, w):
#             mx = max((b[i, j], g[i, j], r[i, j]))
#             mn = min((b[i, j], g[i, j], r[i, j]))
#             dt=mx-mn
#             if mx == mn:
#                 H[i, j] = 0
#             elif mx == r[i, j]:
#                 if g[i, j] >= b[i, j]:
#                     H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
#                 else:
#                     H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt) + 360
#             elif mx == g[i, j]:
#                 H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
#             elif mx == b[i, j]:
#                 H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt+ 240
#             #S
#             if mx == 0:
#                 S[i, j] = 0
#             else:
#                 S[i, j] = dt/mx
#             #V
#             V[i, j] =mx
#     HSV[:, :, 0], HSV[:, :, 1], HSV[:, :, 2] = H, S, V
#     return HSV

def get_hsv(im):
    eps = 1e-7
    img = im * 0.5 + 0.5
    hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]
    return hue, saturation, value


def get_rgb_from_hsv(hsv):
    C = hsv[2] * hsv[1]
    X = C * (1 - abs((hsv[0] * 6) % 2 - 1))
    m = hsv[2] - C

    if hsv[0] < 1 / 6:
        R_hat, G_hat, B_hat = C, X, 0
    elif hsv[0] < 2 / 6:
        R_hat, G_hat, B_hat = X, C, 0
    elif hsv[0] < 3 / 6:
        R_hat, G_hat, B_hat = 0, C, X
    elif hsv[0] < 4 / 6:
        R_hat, G_hat, B_hat = 0, X, C
    elif hsv[0] < 5 / 6:
        R_hat, G_hat, B_hat = X, 0, C
    elif hsv[0] <= 6 / 6:
        R_hat, G_hat, B_hat = C, 0, X

    R, G, B = (R_hat + m), (G_hat + m), (B_hat + m)

    return R, G, B


def RGB2HSV():
    path = "../LIIE/Image/LOL/low/"
    save_path = "LOL/low_HSV/"
    list = os.listdir(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fileName in tqdm(list):
        dir = path + fileName
        RGB = cv2.imread(dir)
        RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
        RGB = RGB / 255.0
        RGB = RGB.astype(np.float32)
        HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
        np.save(save_path + "HSV_" + fileName.split(".")[0], HSV)

def Data():
    path1 = "../LIIE/Image/LOL/low/"
    path2 = "../LIIE/Image/LOL/high/"
    save_path = "LOL/low_HSV/"
    list1 = os.listdir(path1)
    # list2 = os.listdir(path2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fileName in tqdm(list1):
        dir1 = path1 + fileName
        dir2 = path2 + fileName
        RGB = cv2.imread(dir1)
        RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
        RGB = RGB / 255.0
        RGB = RGB.astype(np.float32)
        HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)

        RGB_h = cv2.imread(dir2)
        RGB_h = cv2.cvtColor(RGB_h, cv2.COLOR_BGR2RGB)
        RGB_h = RGB_h / 255.0
        RGB_h = RGB_h.astype(np.float32)
        HSV_h = cv2.cvtColor(RGB_h, cv2.COLOR_RGB2HSV)
        HSV[:, :, 2] = HSV_h[:, :, 2]

        np.save(save_path + "HSV_" + fileName.split(".")[0], HSV)

def getHLS(dir):
    RGB = cv2.imread(dir)
    cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
    RGB = RGB / 255.0
    RGB = RGB.astype(np.float32)
    HLS = cv2.cvtColor(RGB, cv2.COLOR_RGB2HLS)
    return HLS[:, :, 0],  HLS[:, :, 2],  HLS[:, :, 1]

# RGB2HSV()

# RGB = cv2.imread("778.png")
# RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
# RGB = RGB / 255.0
#
# RGB = RGB.astype(np.float32)
#
# HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
# np.save("778_h_HSV.npy", HSV)
# HSV = np.load("778_h_HSV.npy")
# HSV = np.load("LOL/eval15/high_HSV/HSV_778.npy")
# # RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
# # cv2.imshow("1231", RGB)
#
# V = HSV[:, :, 2]
# S = HSV[:, :, 1]
# H = HSV[:, :, 0]
# RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
# # cv2.imshow("alter", RGB)
# # cv2.imwrite("sdas.jpg", RGB * 255)
#
# # H_1, S_1, V_1 = rgb2hsv(RGB)
# #
# # cv2.imshow("H", H)
# # cv2.imshow("S", S)
# # cv2.imshow("V", V)
#
# # cv2.imshow("H1", H_1)
# # cv2.imshow("S1", S_1)
# # cv2.imshow("V1", V_1)
#
#
#
# RGB_h = cv2.imread("778.png")
# RGB_h = cv2.cvtColor(RGB_h, cv2.COLOR_BGR2RGB)
# RGB_h = RGB_h / 255
# RGB_h = RGB_h.astype(np.float32)
# HSV_h = cv2.cvtColor(RGB_h, cv2.COLOR_RGB2HSV)
# V_h = HSV_h[:, :, 2]
#
# cv2.imshow("V", V)
# V_alter =  V ** (8.0)
#
# HSV[:, :, 2] = V_alter
# # HSV[:, :, 2] = HSV_h[:, :, 2]
#
# RGB_alter = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
# cv2.imshow("alter RGB", RGB_alter)
# cv2.imshow("RGB", RGB_h)
#
# # cv2.imshow("V_alter", V_alter)
# # cv2.imwrite("v.jpg", V_alter * 255)
# cv2.waitKey(0)

from ColorSpaceTransform import ColorSpaceTransform

if __name__ == '__main__':
    # HSV = np.load("LOL/low_HSV/HSV_2.npy")
    # RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    # cv2.imshow("RGB", RGB)
    # cv2.waitKey(0)
    # H, S, L = getHLS("778.png")
    # H1, S1, L1 = getHLS("778_h.png")
    # d_h = (H - H1)/360
    # d_S = S - S1
    # d = (d_h + d_S) / 2
    # L = L - d
    # L = L ** (1 / 4.0)
    # # RGB = numpy.empty([400, 600, 3], dtype=float)
    # # # RGB = np.ndarray([H, L, S])
    # # # RGB = RGB.transpose(1, 2, 0)
    # # RGB[:, :, 0] = H
    # # RGB[:, :, 1] = L
    # # RGB[:, :, 2] = S
    # # RGB = RGB.astype(np.float32)
    # # RGB = cv2.cvtColor(RGB, cv2.COLOR_HLS2RGB)
    # cv2.imshow("RGB", L)
    # cv2.waitKey(0)

    img = cv2.imread("42.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.astype(np.float32)

    tensor = torch.from_numpy(img)

    Transform = ColorSpaceTransform()

    HSI = Transform.RGB2HSI(tensor)

    RGB = Transform.HSI2RGB(HSI)
    # RGB = cv2.cvtColor(HSI.numpy(), cv2.COLOR_HLS2BGR)
    # im = RGB
    im = RGB.numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow("1", im)
    cv2.waitKey(0)







