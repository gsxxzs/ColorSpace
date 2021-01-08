import cv2
import torch


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







