import cv2
import torch

from ColorSpaceTransform import ColorSpaceTransform

if __name__ == '__main__':

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







