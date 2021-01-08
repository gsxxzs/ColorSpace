import torch


class ColorSpaceTransform():
    def __init__(self):
        super(ColorSpaceTransform, self).__init__()
        self.eps = 1e-6
        self.pi = 3.141592654

    def RGB2HSI(self, RGB):
        R, G, B = RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2]
        min_RGB = torch.min(RGB, dim=2)[0]
        I = (R + G + B) / 3
        S = 1 - min_RGB * 3 / (R + G + B)

        F =torch.sign

        # 计算每个点与RGB三通道最小值的关系，然后剔除可能重复的点, 这里删除的顺序一定要是BRG，其他可能出现异常点
        B_min_position = F(min_RGB - B) + 1

        R_min_position = F(min_RGB - R) + 1
        R_min_position = R_min_position - (F(((R_min_position + B_min_position) - 1) - 0.5) * 0.5 + 0.5)

        G_min_position = F(min_RGB - G) + 1
        G_min_position = G_min_position - (F(((G_min_position + B_min_position) - 1) - 0.5) * 0.5 + 0.5)
        G_min_position = G_min_position - (F(((G_min_position + R_min_position) - 1) - 0.5) * 0.5 + 0.5)

        # 计算出所有坐标的位置，在一种模式下的位置，然后只保留RGB通道最小的点
        H_in_B = ((G - B) / (3 * (R + G - 2 * B) + self.eps)) * B_min_position
        H_in_R = ((B - R) / (3 * (G + B - 2 * R) + self.eps) + 1 / 3) * R_min_position
        H_in_G = ((R - G) / (3 * (B + R - 2 * G) + self.eps) + 2 / 3) * G_min_position

        # H的值域为（0,360）
        H = (H_in_R + H_in_G + H_in_B) * 360

        HSI = torch.zeros(RGB.shape)
        HSI[:, :, 0], HSI[:, :, 1], HSI[:, :, 2] = H, S, I
        return HSI

    def HSI2RGB(self, HSI):
        H, S, I = HSI[:, :, 0], HSI[:, :, 1], HSI[:, :, 2]
        F = torch.sign
        # 计算出三种模式下需要保留的点的位置
        H_small_120 = F(120 - H) * 0.5 + 0.5
        H_middle = F(240 - H) * 0.5 + 0.5 - H_small_120
        H_large_240 = 1 - H_small_120 - H_middle

        # 将H转化为角度
        H = H * self.pi / 180
        #计算各个模式下的位置并且保留不同角度下的点
        # H值域为(0,2/3 pi)时
        B_low = I * (1 - S) * H_small_120
        R_low = I * (1 + S * torch.cos(H) / (torch.cos(self.pi / 3 - H) + self.eps)) * H_small_120
        G_low = (3 * I - (B_low + R_low)) * H_small_120
        # H值域为(2/3 pi， 4/3 pi)时
        R_middle = I * (1 - S) * H_middle
        G_middle = I * (1 + S * torch.cos(H - 2 * self.pi / 3) / (torch.cos(self.pi - H) + self.eps)) * H_middle
        B_middle = (3 * I - (R_middle + G_middle)) * H_middle
        # H值域为(4/3 pi， 2 pi)时
        G_high = I * (1 - S) * H_large_240
        B_high = I * (1 + S * torch.cos(H - self.pi * 4 / 3) / (torch.cos(self.pi * 5 / 3 - H) + self.eps)) * H_large_240
        R_high = (3 * I - (G_high + B_high)) * H_large_240

        # 进行融合
        R = R_low + R_middle + R_high
        G = G_low + G_middle + G_high
        B = B_low + B_middle + B_high

        RGB = torch.zeros(HSI.shape)
        RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2] = R, G, B,
        return RGB

