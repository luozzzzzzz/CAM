import cv2
import numpy as np

def measure_target(image_path, f_pixel, real_side_limit=(10, 16)):
    """
    测量图像中正方形的距离 D 和边长 x
    :param image_path: 图片路径
    :param f_pixel: 相机的像素焦距 (通过预先测量获得)
    :param real_side_limit: 题目要求的边长范围 (cm)
    :return: D (距离), x (边长)
    """
    # 1. 读取并预处理图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"轮廓个数：{len(contours)}")
    for cnt in contours:
        # 近似多边形，减少顶点数
        peri = cv2.arcLength(cnt, True)
        print(peri)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 题目要求是基本几何图形，此处识别正方形（4个顶点）
        if len(approx) == 4:
            # 计算像素边长 (取四边平均或外接矩形)
            x, y, w, h = cv2.boundingRect(approx)
            pixel_width = (w + h) / 2.0
            
            # --- 核心物理计算 ---
            # 根据单目成像原理: 
            # (实际边长 x / 距离 D) = (像素边长 pixel_width / 像素焦距 f_pixel)
            
            # 注意：在题目基本要求中，D 是基准线到物面的距离。
            # 我们假设摄像头光心 O 就在基准线上，则 D 为物距。
            
            # 由于 D 和 x 都是未知的，我们需要利用 A4 纸边框作为参照物
            # 题目说明：所有目标物均为 A4 纸（21cm x 29.7cm），带 2cm 黑色边框。
            # 这是一个关键突破口！
            
            # 假设我们先识别出 A4 纸的外边框（像素高度 H_paper）
            # 已知 A4 纸高度 H_real = 29.7 cm
            # 则距离 D = (H_real * f_pixel) / H_paper
            # 进而边长 x = (pixel_width * D) / f_pixel
            
            # 这里简化处理：假设 D 已知或通过 A4 边框求得
            # 以下示例逻辑为：已知 D 求 x，或已知 x 求 D。
            # 实际比赛中，建议先通过 A4 纸外框的固定尺寸算出 D。
            
            H_real_a4 = 29.7  # A4 纸高度 cm
            # 假设找到最外层 A4 轮廓的像素高度为 h_a4
            # D = (H_real_a4 * f_pixel) / h_a4
            # x = (pixel_width * D) / f_pixel
            
            #return D, x

    return None, None


image_1 = r"picture\test_canny.jpg"
D,x =measure_target(image_1, f_pixel = 301.77, real_side_limit=(10, 16))
print(f"距离 D: {D} cm, 边长 x: {x} cm")