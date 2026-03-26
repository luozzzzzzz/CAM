import cv2
import numpy as np

def get_focal_length(image_path, real_distance, real_height=28.3):
    """
    通过已知距离的照片计算像素焦距
    :param image_path: 用于校准的照片路径
    :param real_distance: 拍摄时目标物到基准线的物理距离 (D)，单位 cm
    :param real_height: 参照物(A4纸)的实际物理高度，单位 cm
    :return: f_pixel (像素焦距)
    """
    # 1. 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法加载图片")
        return None
    print(f"图片大小：{img.shape[0]} x {img.shape[1]}")

    # 2. 预处理：灰度化 -> 高斯滤波 -> 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # 3. 寻找轮廓并排序（取面积最大的，即 A4 纸外框）
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 画出所有轮廓（绿色）
    img_all = img.copy()
    # cv2.drawContours(img_all, cnts, -1, (0, 255, 0), 2)
    # cv2.imshow("All Contours", img_all)
    
    print(f"轮廓数量：{len(cnts)}")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    cv2.drawContours(img_all, cnts[0], -1, (0, 255, 0), 2)
    cv2.imshow("All Contours", img_all)

    if len(cnts) > 0:
        # 假设最大的轮廓就是 A4 纸
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        print(f"近似多边形顶点数：{len(approx)}")

        img_all = img.copy()
        cv2.drawContours(img_all, approx, -1, (0, 255, 0), 2)
        cv2.imshow("approx", img_all)
        # 获取外接矩形的高度（像素）
        x, y, w_pixel, h_pixel = cv2.boundingRect(approx)
        
        # 绘制识别结果用于预览确认
        cv2.rectangle(img, (x, y), (x + w_pixel, y + h_pixel), (0, 255, 0), 2)
        cv2.putText(img, f"H_pix: {h_pixel}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration Preview", img)
        cv2.imwrite(r"picture/calibration.jpg", img)
        cv2.waitKey(0)

        # 4. 根据公式计算像素焦距
        # f_pixel = (h_pixel * D) / H
        f_pixel = (h_pixel * real_distance) / real_height
        return f_pixel
    else:
        print("未识别到目标轮廓")
        return None

if __name__ == "__main__":
    # --- 用户设定参数 ---
    # 假设你把 A4 纸放在 D = 150cm 处拍了一张照
    test_distance = 20.0 
    image_path = r"picture/3_26_1.png" # 替换为你的实拍图路径
    
    f_val = get_focal_length(image_path, test_distance)
    
    if f_val:
        print("-" * 30)
        print(f"校准成功！")
        print(f"计算得出的像素焦距 f_pixel: {f_val:.2f}")
        print(f"请将此数值保存到你的主程序配置文件中。")
        print("-" * 30)