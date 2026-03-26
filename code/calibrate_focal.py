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
    #img_all = img.copy()
    #cv2.drawContours(img_all, cnts, -1, (0, 255, 0), 2)
    #cv2.imshow("All Contours", img_all)
    
    print(f"轮廓数量：{len(cnts)}")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # cv2.drawContours(img_all, cnts[0], -1, (0, 255, 0), 2)
    # cv2.imshow("All Contours", img_all)

    if len(cnts) > 0:
        # 假设最大的轮廓就是 A4 纸
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        if len(approx) == 4:
            print(f"近似多边形顶点数：{len(approx)}")

            # 4. 增加顶点精度 将 approx 转换为 float32 格式
            corners = np.float32(approx).reshape(-1, 2)

            # 设置亚像素搜索停止准则
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # 在原灰度图中精细化角点位置
            # gray 是预处理后的灰度图，(5,5) 是搜索窗口大小
            refined_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            # 此时 refined_corners 里的坐标是带小数的，比如 (120.45, 345.78)用这些点去算距离 D，精度会有一个质的飞跃。

            # 5. 重新排序角点并计算长和宽
            rect = np.zeros((4, 2), dtype="float32")   
            s = refined_corners.sum(axis=1)
            rect[0] = refined_corners[np.argmin(s)]       # 左上 TL
            rect[2] = refined_corners[np.argmax(s)]       # 右下 BR
            
            # 计算每个点的坐标之差 (y - x)
            diff = np.diff(refined_corners, axis=1)
            rect[1] = refined_corners[np.argmin(diff)]    # 右上 TR
            rect[3] = refined_corners[np.argmax(diff)]    # 左下 BL

            (tl, tr, br, bl) = rect
    
            # 计算顶边和底边的像素宽度
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            
            # 计算左边和右边的像素高度
            height_left = np.linalg.norm(tl - bl)
            height_right = np.linalg.norm(tr - br)
            
            # 在透视投影中，对边不一定相等
            # 计算平均值，或者根据竞赛需求取最大值
            w_pixel = (width_top + width_bottom) / 2
            h_pixel = (height_left + height_right) / 2
            print(f"精确化后的像素宽度 w_pixel: {w_pixel:.2f},精确化后的像素高度 h_pixel: {h_pixel:.2f}")

            # 6. 根据公式计算像素焦距

            # f_pixel = (h_pixel * D) / H
            f_pixel = (h_pixel * real_distance) / real_height

            # 7. 绘制识别结果用于预览确认
            img_all = img.copy()

            cv2.drawContours(img_all, approx, -1, (0, 255, 0), 2)
            cv2.imshow("approx", img_all)#显示拟合得到的端点          
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            
            #  格式化文字 (保留两位小数)
            text_w = f"W: {w_pixel:.2f}px"
            text_h = f"H: {h_pixel:.2f}px"

            # 计算标点位置 (取边中点再偏移一点，避免压线)
            # 宽度的标点：顶边 (tl 和 tr) 的中心
            pos_w = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2) - 10)

            # 高度的标点：左边 (tl 和 bl) 的中心
            pos_h = (int((tl[0] + bl[0]) / 2) - 80, int((tl[1] + bl[1]) / 2))

            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_rectangle", img)#显示拟合的矩形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            

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