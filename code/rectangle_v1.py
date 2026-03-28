import cv2
import numpy as np
import tools 

def measure_target(image_path, f_pixel, h_outcontour = 28.3,w_outcontour=21.0,real_side_limit=(10, 16)):
    """
    测量图像中正方形的距离 D 和边长 x
    :param image_path: 图片路径
    :param f_pixel: 相机的像素焦距 (通过预先测量获得)
    :param real_side_limit: 题目要求的边长范围 (cm)
    :return: D (距离), x (边长)
    """
    cnts,img_gray = tools.get_conTours(image_path)

    if len(cnts) > 0:
        #一、根据外轮廓计算距离 D
        # 假设最大的轮廓就是 A4 纸
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)

        if len(approx) == 4:
            print(f"近似多边形顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            # 1. 计算像素长度和宽度（重新排序角点）
            rect = np.zeros((4, 2), dtype="float32")   

            sum = refined_corners.sum(axis=1)
            rect[0] = refined_corners[np.argmin(sum)]       # 左上 TL
            rect[2] = refined_corners[np.argmax(sum)]       # 右下 BR
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
            w_pixel_ex = (width_top + width_bottom) / 2
            h_pixel_ex = (height_left + height_right) / 2
            print(f"精确化后的外边框像素宽度 w_pixel_ex: {w_pixel_ex:.2f},精确化后的外边框像素高度 h_pixel_ex: {h_pixel_ex:.2f}")

            # 2. 根据公式计算
 
            distance_1 = h_outcontour * f_pixel / h_pixel_ex
            distance_2 = w_outcontour * f_pixel / w_pixel_ex
            distance = (distance_1 + distance_2) / 2
            print(f"距离 D: {distance:.2f} cm")

        # 二、根据内轮廓计算边长 x

        peri = cv2.arcLength(cnts[-1], True)
        approx = cv2.approxPolyDP(cnts[-1], 0.02 * peri, True)

        if len(approx) == 4:
            print(f"目标为正方形，顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            # 1. 计算像素长度和宽度（重新排序角点）
            rect = np.zeros((4, 2), dtype="float32")   

            sum = refined_corners.sum(axis=1)
            rect[0] = refined_corners[np.argmin(sum)]       # 左上 TL
            rect[2] = refined_corners[np.argmax(sum)]       # 右下 BR
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
            w_pixel_obj = (width_top + width_bottom) / 2
            h_pixel_obj = (height_left + height_right) / 2
            print(f"精确化后的目标像素宽度 w_pixel_obj: {w_pixel_obj:.2f},精确化后的目标像素高度 h_pixel_obj: {h_pixel_obj:.2f}")
            # 2. 计算实际边长
            x_1 = h_pixel_obj * h_outcontour / h_pixel_obj  # 这里需要根据实际情况调整公式
            x_2 = w_pixel_obj * w_outcontour / w_pixel_obj  # 这里需要根据实际情况调整公式
            x = (x_1 + x_2) / 2

            print(f"精确化后的目标实际边长 x: {x:.2f}")


            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()      

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
            
            #  格式化文字 (保留两位小数)
            text_w = f"W: {w_pixel_obj:.2f}px"
            text_h = f"H: {h_pixel_obj:.2f}px"

            # 计算标点位置 (取边中点再偏移一点，避免压线)
            # 宽度的标点：顶边 (tl 和 tr) 的中心
            pos_w = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2) - 10)

            # 高度的标点：左边 (tl 和 bl) 的中心
            pos_h = (int((tl[0] + bl[0]) / 2) - 80, int((tl[1] + bl[1]) / 2))

            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_all, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_rectangle", img_all)#显示拟合的矩形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            return distance, x
        
        elif len(approx) == 3:
            print(f"目标为三角形，顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            # 1. 计算像素长度和宽度（重新排序角点）
            rect = np.zeros((4, 2), dtype="float32")   

            sum = refined_corners.sum(axis=1)
            rect[0] = refined_corners[np.argmin(sum)]       # 左上 TL
            rect[2] = refined_corners[np.argmax(sum)]       # 右下 BR
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
            w_pixel_obj = (width_top + width_bottom) / 2
            h_pixel_obj = (height_left + height_right) / 2
            print(f"精确化后的目标像素宽度 w_pixel_obj: {w_pixel_obj:.2f},精确化后的目标像素高度 h_pixel_obj: {h_pixel_obj:.2f}")
            # 2. 计算实际边长
            x_1 = h_pixel_obj * h_outcontour / h_pixel_obj  # 这里需要根据实际情况调整公式
            x_2 = w_pixel_obj * w_outcontour / w_pixel_obj  # 这里需要根据实际情况调整公式
            x = (x_1 + x_2) / 2

            print(f"精确化后的目标实际边长 x: {x:.2f}")


            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()      

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
            
            #  格式化文字 (保留两位小数)
            text_w = f"W: {w_pixel_obj:.2f}px"
            text_h = f"H: {h_pixel_obj:.2f}px"

            # 计算标点位置 (取边中点再偏移一点，避免压线)
            # 宽度的标点：顶边 (tl 和 tr) 的中心
            pos_w = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2) - 10)

            # 高度的标点：左边 (tl 和 bl) 的中心
            pos_h = (int((tl[0] + bl[0]) / 2) - 80, int((tl[1] + bl[1]) / 2))

            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_all, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_rectangle", img_all)#显示拟合的矩形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            return distance, x
        else:
            print(f"目标为其他形状，顶点数：{len(approx)}")

    else:
        print("未识别到目标轮廓")
        return None

if __name__ == "__main__":
    image_path = r"picture/3_26_1.png"
    D,x =measure_target(image_path, f_pixel = 700, h_outcontour = 28.3,w_outcontour=21.0, real_side_limit=(10, 16))
    print(f"距离 D: {D} cm, 边长 x: {x} cm")