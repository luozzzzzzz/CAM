import cv2
import numpy as np
import tools 

def measure_target_ex(image_path, f_pixel, h_outcontour = 29.7,w_outcontour=21.0):
    """
    测量图像中正方形的距离 D 和边长 x
    :param image_path: 图片路径
    :param f_pixel: 相机的像素焦距 (通过预先测量获得)
    :param real_side_limit: 题目要求的边长范围 (cm)
    :return: D (距离), x (边长)
    """
    a4_out,border_in,target,img_gray = tools.get_conTours_ex(image_path)
    
    if a4_out is not None and target is not None:
        #一、根据外轮廓计算距离 D
        # 假设最大的轮廓就是 A4 纸
        peri = cv2.arcLength(a4_out, True)
        approx = cv2.approxPolyDP(a4_out, 0.02 * peri, True)

        if len(approx) == 4:
            #print(f"外轮廓近似多边形顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            w_pixel_ex,h_pixel_ex,text,pos = tools.caculate_square_x(refined_corners)

            #print(f"精确化后的外边框像素宽度 w_pixel_ex: {w_pixel_ex:.2f},精确化后的外边框像素高度 h_pixel_ex: {h_pixel_ex:.2f}")

            # 2. 根据公式计算
 
            distance_1 = h_outcontour * f_pixel / h_pixel_ex
            distance_2 = w_outcontour * f_pixel / w_pixel_ex
            distance = (distance_1 + distance_2) / 2
            #print(f"距离 D: {distance:.2f} cm")

        # 二、计算边长 x

        peri = cv2.arcLength(target[-1], True)
        approx = cv2.approxPolyDP(target[-1], 0.02 * peri, True)#第二个参数为拟合的多边形与原始轮廓的最大距离，越小越精确

        if len(approx) == 4:
            print(f"---目标为正方形!---，顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            w_pixel_obj,h_pixel_obj,text,pos = tools.caculate_square_x(refined_corners)
            
            [text_w,text_h] = text
            [pos_w,pos_h] = pos
            #print(f"精确化后的目标像素宽度 w_pixel_obj: {w_pixel_obj:.2f},精确化后的目标像素高度 h_pixel_obj: {h_pixel_obj:.2f}")
            
            # 2. 计算实际边长
            x_1 = h_pixel_obj * h_outcontour / h_pixel_ex  # 这里需要根据实际情况调整公式
            x_2 = w_pixel_obj * w_outcontour / w_pixel_ex  # 这里需要根据实际情况调整公式
            x = (x_1 + x_2) / 2

            print(f"正方形目标实际边长 x: {x:.2f}")


            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()      

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
            
            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_all, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.namedWindow("approx_rectangle", cv2.WINDOW_NORMAL)
            cv2.imshow("approx_rectangle", img_all)#显示拟合的矩形
            cv2.waitKey(0)
            

            return distance, x
        



    else:
        print("未识别到目标轮廓")
        return None

if __name__ == "__main__":
    image_path = r"picture/4_13_4.jpg"
    D,x = measure_target_ex(image_path, f_pixel = 2581.87)
    print(f"距离 D: {D:.2f} cm, 边长/直径 x: {x:.2f} cm")