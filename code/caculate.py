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
            #print(f"外轮廓近似多边形顶点数：{len(approx)}")

            refined_corners = tools.refine_approx(approx, img_gray)

            w_pixel_ex,h_pixel_ex,text,pos = tools.caculate_square_x(refined_corners)

            
            # [text_w,text_h] = text
            # [pos_w,pos_h] = pos

            #print(f"精确化后的外边框像素宽度 w_pixel_ex: {w_pixel_ex:.2f},精确化后的外边框像素高度 h_pixel_ex: {h_pixel_ex:.2f}")

            # 2. 根据公式计算
 
            distance_1 = h_outcontour * f_pixel / h_pixel_ex
            distance_2 = w_outcontour * f_pixel / w_pixel_ex
            distance = (distance_1 + distance_2) / 2
            #print(f"距离 D: {distance:.2f} cm")

        # 二、计算边长 x

        peri = cv2.arcLength(cnts[-1], True)
        approx = cv2.approxPolyDP(cnts[-1], 0.02 * peri, True)#第二个参数为拟合的多边形与原始轮廓的最大距离，越小越精确

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

            #print(f"精确化后的目标实际边长 x: {x:.2f}")


            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()      

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
            
            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_all, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_rectangle", img_all)#显示拟合的矩形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            return distance, x
        
        elif len(approx) == 3:
            print(f"---目标为三角形！，顶点数：{len(approx)}---")

            refined_corners = tools.refine_approx(approx, img_gray)

            x_pixel,text,pos = tools.caculate_triangle_x(refined_corners)

            #print(f"精确化后的目标三角形像素边长 x_pixel: {x_pixel:.2f}")

            # 2. 计算实际边长
            
            x = x_pixel * h_outcontour / h_pixel_ex  # 这里需要根据实际情况调整公式

            #print(f"精确化后的目标实际边长 x: {x:.2f}")


            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()      

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
            
           
            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text, pos, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_tritangle", img_all)#显示拟合的三角形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            return distance, x
        elif len(approx) > 4:

            print(f"---目标为圆形！---")
            D_pixel, radius, pos_circle,pos_text = tools.caculate_circle_x(cnts[-1])
            print(f"拟合圆的像素直径 D_pixel: {D_pixel:.2f}")

            # 2. 计算实际直径
            Dia = D_pixel * h_outcontour / h_pixel_ex
            print(f"目标圆的实际直径 Dia: {Dia:.2f} cm")

            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            img_all = img.copy()     
            
            text = f"Dia: {Dia:.2f}cm"
        
            # 画圆和标注
            cv2.circle(img_all, pos_circle , int(radius), (0, 255, 0), 2) # 画出识别到的圆
            cv2.putText(img_all, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("approx_circle", img_all)#显示拟合的圆形
            cv2.waitKey(0)

            return distance, Dia


    else:
        print("未识别到目标轮廓")
        return None

if __name__ == "__main__":
    image_path = r"picture/test_canny.jpg"
    D,x =measure_target(image_path, f_pixel = 700, h_outcontour = 28.3,w_outcontour=21.0, real_side_limit=(10, 16))
    print(f"距离 D: {D:.2f} cm, 边长/直径 x: {x:.2f} cm")