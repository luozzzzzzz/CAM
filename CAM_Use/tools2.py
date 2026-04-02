import cv2
import numpy as np
import tools

def order_points(pts):
    """
    对四边形的四个顶点排序，固定顺序：左上、右上、右下、左下
    用于保证透视变换的正确性
    """
    rect = np.zeros((4, 2), dtype="float32")
    # 左上点x+y和最小，右下点x+y和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上点x-y差最小，左下点x-y差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, output_size=500):
    """透视变换，将倾斜的四边形转正为标准正方形"""
    rect = order_points(pts)
    # 定义输出正方形的四个顶点坐标
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]], dtype="float32")
    # 计算变换矩阵并执行透视矫正
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))
    return warped

def extract_square(img,cnt, output_size=56):
    """
    从图像中提取正方形并return,
    :param image: 输入彩色图像,cv2格式,
    :param output_size: 输出正方形的像素边长
    :return 从图像中提取给定边框的正方形，MaixCam格式，可直接传入模型判断
    """
    # 多边形逼近，提取轮廓的顶点
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

    # 提取顶点并做透视矫正
    pts = approx.reshape(4, 2)
    warped_square = four_point_transform(img, pts, output_size)
    #将square转化为maix格式：
    sq=img.cv2image(warped_square,bgr=False,copy=False)  #注意此处的bgr到底是要False还是True

    # 展示正方形
    """
    cv2.imshow(f"{count}",warped_square)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return sq


def get_square_x(approx,img_gray,w_pixel_ex,h_pixel_ex,h_outcontour = 28.3,w_outcontour=21.0):
        refined_corners = tools.refine_approx(approx, img_gray)

        w_pixel_obj,h_pixel_obj,text,pos = tools.caculate_square_x(refined_corners)
        
        [text_w,text_h] = text
        [pos_w,pos_h] = pos
        #print(f"精确化后的目标像素宽度 w_pixel_obj: {w_pixel_obj:.2f},精确化后的目标像素高度 h_pixel_obj: {h_pixel_obj:.2f}")
        
        # 2. 计算实际边长
        x_1 = h_pixel_obj * h_outcontour / h_pixel_ex  # 这里需要根据实际情况调整公式
        x_2 = w_pixel_obj * w_outcontour / w_pixel_ex  # 这里需要根据实际情况调整公式
        x = (x_1 + x_2) / 2

        return x
