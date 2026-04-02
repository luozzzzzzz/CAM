import cv2
import numpy as np
import tools
#from tools import get_conTours_ex

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

def extract_squares(image_path, output_size=56):
    """
    从图像中提取正方形并return
    :param image_path: 输入图像
    :param output_size: 输出正方形的像素边长
    """
    # 1. 读取图片
    squares=[]
    img=cv2.imread(image_path)
    _,_,cnts,_ = tools.get_conTours_ex(img)
    print("0")
    # 遍历轮廓，提取正方形
    count=0
    for cnt in cnts:
        count+=1
        # 多边形逼近，提取轮廓的顶点
        perimeter = cv2.arcLength(cnt, True)
        print("1")
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        print("2")

        # 提取顶点并做透视矫正
        pts = approx.reshape(4, 2)
        warped_square = four_point_transform(img, pts, output_size)
        squares.append(warped_square)
        # 展示正方形
        """
        cv2.imshow(f"{count}",warped_square)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
    return squares


# ------------------- 执行入口 -------------------
if __name__ == "__main__":
    # 请修改此处为你的图片路径（和脚本同文件夹直接写文件名即可）
    IMAGE_PATH = "data/aaa.jpg"

    try:
        extract_squares(
            image_path=IMAGE_PATH,
            output_size=600,   # 输出正方形的像素大小，越大越清晰
            min_area=2000,     # 最小面积阈值，识别不到可调小，有噪点可调大
            aspect_ratio_tol=0.15  # 长宽比容差，方块倾斜大可调至0.2
        )
    except Exception as e:
        print(f"执行出错：{e}")