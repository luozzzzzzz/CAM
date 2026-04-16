import cv2
import numpy as np

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

def extract_squares(image_path, output_size=600, min_area=2000, aspect_ratio_tol=0.15):
    """
    核心函数：从图像中提取正方形并保存
    :param image_path: 输入图片的路径
    :param output_size: 输出正方形的像素边长
    :param min_area: 最小面积阈值，过滤噪点
    :param aspect_ratio_tol: 正方形长宽比容差，倾斜越大可适当调大
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确：{image_path}")

    # 2. 图像预处理（降噪+二值化，强化方块和背景的对比度）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 反转二值化：黑色方块转为白色，方便轮廓检测
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 查找轮廓（仅提取最外层轮廓，忽略方块内部的数字）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_count = 0
    square_images = []

    # 4. 遍历轮廓，筛选符合条件的正方形
    for cnt in contours:
        # 多边形逼近，提取轮廓的顶点
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # 筛选条件1：必须是四边形（4个顶点）
        if len(approx) != 4:
            continue
        # 筛选条件2：面积大于阈值，过滤微小噪点
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # 筛选条件3：长宽比接近1，判定为正方形
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h
        if abs(1 - aspect_ratio) > aspect_ratio_tol:
            continue

        # 符合条件，提取并保存正方形
        square_count += 1
        print(f"成功识别第{square_count}个正方形，面积：{area:.0f}")
        
        # 提取顶点并做透视矫正
        pts = approx.reshape(4, 2)
        warped_square = four_point_transform(img, pts, output_size)
        
        # 保存单个正方形文件
        save_path = f"square_{square_count}.png"
        cv2.imwrite(save_path, warped_square)
        print(f"已保存至：{save_path}")
        
        square_images.append(warped_square)
        # 在原图上绘制识别到的轮廓，用于调试核对
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

    # 保存调试图（带识别轮廓）
    cv2.imwrite("debug_contours.png", img)
    print(f"\n调试图已保存至：debug_contours.png，可查看轮廓识别是否准确")

    if square_count == 0:
        print("警告：未识别到符合条件的正方形，请调整min_area或aspect_ratio_tol参数")
    else:
        print(f"提取完成，共识别并保存{square_count}个正方形")

    return square_images

# ------------------- 执行入口 -------------------
if __name__ == "__main__":
    # 请修改此处为你的图片路径（和脚本同文件夹直接写文件名即可）
    IMAGE_PATH = "data/test.jpg"

    try:
        extract_squares(
            image_path=IMAGE_PATH,
            output_size=600,   # 输出正方形的像素大小，越大越清晰
            min_area=2000,     # 最小面积阈值，识别不到可调小，有噪点可调大
            aspect_ratio_tol=0.15  # 长宽比容差，方块倾斜大可调至0.2
        )
    except Exception as e:
        print(f"执行出错：{e}")