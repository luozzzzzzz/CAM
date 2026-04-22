import cv2
import numpy as np

def correct_a4_perspective(image, corners, bottom_mid_marker=None):
    """
    对发生透视畸变的 A4 纸进行矫正，并保证中轴线对齐
    :param image: 原始畸变图像
    :param corners: 4个角点的坐标列表或数组，顺序需为：左上, 右上, 右下, 左下
    :param bottom_mid_marker: 底边物理中点标记的坐标 (x, y)，可选
    :return: 矫正后的正视图图像
    """
    # 将角点转换为 float32 格式的 numpy 数组
    src_pts = np.array(corners, dtype="float32")
    
    # 1. 计算矫正后的图像尺寸
    if bottom_mid_marker is None:
        # 获取图像中轴线的像素宽度，取最大值作为基准宽度 H
        H_left = np.linalg.norm(src_pts[0] - src_pts[3])
        H_right = np.linalg.norm(src_pts[1] - src_pts[2])
        H = int((H_left + H_right)/2)
        
        # 根据 A4 纸标准比例 1 : 1.414，计算宽度 W
        W = int(H / 1.4142)
    else:
        src_mid_under = np.array([bottom_mid_marker], dtype="float32")
        print(src_mid_under.shape)
        k=(src_pts[0][0]-src_mid_under[0][0])/(src_pts[0][0]-src_pts[1][0])
        src_mid_upper=src_pts[0]*(1-k)+src_pts[1]*k
        print(src_mid_under,src_mid_upper)
        H=int(np.linalg.norm(src_mid_under-src_mid_upper))
        W=int(H / 1.4142)
    
    # 2. 定义理想状态下的目标角点坐标 (左上, 右上, 右下, 左下)
    dst_pts = np.array([
        [0, 0],       # 左上角
        [W - 1, 0],   # 右上角
        [W - 1, H - 1], # 右下角
        [0, H - 1]    # 左下角
    ], dtype="float32")

    # 3. 计算单应性矩阵 H_matrix
    if bottom_mid_marker is not None:
        # 【方案二】有物理中点标记（5点透视）
        print("模式：使用 5 点计算单应性矩阵（含物理中点标记）")
        
        # 将原图中的中点标记坐标加入源点
        src_mid = np.array([bottom_mid_marker], dtype="float32")
        src_pts = np.vstack((src_pts, src_mid))
        
        # 强制将目标中点设定在图像的绝对中心 (W/2, H-1)
        dst_mid = np.array([[W / 2.0, H - 1]], dtype="float32")
        dst_pts = np.vstack((dst_pts, dst_mid))
        
        # 因为点数 > 4，使用 cv2.findHomography（基于最小二乘/RANSAC求解）
        # cv2.RANSAC 用于过滤可能存在误差的点，5.0 是阈值
        H_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        # 【方案一】无标记（标准的 4 点透视）
        print("模式：使用 4 点计算透视变换矩阵")
        
        # 仅有4个点，使用 cv2.getPerspectiveTransform 即可得到唯一解
        H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. 执行透视变换进行图像重采样
    warped_image = cv2.warpPerspective(image, H_matrix, (W, H))
    
    # 5. 绘制中轴线（用于验证）
    # 在矫正后的图像中，中轴线就是贯穿 x = W/2 的竖直线
    center_x = int(W / 2)
    cv2.line(warped_image, (center_x, 0), (center_x, H), (0, 255, 0), 2)  # 画一条绿色中轴线
    cv2.circle(warped_image, (center_x, H - 1), 5, (0, 0, 255), -1)       # 在底部中点画一个红点
    
    return warped_image

# ================= 测试代码 =================
if __name__ == "__main__":
    # 创建一张 800x800 的黑色空白图像作为测试背景
    mock_image = np.zeros((800, 800, 3), dtype="uint8")
    
    # 模拟相机拍到的畸变四边形（上边窄，下边宽，且整体右倾）
    distorted_corners = [
        [300, 150],  # 左上
        [500, 180],  # 右上
        [650, 700],  # 右下
        [150, 650]   # 左下
    ]
    
    # 在原图上画出这个畸变的 A4 纸轮廓（白色）
    pts = np.array(distorted_corners, np.int32)
    cv2.polylines(mock_image, [pts], True, (255, 255, 255), 2)
    
    # 模拟原图中由于透视缩短发生的偏移的“物理中点”
    # 注意：这里的点并非线段 [150,650] 到 [650,700] 的算术中点 (400,675)
    # 因为右侧距离相机更近，实际物理中点在图像上会偏向左侧（近大远小规律）
    physical_marker_in_image = [380, 665] 
    cv2.circle(mock_image, tuple(physical_marker_in_image), 5, (0, 0, 255), -1)
    
    # 1. 运行无标记矫正（仅使用4个角点）
    result_4pt = correct_a4_perspective(mock_image.copy(), distorted_corners)
    
    # 2. 运行有标记矫正（引入第5点）
    result_5pt = correct_a4_perspective(mock_image.copy(), distorted_corners, bottom_mid_marker=physical_marker_in_image)
    
    # 显示结果 (如果你的环境无法弹出窗口，可以将 cv2.imshow 替换为 cv2.imwrite 保存图片)
    cv2.imshow("Original Distorted Image", mock_image)
    cv2.imshow("Corrected (4 Points)", result_4pt)
    cv2.imshow("Corrected (5 Points, with Centerline Locked)", result_5pt)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()