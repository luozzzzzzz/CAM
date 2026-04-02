import cv2
import numpy as np

def get_conTours(image_path):
    
    """
    input param:文件路径
    output param:过滤后的轮廓列表,原图像的灰度图
    """
    # 1. 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法加载图片")
        return None
    print(f"图片读取成功！大小：{img.shape[0]} x {img.shape[1]}")

    # 2. 预处理：灰度化 -> 高斯滤波 -> 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 100, 150)
    #100：低阈值（threshold1），用于检测弱边缘。梯度值低于此阈值的像素会被丢弃。
    #150：高阈值（threshold2），用于检测强边缘。梯度值高于此值的像素被认为是确定的边缘。

    cv2.imshow("Edged", edged)
    # 3. 寻找轮廓并排序（取面积最大的，即 A4 纸外框）
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #print(f"原始轮廓数量：{len(cnts)}")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 4.用面积与周长之比的方法来过滤掉一些不规则的轮廓，保留更接近矩形的轮廓
    cnts = [cnt for cnt in cnts if cv2.arcLength(cnt, True) > 0 and cv2.contourArea(cnt) / cv2.arcLength(cnt, True) >= 2.0]
    cnts = [cnts[i] for i in range(0, len(cnts), 2)]
    print(f"轮廓数量：{len(cnts)}")

    # 5.绘图可视化显示所有轮廓
    img_all = img.copy()
    cv2.drawContours(img_all, cnts, -1, (0, 255, 0), 2)
    cv2.imshow("All Contours", img_all)
    #cv2.waitKey(0)#调试时开启

    return cnts,gray

def refine_approx(approx,img_gray):
    """
    增加顶点精度 将 approx 转换为 float32 格式
    param approx: 近似多边形的顶点坐标
    param img_gray: 原图像的灰度图
    return: 精细化的角点坐标
    """
    cnts = np.float32(approx).reshape(-1, 2)

    # 设置亚像素搜索停止准则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 在原灰度图中精细化角点位置
    # gray 是预处理后的灰度图，(5,5) 是搜索窗口大小
    refined_cnts = cv2.cornerSubPix(img_gray, cnts, (5, 5), (-1, -1), criteria)

    # 此时 refined_cnts 里的坐标是带小数的，比如 (120.45, 345.78)用这些点去算距离 D，精度会有一个质的飞跃。
    return refined_cnts

def caculate_square_x(cnts):
            """
            输入精确后的端点，得到多边形的边长
            param cnts:精确后的端点
            param rect:排序后的端点
            """
     # 1. 计算像素长度和宽度（重新排序角点）
            rect = np.zeros((4, 2), dtype="float32")   

            sum = cnts.sum(axis=1)
            rect[0] = cnts[np.argmin(sum)]       # 左上 TL
            rect[2] = cnts[np.argmax(sum)]       # 右下 BR
            diff = np.diff(cnts, axis=1)
            rect[1] = cnts[np.argmin(diff)]    # 右上 TR
            rect[3] = cnts[np.argmax(diff)]    # 左下 BL

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

            #  格式化文字 (保留两位小数)
            text_w = f"W: {w_pixel:.2f}px"
            text_h = f"H: {h_pixel:.2f}px"
            text = [text_w,text_h]

            # 计算标点位置 (取边中点再偏移一点，避免压线)
            # 宽度的标点：顶边 (tl 和 tr) 的中心
            pos_w = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2) - 10)

            # 高度的标点：左边 (tl 和 bl) 的中心
            pos_h = (int((tl[0] + bl[0]) / 2) - 80, int((tl[1] + bl[1]) / 2))
            pos = [pos_w,pos_h]
            
            return w_pixel,h_pixel,text,pos

def caculate_triangle_x(cnts):
    # pts 是 (3, 2) 的坐标阵
    pts = np.zeros((3, 2), dtype="float32")
    pts[0], pts[1], pts[2] = cnts[0], cnts[1], cnts[2]
    
    # 计算三边像素距离
    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[1] - pts[2])
    d3 = np.linalg.norm(pts[2] - pts[0])
    
    # 取平均像素边长
    avg_x = (d1 + d2 + d3) / 3.0

    #绘图的一些参数
    text = f"W: {avg_x:.2f}px"
    center = np.mean(pts, axis=0)
    pos = (int(center[0]), int(center[1]))

    return avg_x,text,pos


def caculate_circle_x(cnts):
# 1. 计算像素直径
    (x, y), radius = cv2.minEnclosingCircle(cnts)#拟合最小外接圆
    D_pixel = radius * 2
    print(f"拟合圆的像素直径 D_pixel: {D_pixel:.2f}")

    # 2. 绘制识别结果用于预览确认
     
    text = "Dia: cm"
    # 获取文字大小以实现居中
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # 计算位置：圆心坐标 - (文字宽度一半, 向上偏移)
    pos_circle = (int(x), int(y))
    pos_text = (int(x - tw / 2), int(y - th / 2 - 10))

    return D_pixel,radius,pos_circle,pos_text
    