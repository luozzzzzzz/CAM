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
    """
    gray[gray < 50] = 0
    gray[gray >= 50] = 255#二值化，主要去除阴影
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 100, 150)
    
    #100：低阈值（threshold1），用于检测弱边缘。梯度值低于此阈值的像素会被丢弃。
    #150：高阈值（threshold2），用于检测强边缘。梯度值高于此阈值的像素被认为是确定的边缘。

    #cv2.imshow("Edged", edged)
    
    # 3. 寻找轮廓并排序（取面积最大的，即 A4 纸外框）
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #print(f"原始轮廓数量：{len(cnts)}")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 4.用面积与周长之比的方法来过滤掉一些不规则的轮廓，保留更接近矩形的轮廓
    min_area = 100  # 根据实际情况调整最小面积阈值
    good_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        # 稍微减小 epsilon 参数 (0.02 -> 0.015)，防止三角形顶点被过度简化
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        
        # 凸性检测
        # if not cv2.isContourConvex(approx):
        #     continue

        num_vertices = len(approx)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        ratio = w / h if h != 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area != 0 else 0

        # --- 开始分类筛选逻辑 ---
        
        # 1. 三角形判定 (通常拟合为 3 个点)
        if num_vertices == 3:
            # 三角形的占空比在 0.4 到 0.6 之间比较合理
            if 0.4 < extent < 0.7 and 0.5 < ratio < 2.0:
                good_cnts.append(cnt)
                
        # 2. 矩形/A4纸判定 (通常拟合为 4 个点)
        elif num_vertices == 4:
            # 矩形占空比很高，通常 > 0.7 (理想是 1.0)
            if extent > 0.65 and 0.5 < ratio < 2.0:
                good_cnts.append(cnt)
                
        # 3. 圆形或多边形判定 (通常拟合点数 > 5)
        elif num_vertices > 4:
            # 圆形的占空比约为 0.785，且长宽比非常接近 1
            # 增加一个“圆度”校验 (4*pi*area / peri^2)，这是识别圆的最准方法
            circularity = (4 * np.pi * area) / (peri ** 2)
            if circularity > 0.7 and 0.8 < ratio < 1.2:
                good_cnts.append(cnt)
    
    ft_cnts = filter_similar_contours(good_cnts)
    
    a4_out, border_in, target = get_final_nested_contours(ft_cnts)

    num_cnt = 0
    if a4_out is not None:
        num_cnt += 1
    if border_in is not None:
        num_cnt += 1
    if target is not None:
        if isinstance(target[0], np.ndarray) and target[0].ndim > 1:
            num_cnt += len(target)
        else:
            num_cnt += 1
            

    print(f"轮廓数量：{num_cnt} (A4外框: {len(a4_out)}, 内沿: {len(border_in)}, 目标: {len(target)})")

    # 5.绘图可视化显示所有轮廓
    img_all = img.copy()
    #cv2.drawContours(img_all, [a4_out, border_in, target], -1, (0, 255, 0), 2)
    #cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
    #cv2.imshow("All Contours", img_all)
    #cv2.waitKey(-1)#调试时开启

    return a4_out, border_in, target, gray
def get_conTours_ex(image_path):
    
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

    
    gray[gray < 50] = 0
    gray[gray >= 50] = 255#二值化，主要去除阴影
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 100, 150)
    #100：低阈值（threshold1），用于检测弱边缘。梯度值低于此阈值的像素会被丢弃。
    #150：高阈值（threshold2），用于检测强边缘。梯度值高于此阈值的像素被认为是确定的边缘。

    #cv2.imshow("Edged", edged)
    
    # 3. 寻找轮廓并排序（取面积最大的，即 A4 纸外框）
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #print(f"原始轮廓数量：{len(cnts)}")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 4.用面积与周长之比的方法来过滤掉一些不规则的轮廓，保留更接近矩形的轮廓
    min_area = 100  # 根据实际情况调整最小面积阈值
    good_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        # 稍微减小 epsilon 参数 (0.02 -> 0.015)，防止三角形顶点被过度简化
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        
        # 凸性检测
        # if not cv2.isContourConvex(approx):
        #     continue

        num_vertices = len(approx)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        ratio = w / h if h != 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area != 0 else 0

       
        # . 矩形/A4纸判定 (通常拟合为 4 个点)
        if num_vertices == 4:
            # 矩形占空比很高，通常 > 0.7 (理想是 1.0)
            if extent > 0.65 and 0.5 < ratio < 2.0:
                good_cnts.append(cnt)
                
        
    
    ft_cnts = filter_similar_contours(good_cnts)
    
    a4_out, border_in, target = get_challenge_contours(ft_cnts)

    num_cnt = 0
    if a4_out is not None:
        num_cnt += 1
    if border_in is not None:
        num_cnt += 1
    if target is not None:
        if isinstance(target[0], np.ndarray) and target[0].ndim > 1:
            num_cnt += len(target)
        else:
            num_cnt += 1

    print(f"轮廓数量：{num_cnt} (A4外框: {len(a4_out)}, 内沿: {len(border_in)}, 目标: {len(target)})")

    # 5.绘图可视化显示所有轮廓
    img_all = img.copy()
    cv2.drawContours(img_all, [a4_out, border_in]+ target, -1, (0, 255, 0), 2)
    cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("All Contours", img_all)
    #cv2.waitKey(-1)#调试时开启

    return a4_out, border_in, target, gray

def get_conTours_cplx(image_path):

    return None
def get_challenge_contours(cnts):
    """
    针对发挥部分：锁定A4外框、黑框内沿，并获取内部所有独立的正方形目标 
    """
    # 1. 确保按面积从大到小排序 [cite: 3]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    a4_out = None
    border_in = None
    targets = []

    # 2. 第一步：寻找 A4 外框和黑框内沿 (父子关系)
    for i in range(len(cnts)):
        c_parent = cnts[i]
        for j in range(i + 1, len(cnts)):
            c_child = cnts[j]
            # 计算子轮廓质心 [cite: 3]
            M = cv2.moments(c_child)
            if M["m00"] == 0: continue
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # 判断嵌套关系 [cite: 3]
            if cv2.pointPolygonTest(c_parent, centroid, False) >= 0:
                a4_out = c_parent
                border_in = c_child
                break
        if a4_out is not None: break

    # 3. 第二步：在 border_in 内部搜集所有符合正方形特征的轮廓
    if border_in is not None:
        for k in range(len(cnts)):
            cand = cnts[k]
            # 跳过已经识别的外框和内框
            if cand is a4_out or cand is border_in: continue
            
            # 特征预筛选：必须在内沿内部 [cite: 3]
            M = cv2.moments(cand)
            if M["m00"] == 0: continue
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            if cv2.pointPolygonTest(border_in, centroid, False) >= 0:
                # 形状判定：正方形 (4个顶点) 
                peri = cv2.arcLength(cand, True)
                approx = cv2.approxPolyDP(cand, 0.02 * peri, True)
                if len(approx) == 4:
                    targets.append(cand)
    if targets:
        targets = sorted(targets, key=lambda c: cv2.arcLength(c, True), reverse=True)
    # 返回结果：[A4外框, 黑框内沿, 目标列表]
    return a4_out, border_in, targets
    
def get_final_nested_contours(cnts):
    """
    寻找最深的嵌套链，并保留面积最小的三个轮廓
    """
    # 1. 按面积从大到小排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    best_chain = []

    # 2. 遍历每一个轮廓作为起始点（父容器）
    for i in range(len(cnts)):
        current_chain = [cnts[i]]
        last_parent = cnts[i]
        
        # 3. 寻找该轮廓内部的所有嵌套子轮廓
        for j in range(i + 1, len(cnts)):
            child_candidate = cnts[j]
            
            # 计算质心
            M = cv2.moments(child_candidate)
            if M["m00"] == 0: continue
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # 判断质心是否在当前父轮廓内部
            if cv2.pointPolygonTest(last_parent, centroid, False) >= 0:
                current_chain.append(child_candidate)
                last_parent = child_candidate # 更新父容器，寻找下一层嵌套
        
        # 记录发现的最长嵌套链
        if len(current_chain) > len(best_chain):
            best_chain = current_chain

    # 4. 如果嵌套层数 >= 3，保留面积最小的三个（即列表的最后三个）
    if len(best_chain) >= 3:
        # 面积从大到小排列，-3: 表示取最后三个
        final_three = best_chain[-3:] 
        # 此时得到的顺序是：[倒数第三大(A4外), 倒数第二大(内沿), 最小(中心图形)]
        a4_out = final_three[0]
        border_in = final_three[1]
        target = final_three[2]
        return a4_out, border_in, target
    
    print("未找到完整的三层嵌套结构，返回发现的所有层")
    return None # 如果不足三层，则返回发现的所有层
    return None # 如果不足三层，则返回发现的所有层

def filter_similar_contours(cnts, dist_thresh=10, area_ratio_thresh=0.9, shape_thresh=0.1):
    if not cnts:
        return []

    # 1. 按面积从大到小排序，确保优先保留“外侧”或更完整的轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    unique_cnts = []
    centers = []
    areas = []

    for cnt in cnts:
        # 计算当前轮廓的特征
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        curr_cx = int(M["m10"] / M["m00"])
        curr_cy = int(M["m01"] / M["m00"])
        curr_area = M["m00"]
        
        is_duplicate = False
        
        # 2. 与已经保留的唯一轮廓逐一比对
        for i, (prev_cx, prev_cy) in enumerate(centers):
            # a. 计算重心欧氏距离
            dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            
            # b. 计算面积比例
            area_ratio = curr_area / areas[i] # 因为已排过序，所以必定 <= 1
            
            # c. 计算形状相似度 (Hu矩)
            # cv2.CONTOURS_MATCH_I1 是最常用的匹配指标，越小越像
            shape_similarity = cv2.matchShapes(cnt, unique_cnts[i], cv2.CONTOURS_MATCH_I1, 0.0)

            # --- 判定准则 ---
            # 如果重心离得近，且面积、形状都高度相似，则判定为重复
            if dist < dist_thresh and area_ratio > area_ratio_thresh and shape_similarity < shape_thresh:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_cnts.append(cnt)
            centers.append((curr_cx, curr_cy))
            areas.append(curr_area)
            
    return unique_cnts

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
    
    

    # 2. 绘制识别结果用于预览确认
     
    text = "Dia: cm"
    # 获取文字大小以实现居中
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # 计算位置：圆心坐标 - (文字宽度一半, 向上偏移)
    pos_circle = (int(x), int(y))
    pos_text = (int(x - tw / 2), int(y - th / 2 - 10))

    return D_pixel,radius,pos_circle,pos_text
    