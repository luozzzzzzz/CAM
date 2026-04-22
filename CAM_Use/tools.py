import cv2
import numpy as np

def get_conTours(image):
    """
    input param:彩色图像
    output param:过滤后的轮廓列表,原图像的灰度图
    """
    # 1. 加载图片
    img=image
    if img is None:
        #print("错误：无法加载图片")
        return None
    #print(f"图片读取成功！大小：{img.shape[0]} x {img.shape[1]}")

    # 2. 预处理:  灰度化 -> 高斯滤波 -> 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 150)
    
    #100：低阈值（threshold1），用于检测弱边缘。梯度值低于此阈值的像素会被丢弃。
    #150：高阈值（threshold2），用于检测强边缘。梯度值高于此阈值的像素被认为是确定的边缘。

    
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
    
    final_cnts = get_final_nested_contours(ft_cnts)
    #print(f"轮廓数量：{len(final_cnts)}")
    """
    # 5.绘图可视化显示所有轮廓
    img_all = img.copy()
    cv2.drawContours(img_all, final_cnts, -1, (0, 255, 0), 2)
    cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("All Contours", img_all)
    #cv2.waitKey(-1)#调试时开启
    """
    return final_cnts,gray
def get_conTours_ex(image):
    
    """
    input param:文件路径
    output param:过滤后的轮廓列表,原图像的灰度图
    """
    img=image
    if img is None:
        #print("错误：无法加载图片")
        return None
    # 2. 预处理：灰度化 -> 高斯滤波 -> 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray[gray < 80] = 0
    gray[gray >= 80] = 255#二值化，主要去除阴影
    
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
        
        # 保持凸性检测
        if not cv2.isContourConvex(approx):
            continue

        num_vertices = len(approx)
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
    
    a4_out, border_in, target = get_cplx_conTours(ft_cnts,gray)

    num_cnt = 0
    if a4_out is not None:
        num_cnt += 1
    else:
        print(1)
    if border_in is not None:
        num_cnt += 1
    else:
        print(2)
    if len(target)>0:
        if isinstance(target[0], np.ndarray) and target[0].ndim > 1:
            num_cnt += len(target) 
        else:
            num_cnt += 1
    else:
        print(3)
    if a4_out is not None and border_in is not None and len(target)>0:
        print(f"轮廓数量：{num_cnt} (A4外框: {len(a4_out)}, 内沿: {len(border_in)}, 目标: {len(target)})\n")
    else:
        print("没有检测到三层轮廓")

    # 5.绘图可视化显示所有轮廓
    #img_all = img.copy()
    #cv2.drawContours(img_all, [a4_out, border_in]+ target, -1, (0, 255, 0), 2)
    #cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
    #cv2.imshow("All Contours", img_all)
    #cv2.waitKey(-1)#调试时开启

    return a4_out, border_in, target, gray,edged

def get_cplx_conTours(ft_cnts,gray_img):
    """
    ft_cnts: 预过滤后的轮廓列表
    """
    # 按面积降序
    cnts = sorted(ft_cnts, key=cv2.contourArea, reverse=True)
    
    a4_out = None
    border_in = None
    targets = []

    # 第一步：锁定边框
    for i in range(len(cnts)):
        c_parent = cnts[i]
        for j in range(i + 1, len(cnts)):
            c_child = cnts[j]
            M = cv2.moments(c_child)
            if M["m00"] == 0: continue
            centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            if cv2.pointPolygonTest(c_parent, centroid, False) >= 0 and cv2.contourArea(c_child)/cv2.contourArea(c_parent)>0.7 and cv2.contourArea(c_child)/cv2.contourArea(c_parent)<0.9:
                a4_out, border_in = c_parent, c_child
                break
        if a4_out is not None: break

    # 第二步：提取目标
    if border_in is not None:
        for cand in cnts:
            if cand is a4_out or cand is border_in: continue
            
            M = cv2.moments(cand)
            if M["m00"] == 0: continue
            centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # 必须在黑框内沿内部
            if cv2.pointPolygonTest(border_in, centroid, False) >= 0:
                peri = cv2.arcLength(cand, True)
                approx = cv2.approxPolyDP(cand, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # 正常正方形
                    targets.append(cand)
                elif len(approx) > 4:
                    # 复杂重叠多边形 -> 拆分
                    split_res = split_fused_squares(cand,gray_img)
                    targets.extend(split_res)

    targets = sorted(targets, key=cv2.contourArea, reverse=True)        

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
        return final_three
    #print("未找到完整的三层嵌套结构，返回发现的所有层")
    return best_chain # 如果不足三层，则返回发现的所有层

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
            #w_pixel = h_pixel/1.4142 #选择更相信高，利用A4纸先验
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
    #print(f"拟合圆的像素直径 D_pixel: {D_pixel:.2f}")

    # 2. 绘制识别结果用于预览确认
     
    text = "Dia: cm"
    # 获取文字大小以实现居中
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # 计算位置：圆心坐标 - (文字宽度一半, 向上偏移)
    pos_circle = (int(x), int(y))
    pos_text = (int(x - tw / 2), int(y - th / 2 - 10))

    return D_pixel,radius,pos_circle,pos_text


def split_fused_squares(cand_contour, gray_img, debug_image=None):
    """
    直接利用原图灰度图进行角点检测并拟合
    :param cand_contour: 原始复杂轮廓
    :param gray_img: 全图灰度图 (用于检测)
    :param debug_image: 用于绘制调试信息的原图
    """
    #debug_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # 1. 确定搜索范围 (ROI)
    x, y, w, h = cv2.boundingRect(cand_contour)
    pad = 20
    # 局部画布尺寸
    roi_h, roi_w = h + pad*2, w + pad*2
    # 创建一张全黑的局部画布
    roi_gray = np.zeros((roi_h, roi_w), dtype=np.uint8)
    
    # 计算平移偏移量
    offset_x = max(0, x - pad)
    offset_y = max(0, y - pad)
    offset = np.array([offset_x, offset_y])

    # 2. 将目标轮廓绘制到局部画布上 (填充为实心白色)
    # 这样可以过滤掉 ROI 矩形区域内所有不属于该轮廓的干扰点
    shifted_cnt = (cand_contour.astype(np.int32) - offset).astype(np.int32)
    cv2.drawContours(roi_gray, [shifted_cnt], -1, 255, -1)
    #cv2.imshow("ROI ", roi_gray)  # 调试时开启

    debug_image = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    # 2. 直接在灰度 ROI 上检测角点
    # 对于黑色实心正方形，直角顶点是 Harris 响应的最强点
    corners = cv2.goodFeaturesToTrack(roi_gray, maxCorners=20, 
                                      qualityLevel=0.3, minDistance=5, blockSize=5)
    
    #print(f"拟合顶点数量：{len(corners) if corners is not None else 0}")

    if corners is None:
        #print("未检测到任何角点，无法拆分")
        return []

    # 转化为相对于 ROI 的坐标列表
    raw_pools = [c.ravel() for c in corners]
    pools_new = []

    edges = roi_gray

    # 3. 计算所有点之间的距离矩阵，寻找每个点的最近邻距离
    # 使用 NumPy 的广播机制快速计算
    pts_array = np.array(raw_pools) # (N, 2)

    diff = pts_array[:, np.newaxis, :] - pts_array[np.newaxis, :, :] # (N, N, 2)

    dist_matrix = np.linalg.norm(diff, axis=2) # (N, N)
    
    # 将对角线（点到自身的距离 0）填充为无穷大，方便取最小值
    np.fill_diagonal(dist_matrix, np.inf)

    # 4. 遍历并利用自适应半径进行角度验证
    for i in range(len(raw_pools)):
        p_center = raw_pools[i]
        
        # --- 核心逻辑：动态计算半径 ---
        min_dist = np.min(dist_matrix[i])
        # 半径取最近邻距离的一半，设定上下限防止极端情况
        # 上限 20 像素防止采样太远，下限 2 像素确保能算出向量
        if min_dist <5:
            adaptive_radius = 2
        else:
            adaptive_radius = np.clip(min_dist *1/2, 2, 30)
        
        
        sample_mask = np.zeros_like(edges)
        cv2.circle(sample_mask, (int(p_center[0]), int(p_center[1])), int(adaptive_radius), 255, 1)
        intersection = cv2.bitwise_and(sample_mask, edges)
        iy, ix = np.where(intersection > 0)

        if len(ix) >= 2:
            # 取距离最远的两个采样点作为向量末端
            p_edge1 = np.array([ix[0], iy[0]])
            p_edge2 = np.array([ix[-1], iy[-1]])
            
            v1, v2 = p_edge1 - p_center, p_edge2 - p_center
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if l1 > 0 and l2 > 0:
                cos_theta = np.dot(v1, v2) / (l1 * l2)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                #print(f"第 {i+1} 点 {p_center} 的自适应半径: {adaptive_radius:.2f}, 夹角: {angle:.2f} 度")
                # 直角判定
                if 75 <= angle <= 105 or angle<7 or angle>173:
                    pools_new.append(p_center)
        else:
            continue
            #print(f"第 {i+1} 点 {p_center} 的自适应半径: {adaptive_radius:.2f}, 采样点不足，无法计算夹角")
    # --- 修正后的绘图部分 调试时开启！---
    # if debug_image is not None and len(ix) > 0:
    #         for tx, ty in zip(ix, iy):
    #             # 1. 转换到全局坐标
    #             gx = tx 
    #             gy = ty 
                
    #             # 2. 绘制交点作为小黄色圆点 (radius=2)
    #             #cv2.circle(debug_image, (gx, gy), 1, [0,0,255], 1)
            
    #         # 【可选 bonus】同时也把这个自适应圆周本身画在 debug_image 上，方便对照
    #         g_center = (int(p_center[0] ), int(p_center[1] ))
    #         # 同样使用黄色，线宽设为 1]
    #         cv2.circle(debug_image, g_center, int(adaptive_radius), [0,0,255], 1)

    # if debug_image is not None:
    #     for p in pools_new:
    #         # p 是 [x, y]，offset 是 [ox, oy]
    #         # 确保 cp 是一个一维数组 [x_global, y_global]
    #         cp = p.ravel() 
    #         # 绘图时强制转换为 int 类型的元组 (x, y)
    #         center = (int(cp[0]), int(cp[1]))
    #         cv2.circle(debug_image, center, 5, (0, 255, 0), -1)

    #     cv2.namedWindow("Detected Corners", cv2.WINDOW_NORMAL)
    #     cv2.imshow("Detected Corners", debug_image)

    squares = []
    used_indices = set()

    # 3. 遍历顶点：寻找符合正方形几何特征的三点组
    for i in range(len(pools_new)):
        if i in used_indices: continue
        for j in range(len(pools_new)):
            if i == j or j in used_indices: continue
            for k in range(len(pools_new)):
                if i == k or j == k or k in used_indices: continue
                
                # B 为潜在直角顶点，A、C 为两端点
                A, B, C = pools_new[i], pools_new[j], pools_new[k]
                v_BA = A - B
                v_BC = C - B
                l_BA = np.linalg.norm(v_BA)
                l_BC = np.linalg.norm(v_BC)
                
                # 判定条件：两条边长度比例接近 1 且 夹角接近 90 度
                if 0.95 < (l_BA / l_BC) < 1.05:
                    cos_theta = np.dot(v_BA, v_BC) / (l_BA * l_BC)
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    if 75 <= angle <= 105 or angle<7 or angle>173:
                        D = A + C - B # 补全第四点
                        sq = np.array([A, B, C, D], dtype=np.float32)
                        
                        # 转换回全局坐标并存储
                        sq_global = (sq + offset).astype(np.int32).reshape(-1, 1, 2)
                        squares.append(sq_global)
                        
                        used_indices.update([i, j, k])
                        # NMS: 如果 D 点本身也在检测到的角点池中，将其剔除
                        for m, pt in enumerate(pools_new):
                            if m not in used_indices and np.linalg.norm(pt - D) < 100:
                                used_indices.add(m)
                                break
                        break
            if i in used_indices: break

    #4. 剩下的两两成对点：根据边拟合
    remaining = [pools_new[idx] for idx in range(len(pools_new)) if idx not in used_indices]
    
    # 将 cand_contour 转换到 ROI 坐标系，用于内部测试
    shifted_cand_cnt = (cand_contour.astype(np.int32) - offset).astype(np.int32)
    
    # 计算 ROI 图像中心点（用于优化正方形中心位置）
    roi_center = np.array([roi_gray.shape[1] / 2.0, roi_gray.shape[0] / 2.0])
    
    while len(remaining) >= 2:
        p1 = remaining.pop(0)
        # 寻找最近的邻居
        remaining.sort(key=lambda p: np.linalg.norm(p1 - p))
        p2 = remaining.pop(0)
        
        v = p2 - p1
        # 正交向量 (边长强制相等)
        n1 = np.array([-v[1], v[0]])
        n2 = np.array([v[1], -v[0]])
        
        # 验证两个方向的中心点
        c1 = (p1 + p2 + (p2 + n1) + (p1 + n1)) / 4
        c2 = (p1 + p2 + (p2 + n2) + (p1 + n2)) / 4
        
        # 使用 pointPolygonTest 确保还原的正方形在原始重叠多边形区域内
        d1 = cv2.pointPolygonTest(shifted_cand_cnt, (float(c1[0]), float(c1[1])), False)
        d2 = cv2.pointPolygonTest(shifted_cand_cnt, (float(c2[0]), float(c2[1])), False)
        
        # 优先选择在轮廓内的方向，其次选择中心距离图像中心最近的
        dist1 = np.linalg.norm(c1 - roi_center)
        dist2 = np.linalg.norm(c2 - roi_center)
        
        if d1 >= 0 and d2 >= 0:
            # 两个都在内部，选择中心更接近图像中心的
            best_n = n1 if dist1 <= dist2 else n2
        elif d1 >= 0:
            best_n = n1
        elif d2 >= 0:
            best_n = n2
        else:
            # 两个都不在内部，选择中心更接近图像中心的
            best_n = n1 if dist1 <= dist2 else n2
        
        sq = np.array([p1, p2, p2 + best_n, p1 + best_n], dtype=np.float32)
        squares.append((sq + offset).astype(np.int32).reshape(-1, 1, 2))
    
    return squares