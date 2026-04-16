from maix import image, display, app, time, camera
import cv2
import numpy as np
import tools

def get_focal_length(image_path, real_distance, real_height=28.3):
    """
    通过已知距离的照片计算像素焦距
    :param image_path: 用于校准的照片路径
    :param real_distance: 拍摄时目标物到基准线的物理距离 (D)，单位 cm
    :param real_height: 参照物(A4纸)的实际物理高度，单位 cm
    :return: f_pixel (像素焦距)
    """
    cnts,img_gray = tools.get_conTours(image_path)

    if len(cnts) > 0:
        # 最大的轮廓就是 A4 纸
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
       
        if len(approx) == 4:
            print(f"近似多边形顶点数：{len(approx)}")

            #进一步精确端点
            refined_corners = tools.refine_approx(approx, img_gray)
            #计算得到像素宽度、高度以及有序端点

            w_pixel,h_pixel,text,pos = tools.caculate_square_x(refined_corners)

            
            [text_w,text_h] = text
            [pos_w,pos_h] = pos

            print(f"精确化后的外边框像素宽度 w_pixel: {w_pixel:.2f},精确化后的外边框像素高度 h_pixel: {h_pixel:.2f}")

            # 2. 根据公式计算像素焦距
            f_pixel = (h_pixel * real_distance) / real_height

            # 3. 绘制识别结果用于预览确认
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 转回彩色图以便绘制彩色轮廓
            img_all = img.copy()

            cv2.drawContours(img_all, approx, -1, (0, 255, 0), 2)
            #cv2.imshow("approx", img_all)#显示拟合得到的端点       

            cv2.drawContours(img_all, [approx], -1, (0, 255, 0), 2)
           


            # 绘制文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_all, text_w, pos_w, font, 0.6, (0, 255, 0), 2)
            cv2.putText(img_all, text_h, pos_h, font, 0.6, (0, 255, 0), 2)
            cv2.imshow("approx_rectangle", img_all)#显示拟合的矩形
            cv2.waitKey(0)
            # cv2.imwrite(r"picture/calibration.jpg", img)

            return f_pixel
            
    else:
        print("未识别到目标轮廓")
        return None



disp = display.Display()
cam = camera.Camera(960, 720, image.Format.FMT_GRAYSCALE)  #获取灰度图像即可
cam.skip_frames(30)
while not app.need_exit():
    img_show = cam.read()
    #img = img.lens_corr(strength=1.7)
    
    # convert maix.image.Image object to numpy.ndarray object
    #t = time.ticks_ms()
    img = image.image2cv(img_show, ensure_bgr=False, copy=False)
    
    #print("time: ", time.ticks_ms() - t)

    # 用cv2处理图像
    test_distance = 385
    
    f_val = get_focal_length(img, test_distance)
    
    if f_val:
        print("-" * 30)
        print(f"校准成功！")
        print(f"计算得出的像素焦距 f_pixel: {f_val:.2f}")
        print(f"请将此数值保存到你的主程序配置文件中。")
        print("-" * 30)
    
    # show by maix.display
    #img_show = image.cv2image(img_cv2, bgr=True, copy=False)
    disp.show(img_show)

        
        

