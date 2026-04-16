from maix import image, display, app, uart, touchscreen, time, camera
import cv2
import numpy as np
import tools
from struct import pack

def measure_target(image, f_pixel, h_outcontour = 28.3,w_outcontour=21.0,real_side_limit=(10, 16)):
    """
    测量图像中正方形的距离 D 和边长 x
    :param image_path: 图片路径
    :param f_pixel: 相机的像素焦距 (通过预先测量获得)
    :param real_side_limit: 题目要求的边长范围 (cm)
    :return: D (距离), x (边长), classes(类别)
    """
    distance=-1
    x=-1
    classes="000"

    cnts,img_gray = tools.get_conTours(image)
    
    if len(cnts) > 0:
        #一、根据外轮廓计算距离 D
        # 假设最大的轮廓就是 A4 纸
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)

        if len(approx) == 4:
            #print(f"外轮廓近似多边形顶点数：{len(approx)}")

            # 对A4纸进行透视矫正


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
                #print(f"---目标为正方形!---，顶点数：{len(approx)}")
                classes="squ"
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

                return distance, x,classes
            
            elif len(approx) == 3:
                #print(f"---目标为三角形！，顶点数：{len(approx)}---")
                classes="tri"
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

                return distance, x, classes
            elif len(approx) > 4:

                #print(f"---目标为圆形！---")
                classes="cir"
                D_pixel, radius, pos_circle,pos_text = tools.caculate_circle_x(cnts[-1])
                #print(f"拟合圆的像素直径 D_pixel: {D_pixel:.2f}")

                # 2. 计算实际直径
                Dia = D_pixel * h_outcontour / h_pixel_ex
                #print(f"目标圆的实际直径 Dia: {Dia:.2f} cm")

                # 3. 绘制识别结果用于预览确认
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                img_all = img.copy()     
                
                text = f"Dia: {Dia:.2f}cm"
            
                # 画圆和标注
                cv2.circle(img_all, pos_circle , int(radius), (0, 255, 0), 2) # 画出识别到的圆
                cv2.putText(img_all, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                return distance, Dia,classes
            else:
                #内部为未知形状
                classes="ukn"
                return distance,x,classes
        else:
            #print("未识别到目标轮廓:A4")
            distance=-1
            x=-1
            return distance,x,classes
    else:
        print("没有检测到任何边框")
        classes="bad"
        return distance,x,classes

# 用于在图像中绘制退出按键
def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]

def get_back_btn_img(width):
    ret_width = int(width * 0.1)
    img_back = image.load("/maixapp/share/icon/ret.png")
    w, h = (ret_width, img_back.height() * ret_width // img_back.width())
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    img_back = img_back.resize(w, h)
    return img_back

def answer(serial,img_show):
    D_set=[]
    x_set=[]
    classes="ini"
    for i in range(0,20):
        # convert maix.image.Image object to numpy.ndarray object
        #t = time.ticks_ms()
        img = image.image2cv(img_show, ensure_bgr=False, copy=False)
        # 用opencv处理图像
        D_raw,x_raw,classes =measure_target(img, f_pixel = 2705.25, h_outcontour = 28.3,w_outcontour=20.1, real_side_limit=(10, 16))
        D_set.append(D_raw)
        x_set.append(x_raw)
        #print(f"距离 D: {D:.2f} cm, 边长/直径 x: {x:.2f} cm")
    # 返回测量结果：
    def trim_mean(x_set,p):
        x_set=np.sort(x_set)
        cut=int(p*len(x_set))
        return np.mean(x_set[cut:len(x_set)-cut])
    D=trim_mean(D_set,0.1)   #去除头尾10%求平均
    x=trim_mean(x_set,0.1)
    """
    #bytes_content = b'\xAA'
    bytes_content = pack("<f", D)    # 小端编码,4个字节
    bytes_content += b'\x2C'
    bytes_content += pack("<f", x)    # 小端编码
    bytes_content += pack("<i", classs)    # 小端编码
    bytes_content += b'\x0D'
    bytes_content += b'\x0A'
    serial.write(bytes_content)
    #print(bytes_content, type(bytes_content))
    """
    s=f"{D:.3f},{x:.3f},{classes}\r\n"
    serial.write_str(s)

def main(disp):
    cam = camera.Camera(960, 720)  
    cam.skip_frames(30)
    ts=touchscreen.TouchScreen()

    img_back = get_back_btn_img(cam.width())
    back_rect = [0, 0, img_back.width(), img_back.height()]
    back_rect_disp = image.resize_map_pos(cam.width(), cam.height(), disp.width(), disp.height(), image.Fit.FIT_CONTAIN, back_rect[0], back_rect[1], back_rect[2], back_rect[3])

    serial_dev = uart.UART("/dev/ttyS0", 115200)

    while not app.need_exit():
        img_show = cam.read()
        #img = img.lens_corr(strength=1.7)  #矫正图像畸变

        rx_data=serial_dev.read().decode()  #以ASCII编码恢复字符串
        if rx_data:
            serial_dev.write_str("hello\r\n")
        if rx_data=="tri":
            answer(serial_dev,img_show)
        elif rx_data=="cir":
            answer(serial_dev,img_show)
        elif rx_data=="squ":
            answer(serial_dev,img_show)

        # show by maix.display
        #img_show = image.cv2image(img_cv2, bgr=True, copy=False)
        # 现在直接展示原始图像，后面考虑绘制框选图像
        img_show.draw_image(0, 0, img_back)
        disp.show(img_show)
        x, y, preesed = ts.read()
        if is_in_button(x, y, back_rect_disp):
            app.set_exit_flag(True)

disp = display.Display()  
try:
    main(disp)
except Exception:
    import traceback
    msg = traceback.format_exc()
    img = image.Image(960, 720)
    img.draw_string(0, 0, msg, image.COLOR_WHITE)
    disp.show(img)
    while not app.need_exit():
        time.sleep_ms(100)

