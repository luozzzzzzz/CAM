from maix import image, display, app, time, camera
import cv2
import numpy as np

file_path1="/maixapp/share/icon/raw_image.png"
file_path2="/maixapp/share/icon/binary_image.png"
disp = display.Display()
cam = camera.Camera(960,720, image.Format.FMT_GRAYSCALE)  #获取灰度图像即可
 
cam.skip_frames(40)


img = cam.read()
#img = img.lens_corr(strength=1.7)
# convert maix.image.Image object to numpy.ndarray object
#t = time.ticks_ms()
img = image.image2cv(img, ensure_bgr=False, copy=False)

#print("time: ", time.ticks_ms() - t)

# 用cv2处理图像
#blurred = cv2.GaussianBlur(img,(5,5),0)
edged = cv2.Canny(img, 50, 150)
cnts,_=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
if len(cnts) > 0:
    # 假设最大的轮廓就是 A4 纸
    peri = cv2.arcLength(cnts[0], True)
    approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)  #多边形逼近

    x, y, w, h_pixel = cv2.boundingRect(approx)  # 矩形左上角坐标与矩形宽度和高度
    
    # 提取出内容
    x1 = max(x - 5, 0)
    y1 = max(y - 5, 0)
    x2 = min(x + w + 5, img.shape[1])
    y2 = min(y + h_pixel + 5, img.shape[0])
    target = img[y1:y2, x1:x2]

    ret, binary = cv2.threshold(target, 100, 255, cv2.THRESH_BINARY)
else:
    """
    # 绘制识别结果用于预览确认
    cv2.rectangle(img, (x, y), (x + w, y + h_pixel), (0, 255, 0), 2)  #绿色框
    cv2.putText(img, f"H_pix: {h_pixel}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    """
    binary=img
#edge2=cv2.Canny(binary)

# show by maix.display

# 受光照影响大，框选不稳定
binary=cv2.resize(binary,(480,640))
img_show = image.cv2image(binary, bgr=True, copy=False)
disp.show(img_show)

cv2.imwrite(file_path1,img)
cv2.imwrite(file_path2,binary)


while not app.need_exit():
    time.sleep(5)
    print("ok")
    
    

