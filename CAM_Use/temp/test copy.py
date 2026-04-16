import cv2


img=cv2.imread(r"CAM_Use\temp\square_3.png",0)

_, binary_img = cv2.threshold(
            img, 150, 255, 
            cv2.THRESH_BINARY
        )
cv2.imshow("0",img)
cv2.imshow("1",binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
