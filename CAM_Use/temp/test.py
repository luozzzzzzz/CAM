import cv2


img=cv2.imread(r"augmented_squares\square_1_001_152x152.png")
"""
_, binary_img = cv2.threshold(
            img, 150, 255, 
            cv2.THRESH_BINARY
        )
cv2.imshow("0",img)
cv2.imshow("1",binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img=cv2.resize(img,(111,111))
print(img.shape)