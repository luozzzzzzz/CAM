from maix import camera, display, image, nn, app
import cv2
classifier = nn.Classifier(model="/root/models/maixhub/263962/model_263962.mud")
file_path="/maixapp/share/icon/square_3_017_267x267.png"


img = image.load(file_path)
if img is None:
    raise Exception(f"load image failed")
input_w = classifier.input_width()
input_h = classifier.input_height()
img_resized = img.resize(input_w, input_h)
print(img_resized.format())

res = classifier.classify(img_resized)
max_idx, max_prob = res[0]
msg = f"{max_prob:5.2f}: {classifier.labels[max_idx]}"
print(msg)
print(res)
