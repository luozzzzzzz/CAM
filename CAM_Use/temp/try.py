from maix import image, nn, app

model_path = "/root/models/maixhub/263962/model_263962.mud"
file_path = "/maixapp/share/icon/square_3_017_267x267.png"

# 1. 加载模型
print("Loading model...")
classifier = nn.Classifier(model=model_path,dual_buff=False)
print(f"Model labels: {classifier.labels}")

# 2. 加载图片
img = image.load(file_path)
if img is None:
    print("Failed to load image!")
else:
    # 3. 预处理
    #img = img.to_format(image.Format.FMT_RGB888)
    img_resized = img.resize(classifier.input_width(), classifier.input_height())
    
    # 4. 推理
    print("Running inference...")
    res = classifier.classify(img_resized)
    
    # 5. 解析结果
    if res:
        for idx, prob in res:
            label = classifier.labels[idx]
            print(f"Result: Index {idx} ({label}), Probability: {prob:.4f}")
    else:
        print("No result returned from classifier.")