import cv2
import numpy as np
import os
import random

def augment_squares_random_range(
    input_files,
    output_folder="augmented_squares",
    num_per_square=30,
    min_size=50,
    max_size=300
):
    """
    处理正方形图片：二值化 -> 在[min, max]区间随机选分辨率 -> 批量保存
    """
    # 1. 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 2. 遍历处理每个输入文件
    for file_idx, file_path in enumerate(input_files, 1):
        # 读取图片（灰度模式）
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  无法读取文件：{file_path}，已跳过")
            continue

        # 3. 二值化处理
        _, binary_img = cv2.threshold(
            img, 150, 255, 
            cv2.THRESH_BINARY
        )

        print(f"正在处理第{file_idx}个正方形，生成{num_per_square}张随机分辨率图片...")

        # 4. 生成30张图片
        for i in range(num_per_square):
            # 【核心逻辑】在 min_size 到 max_size 之间随机抽取一个整数作为边长
            target_size = random.randint(min_size, max_size)
            
            # 随机选择插值方法
            interpolation = random.choice([
                cv2.INTER_AREA, 
                cv2.INTER_CUBIC, 
                cv2.INTER_LINEAR
            ])
            
            # 执行Resize
            resized_img = cv2.resize(
                binary_img, 
                (target_size, target_size), 
                interpolation=interpolation
            )

            # 5. 保存文件（文件名中标注具体分辨率，方便查看）
            save_name = f"square_{file_idx}_{str(i+1).zfill(3)}_{target_size}x{target_size}.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, resized_img)

    print(f"\n🎉 全部处理完成！结果已保存至文件夹：{output_folder}")

# ------------------- 执行入口 -------------------
if __name__ == "__main__":
    # 请确保文件名与上一步提取的文件名一致
    INPUT_FILES = [
        "data\square_3.png",
        "data\square_4.png",
        "data\square_5.png",
        "data\square_6.png"
    ]

    try:
        augment_squares_random_range(
            input_files=INPUT_FILES,
            num_per_square=30,
            min_size=50,   # 最小分辨率（包含）
            max_size=300   # 最大分辨率（包含）
        )
    except Exception as e:
        print(f"执行出错：{e}")