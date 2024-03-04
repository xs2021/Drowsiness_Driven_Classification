import os
from PIL import Image
import numpy as np

# 1万张图耗时：几分钟

# folder_path = r'../../cus_data/Drowsiness_Driven_Dataset/data_unpartition/'
folder_path = r'./data/Drowsiness_Driven_Dataset/data_unpartition/'
total_pixels = 0
# 如果是RGB图像，需要三个通道的均值和标准差
sum_normalized_pixel_values = np.zeros(3)
sum_squared_diff = 0.0

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path)
                image_array = np.array(image)
                normalized_image_array = image_array / 255.0
                total_pixels += normalized_image_array.size
                sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))
            except Exception as e:
                print("异常:", e)

# 沿着通道计算均值
mean = sum_normalized_pixel_values / total_pixels

sum_squared_diff_channels = np.zeros(3)
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path)
                image_array = np.array(image)
                normalized_image_array = image_array / 255.0
                diff = (normalized_image_array - mean) ** 2
                sum_squared_diff_channels += np.sum(diff, axis=(0, 1))
            except Exception as e:
                print("异常:", e)

# 沿着通道计算方差
var = sum_squared_diff_channels / total_pixels

print("Mean:", mean)
print("Var:", var)
