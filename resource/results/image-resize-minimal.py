from PIL import Image
import glob
import os

if not os.path.exists("resized"):
    os.makedirs("resized")

# 处理当前文件夹下所有jpg
for img_path in glob.glob("*.png"):
    try:
        img = Image.open(img_path)
        # 直接调整为1280x720
        resized = img.resize((1200, 900), Image.Resampling.LANCZOS)
        # 保存为 原文件名_resized.jpg
        name = img_path.rsplit('.', 1)[0]
        resized.save(os.path.join("resized", f"{name}_resized.png"), quality=100)
        print(f"Processed: {img_path}")
    except Exception as e:
        print(f"Error: {img_path} - {e}")
