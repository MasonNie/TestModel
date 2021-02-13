import os
from PIL import Image


path = "E:\PythonWorkSpace\TestModel\models/0.1/f_ten_db3_10_28.03.pth"
basename = os.path.basename(path)
firstname,lastname = os.path.splitext(basename)
print(os.path.dirname(path))
image = Image.open()
baby_eye_crop = (100, 130, 220, 250)
image.crop(baby_eye_crop).save(fp=os.path.dirname(opt.model) + "/" + firstname + "_crop" + ".bmp")