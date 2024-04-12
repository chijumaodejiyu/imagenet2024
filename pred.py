import os
import cv2
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

#  全局参数
input_dir = "E:\图像分类2024\input/"
output_dir = "E:\图像分类2024\output"
checkpoint = "E:\图像分类2024\checkpoint\model.h5"
class_names = ['chinese-lion', 'no-chinese-lion']

# 读取图片
def get_image_files(input_folder_path):
    files = os.listdir(input_folder_path)
    image_files = []
    for file in files:
        print(file)
        if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.JPEG', '.webp')):
            image_files.append(file)
    print(f'读取到图片{len(image_files)}张')
    return image_files

def image_operate(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# 构建网络
class Model:
    def __init__(self):
        self.model = keras.models.load_model(checkpoint)
        self.model.summary()

    def pred(self, image_files):
        for filename in image_files:
            path = str(input_dir + filename)
            x = image_operate(path)
            print(str(filename + "shape is :" + str(x.shape)))
            pred_array = self.model.predict(x)
            pred_classes, pred_belief = np.argmax(pred_array), round(np.max(pred_array), 4)
            pred_name = class_names[pred_classes]
            # 展示图片
            image = cv2.imread(str(input_dir + filename))
            text = str(pred_name) + str(pred_belief)
            cv2.putText(image, text, (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow('分类后图像', image)
            if cv2.waitKey(100) == '-1':
                cv2.destroyAllWindows()
            # 分类存储
            folder_path = output_dir + '/' + str(pred_classes)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(os.path.join(folder_path, filename), image)
        cv2.destroyAllWindows()
        print('图片分类完成')


if __name__ == "__main__":
    print("图片分类开始")
    image_files = get_image_files(input_dir)
    model = Model()
    model.pred(image_files)
