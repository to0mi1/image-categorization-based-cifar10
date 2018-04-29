# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

result_dir = './result'

# 画像を分類するクラスの定義
classes = ['buri', 'katsuo', 'kuromaguro', 'maaji', 'NG']

def predict(img_path):
    # モデルのロード
    model_json = open(os.path.join(result_dir, 'model.json')).read()
    model = model_from_json(model_json)
    # 重みのロード
    model.load_weights(os.path.join(result_dir, 'weight.h5'))

    # 画像のロード
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # 画像の分類
    pred = model.predict(x)[0]

    # 結果出力
    top = 2
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    for x in result:
        print(x)

if __name__ == '__main__':

    predict(sys.argv[1])
