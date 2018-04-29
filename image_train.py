# -*- coding: utf-8 -*-

"""
モデルを作成しトレーニングを行う
"""
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.preprocessing import image
from keras.callbacks import EarlyStopping

# パラメータ定義
activation = 'relu'
optimizer = 'Adam'
nb_epoch = 30
batch_size = 16

# 訓練・検証用データの格納ディレクトリ
train_path = './train'
valid_path = './valid'

# 学習結果を格納するディレクトリを作成する
if not os.path.exists('./result'):
    os.mkdir('./result')
result_dir = './result'

# 画像を分類するクラスを定義する
classes = ['buri', 'katsuo', 'kuromaguro', 'maaji', 'NG']
nb_classes = len (classes)

def image_train():
    print('Start model building')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(150, 150, 3)))
    model.add(Activation(activation))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(activation))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # モデルをファイルに保存する
    model_json = model.to_json()
    with open(os.path.join(result_dir, 'model.json'), 'w') as f:
        f.write(model_json)

    print('start training.')

    # 訓練データの作成
    train_datagen = ImageDataGenerator(
        #zca_whitening= True, # ZCA白色化を適用します
        rotation_range=40, # 画像をランダムに回転する回転範囲
        width_shift_range=0.2, # ランダムに水平シフトする範囲
        height_shift_range=0.2, # ランダムに垂直シフトする範囲
        shear_range=0.2, # シアー強度（反時計回りのシアー角度（ラジアン））
        zoom_range=0.2, # ランダムにズームする範囲．浮動小数点数が与えられた場合
        horizontal_flip=True, # 水平方向に入力をランダムに反転します
        rescale=1.0 / 255) # Noneか0ならば，適用しない．それ以外であれば，(他の変換を行う前に) 与えられた値をデータに積算する

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    # 検証用データの生成定義
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        valid_path,
        target_size=(150, 150),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    steps_per_epoch = train_generator.samples
    validation_steps = validation_generator.samples

    print('steps_per_epoch is set to %s' % steps_per_epoch)
    print('validation_steps is set to %s' % validation_steps)

    # 訓練の早期終了
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    # 訓練開始
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=[es_cb],
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  epochs=nb_epoch)

    print('Training Complete.')

    model.save_weights(os.path.join(result_dir, 'weight.h5'))
    # plot_model(model, to_file=os.path.join(result_dir, filename_prefix + '_model.png'), show_shapes=True)

if __name__ == '__main__':
    image_train()
