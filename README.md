# image-categorization-based-cifar10

Keras の example にある cifar-10 を使った画像分類を参考に画像分類をする。  
[https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)

## スクリプト概要

スクリプトは訓練・予測のをそれぞれ用意する。

- image_train.py
  - 訓練スクリプト  
  Conv 層 2つ Pooling 層 1 つを 1 セットとし、2 セット用意し、後方に全結合層を 1 つ用意する。
- pred.py
  - 予測スクリプト  
  image_train.py で作成したモデル及び重みを使い画像分類を予測する。

## スクリプト詳細

### 訓練スクリプト

`train`, `valid` ディレクトリをスクリプトと同じ階層のディレクトリに作成し、訓練・検証用の画像データを格納する。  
格納する際には、スクリプト内 `classes` に定義するクラス名と同じ名前でサブディレクトリを作成し、それぞれ画像データを格納する。

訓練・検証データは Keras の [ImageDataGenerator](https://keras.io/ja/preprocessing/image/) を利用し、 画像データのバッチを生成する。

訓練は Early Stopping コールバックを利用し早期終了する。

トレーニング後はスクリプトと同じ階層に `result` ディレクトリを作成し、モデルの `model.json` 重みの `weight.h5` ファイルを出力する。

トレーニングは `python image_train.py` で実行する。

```
Using TensorFlow backend.
Start model building
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 150, 150, 32)      896
_________________________________________________________________
activation_1 (Activation)    (None, 150, 150, 32)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 150, 150, 32)      9248
_________________________________________________________________
activation_2 (Activation)    (None, 150, 150, 32)      0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 75, 75, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 75, 75, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 75, 75, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 75, 75, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 75, 75, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 75, 75, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 37, 37, 64)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 37, 37, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 87616)             0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               44859904
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 2565
_________________________________________________________________
activation_6 (Activation)    (None, 5)                 0
=================================================================
Total params: 44,928,037
Trainable params: 44,928,037
Non-trainable params: 0
_________________________________________________________________
start training.
Found 250 images belonging to 5 classes.
Found 70 images belonging to 5 classes.
steps_per_epoch is set to 250
validation_steps is set to 70
Epoch 1/30
2018-04-29 15:49:30.900222: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2018-04-29 15:49:31.255379: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7715
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 5.01GiB
2018-04-29 15:49:31.263042: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
250/250 [==============================] - 37s 149ms/step - loss: 0.5314 - acc: 0.7981 - val_loss: 0.4983 - val_acc: 0.8000
Epoch 2/30
250/250 [==============================] - 33s 131ms/step - loss: 0.4792 - acc: 0.8016 - val_loss: 0.5530 - val_acc: 0.7886
Epoch 3/30
250/250 [==============================] - 33s 132ms/step - loss: 0.4354 - acc: 0.8136 - val_loss: 0.5203 - val_acc: 0.8029
Epoch 00003: early stopping
Training Complete.
```

### 予測スクリプト

image_train.py にて出力された `./result/model.json` と `./result/weight.h5` からモデルと重みをロードし、予測結果を出力する。

引数に予測対象の画像のパスを指定し、`python pred.py target.jpg` で実行する。

```
Using TensorFlow backend.
2018-04-29 16:02:38.650470: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2018-04-29 16:02:38.903227: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7715
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 5.01GiB
2018-04-29 16:02:38.910873: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
('katsuo', 0.29811853)
('maaji', 0.25752506)
```
