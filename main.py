import tensorflow.compat.v2 as tf
import time

def main(mixed_precision):
    # GPUメモリを必要最低限の確保に抑える
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for k in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[k], True)

    # 混合精度の使用
    if mixed_precision: 
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    # データセットの用意
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # ピクセルの値を 0~1 の間に正規化
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # モデル定義
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # コンパイル
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # 訓練(学習, 最適化)
    model.fit(train_images, train_labels, epochs=5)

    # テスト
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    mixed_precision = False

    start = time.time()
    
    main(mixed_precision)

    print("実行時間:{0}".format(time.time() - start) + "[sec]")
