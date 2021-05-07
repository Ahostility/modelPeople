from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def boston_hous():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    print(f"train: {(len(x_train), len(y_train))}")
    # print(f"test: {(x_test, y_test)}")
    print(x_train.shape[1])
    print(x_train)
    # Среднее значение
    mean = x_train.mean(axis=0)
    # Стандартное отклонение
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    # arr = np.array([[40342651,99178419, 33391162,
    #        25683275, 21518188, 89434613,
    #        91036058, 24758524, 85646254,
    #        34843254, 71818909, 43190599,32920239]])
    # print(x_train)
    # print(arr)
    # mean = arr.mean(axis=0)
    # std = arr.std(axis=0)
    # arr-=mean
    # print(f"maen: {mean}, arr - mean: {arr}")
    # arr/=std
    # print(f"maen: {std}, arr / std: {arr}")
    # print(f"maen: {arr.mean(axis=0)}")
    # x_test -= mean
    # x_test /= std
    print(f"learning: {x_train[1]}")
    # print(f"learning: {x_test[1]}")
    print(f"Answer: {y_train[1]}")
    # print(f"Answer: {y_test[1]}")
    # print(2,x_test)
    # for i in range(x_train.shape[1]):
    f = [0,1]
    print(x_train[1][:])
    # plt.scatter(x_train[1][:],y_train[1],s=10,c='red')
    # plt.plot(f)
    # plt.grid(True)
    # plt.show()

#создание сети
    # print(f"shapeX: {x_train.shape[0]}, shapeY: {x_train.shape[1]}")
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(units=128,
    #                                 activation='relu',
    #                                 input_shape=(x_train.shape[1],)))
    # model.add(tf.keras.layers.Dense(1))
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Обучение
#     model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

if __name__ == '__main__':
    boston_hous()