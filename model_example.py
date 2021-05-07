import tensorflow as tf
import sys
import numpy as np


# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input      = x
#         self.weights1   = np.random.rand(self.input.shape[1],4)
#         self.weights2   = np.random.rand(4,1)
#         self.weights3   = np.random.rand(5,1)
#         self.y          = y
#         self.output     = np.zeros(y.shape)

class Network:
    def __init__(self,x):
        self.input = x
        #2x-1
    def mystery(self):
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(
                units=1,
                # units=128,
                input_shape=[1])]
        )
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam',
                      loss='mse',
                      # metrics=['mae']
                      )
        xs = np.array([float(i) for i in range(-1,5)],dtype=float)
        ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)
        """Посторение таблицы зависимостей sea born"""
        model.fit(xs,ys,epochs=500,batch_size=1)
        res = model.predict([self.input])#Работа обученной модели на данных пользователя
        return res


if __name__ == '__main__':
    print([float(i) for i in range(-1,5)])
    obj = Network(int(input()))
    print(obj.mystery())