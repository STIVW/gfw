import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 将像素值归一化到[0,1]范围内，并将图片转换为浮点数类型
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 定义对比损失函数
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name="contrastive_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # 将y_true转换为浮点数类型
        y_true = tf.cast(y_true, y_pred.dtype)

        # 计算欧氏距离
        distance = tf.norm(y_pred[:, 0, :] - y_pred[:, 1, :], axis=-1)

        # 计算损失
        loss = y_true * tf.square(distance) + (1 - y_true) * tf.square(tf.maximum(self.margin - distance, 0))

        return tf.reduce_mean(loss)

# 定义模型
def create_model(input_shape):
    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 卷积层
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 展平层
    x = layers.Flatten()(x)

    # 全连接层
    x = layers.Dense(128, activation="relu")(x)

    # 嵌入层
    embedding = layers.Dense(64, activation=None, name="embedding")(x)

    # 模型
    model = keras.Model(inputs=inputs, outputs=embedding)

    return model

# 创建模型实例
model = create_model(input_shape=x_train.shape[1:])

# 编译模型
model.compile(optimizer="adam", loss=ContrastiveLoss(margin=1.0), metrics=["accuracy"])

# 自定义DataGenerator生成minibatch的样本图片和标签进行训练
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size : (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        return [batch_x[:, 0], batch_x[:, 1]], batch_y



# 训练模型
batch_size = 256
epochs = 10
generator = DataGenerator(x_train, y_train, batch_size)
model.fit(generator, epochs=epochs)

# 可视化嵌入向量
embedding_model = keras.Model(inputs=model.inputs, outputs=model.get_layer("embedding").output)
embeddings = embedding_model.predict(x_test)
tsne = TSNE(n_components=2, random_state=42)
x_embedded = tsne.fit_transform(embeddings)

# 绘制可视化图
plt.figure(figsize=(10, 10))
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y_test, cmap="jet")
plt.colorbar()
plt.show()

# 构建分类模型
classifier = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(64,)),
    layers.Dense(10, activation="softmax")
])

# 编译分类模型
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 训练分类模型
history = classifier.fit(embeddings, y_test, batch_size=128, epochs=20, validation_split=0.2)

# 测试准确率
test_loss, test_acc = classifier.evaluate(embeddings, y_test)
print("Test accuracy:", test_acc)
