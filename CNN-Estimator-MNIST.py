"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  '''输入层

  layers用于为二维图像数据创建卷积层和合并层的模块中的方法预期输入张量具有默认的形状 。此行为可以使用参数进行更改; 定义如下：
  [batch_size, image_height, image_width, channels]data_format

  batch_size。在训练期间执行梯度下降时要使用的示例子集的大小。
  image_height。示例图像的高度。
  image_width。示例图像的宽度。
  channels。示例图像中的颜色通道数量。对于彩色图像，通道数量是3（红色，绿色，蓝色）。对于单色图像，只有1个通道（黑色）。
  image_height。示例图像的高度。
  data_format。一个字符串，其中一个channels_last（默认）或channels_first。 channels_last对应于具有形状的输入 (batch, ..., channels)
  而channels_first对应于具有形状的输入(batch, channels, ...)。
  这里，我们的MNIST数据集由单色的28x28像素图像组成，因此我们输入图层的所需形状为。[batch_size, 28, 28, 1]

  要将我们的输入特征映射（features）转换为这种形状，我们可以执行以下reshape操作：'''
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  '''卷积层＃1
  在我们的第一个卷积层中，我们希望将32个5x5滤波器应用于输入层，并具有ReLU激活功能。我们可以使用模块中的conv2d()方法 layers创建
  此层，如下所示：
  该inputs参数指定了输入张量，其中必须有形状 。在这里，我们将第一个卷积层连接到具有形状的卷积层。[batch_size, image_height, 
  image_width, channels]input_layer[batch_size, 28, 28, 1]

  注意：相反，它会接受传递参数时 的形状 。 conv2d()[batch_size, channels, image_height, image_width]data_format=channels_first
  所述filters参数指定的过滤器，以应用（这里，32）的数量，并且 kernel_size指定了作为过滤器的尺寸（此处，）。[height, width][5, 5]

  提示：如果过滤器高度和宽度具有相同的值，则可以为kernel_size-eg 指定单个整数kernel_size=5。

  该padding参数指定了两个枚举值（不区分大小写）中的一个：valid（缺省值）或same。要指定输出张量应具有与输入张量相同的高度和宽度
  值，我们padding=same在此处设置，它指示TensorFlow将0值添加到输入张量的边缘以保持28的高度和宽度。（没有填充，a在28x28张量上进行
  5x5卷积将产生24x24张量，因为有24x24个位置从28x28网格中提取5x5瓦片。）

  该activation参数指定了激活功能应用到卷积的输出。在这里，我们指定ReLU激活 tf.nn.relu。

  我们产生的输出张量conv2d()具有以下形状 ：与输入相同的高度和宽度尺寸，但现在有32个通道保持每个滤波器的输出。[batch_size, 28, 28, 32]'''
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  '''汇集层＃1
  接下来，我们将第一个池化层连接到刚刚创建的卷积层。我们可以使用该max_pooling2d()方法layers构造一个层，该层使用2x2过滤器和2的步
  幅执行最大池：
  再次，inputs指定输入张量，形状为 。这里，我们的输入张量是第一个卷积层的输出，其形状为。[batch_size, image_height, image_width, channels]
  conv1[batch_size, 28, 28, 32]

  注意：与之相反conv2d()，max_pooling2d()将接受传递参数时 的形状。[batch_size, channels, image_height, image_width]data_format=channels_first
  该pool_size参数指定了最大池过滤器的大小 （这里）。如果两个维度具有相同的值，则可以改为指定单个整数（例如， ）。[height, width]
  [2, 2]pool_size=2

  该strides参数指定步幅的大小。在这里，我们设置了一个步幅为2，表示过滤器提取的子区域在高度和宽度维度上应分开2个像素（对于2x2过滤
  器，这意味着所提取的区域都不会重叠）。如果要为高度和宽度设置不同的步幅值，可以改为指定元组或列表（例如，stride=[3, 6]）。

  我们的max_pooling2d()（pool1）产生的输出张量具有以下形状 ：2x2滤波器将高度和宽度各减少50％。[batch_size, 14, 14, 32]'''
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  '''卷积层＃2和池层＃2
  我们可以使用conv2d()和max_pooling2d()以前一样将第二个卷积和池化层连接到我们的CNN 。对于卷积层＃2，我们使用ReLU激活配置64个5x5
  过滤器，对于池＃2，我们使用相同的规范作为池＃1（2x2最大池过滤器，步幅为2）：
  注意，卷积层＃2将我们的第一个池化层（pool1）的输出张量作为输入，并将张量conv2作为输出。conv2 具有与（由于）相同的高度和宽度的形
  状，以及用于64个滤波器的64个通道。[batch_size, 14, 14, 64]pool1padding="same"

  池＃2 conv2作为输入，产生pool2输出。pool2 有形状（高度和宽度减少50％）。[batch_size, 7, 7, 64]conv2'''
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  '''密集层
  接下来，我们想要向我们的CNN添加一个密集层（具有1,024个神经元和ReLU激活），以对卷积/池化层提取的特征进行分类。然而，在我们连接图
  层之前，我们会将要素贴图（pool2）展平为形状，以便我们的张量只有两个维度：[batch_size, features]
  在reshape()上面的操作中，-1表示batch_size 将根据输入数据中的示例数量动态计算维度。每个示例都有7（pool2高度）* 7（pool2宽度）*
  64（pool2通道）功能，因此我们希望features维度的值为7 * 7 * 64（总共3136）。输出张量pool2_flat具有形状 。[batch_size, 3136]'''
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  '''现在，我们可以使用该dense()方法layers连接我们的密集层，如下所示：
  该inputs参数指定输入张：我们的扁平化特征图， pool2_flat。该units参数指定在致密层（1024）神经元的数目。该activation参数采用激
  活函数; 再次，我们将使用tf.nn.relu添加ReLU激活。'''
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  '''为了帮助改进模型的结果，我们还使用以下dropout方法将丢失正则化应用于密集层layers：
  再次，inputs指定输入张量，它是来自密集层（dense）的输出张量。

  该rate参数指定了辍学率; 在这里，我们使用0.4，这意味着40％的元素将在训练期间随机丢弃。

  该training参数采用布尔值来指定模型当前是否在训练模式下运行; 如果training是，则仅执行丢失 True。在这里，我们检查mode传递给我们
  的模型函数 cnn_model_fn是否是TRAIN模式。

  我们的输出张量dropout已经形成。[batch_size, 1024]'''
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  '''Logits Layer
  我们的神经网络中的最后一层是logits层，它将返回我们预测的原始值。我们创建了一个包含10个神经元的密集层（每个目标类0-9一个），线性
  激活（默认）：
  CNN的最终输出张量logits已经形成 。[batch_size, 10]'''
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      '''生成预测
      我们模型的logits层将我们的预测作为原始值在一 维张量中返回。让我们将这些原始值转换为我们的模型函数可以返回的两种不同格式：[batch_size, 10]

      每个例子的预测类：0-9的数字。
      的概率为每个实施例的每个可能的目标类：该示例是0的概率，是1，是2等
      对于给定的示例，我们的预测类是具有最高原始值的logits tensor的相应行中的元素。我们可以使用tf.argmax 函数找到这个元素的索引：
      该input参数指定从中提取最大值，这里的张量logits。该axis参数指定的轴input 张量沿找到最大的价值。在这里，我们希望在索引为1的维度
      上找到最大值，这对应于我们的预测（回想一下我们的logits张量有形状）。[batch_size, 10]'''
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      '''我们可以通过使用softmax激活来从logits层导出概率tf.nn.softmax
      注意：我们使用name参数来明确命名此操作 softmax_tensor，因此我们稍后可以引用它。（我们将在“设置记录挂钩”中设置 softmax值
      的记录）。'''
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  '''配置培训操作
  在上一节中，我们将CNN的损失定义为logits图层和我们的标签的softmax交叉熵。让我们配置我们的模型以在训练期间优化此损失值。我们将使
  用0.001的学习率和 随机梯度下降 作为优化算法：'''
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  '''加载培训和测试数据
  我们存储和训练标记（从0-9的相应值对每个图像）作为训练特征数据（的手绘数字55000个图像的原始像素值）numpy的阵列 中train_data和
  train_labels分别。同样，我们将评估特征数据（10,000个图像）和评估标签分别存储在eval_data 和中eval_labels。'''
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  '''创建估算器
  接下来，让Estimator我们为我们的模型创建一个（用于执行高级模型训练，评估和推理的TensorFlow类）。将以下代码添加到main()：
  该model_fn参数指定的模型函数用于培训，评估，和预测; 我们cnn_model_fn在“构建CNN MNIST分类器”中创建了 它。该 model_dir参数指定
  了模型数据（检查站）将被保存的目录（这里，我们指定的临时目录/tmp/mnist_convnet_model，但随时更改为您选择的另一个目录）'''
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  '''设置记录钩
  由于CNN可能需要一段时间才能进行训练，我们可以设置一些记录，以便我们可以在训练期间跟踪进度。我们可以使用TensorFlow 
  tf.train.SessionRunHook创建一个 tf.train.LoggingTensorHook 记录来自CNN的softmax层的概率值。
  我们存储了我们想要登录的张量的词典tensors_to_log。每个键都是我们选择的标签，将打印在日志输出中，相应的标签是TensorTensorFlow图
  中的a的名称。在这里，我们 probabilities可以找到softmax_tensor，我们在生成概率时提供softmax操作的名称cnn_model_fn。
  接下来，我们创建LoggingTensorHook，传递tensors_to_log给 tensors参数。我们设置every_n_iter=50，它指定在每50个训练步骤之后应
  该记录概率。'''
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  '''训练模型
  现在我们已经准备好训练我们的模型，我们可以通过创建train_input_fn 和调用train()来完成mnist_classifier。
  在numpy_input_fn通话中，我们将训练特征数据和标签分别传递给 x（作为dict）和y。我们设置batch_size的100（这意味着该模型将上的100
  个例子minibatches培养在每一个步骤）。 num_epochs=None表示模型将训练直到达到指定的步数。我们还设置shuffle=True了改组培训数据。
  在train通话中，我们设置steps=20000 （这意味着模型将训练总共20,000步）。我们将我们传递 logging_hook给hooks论证，以便在训练期
  间触发。'''
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  '''评估模型
  培训完成后，我们要评估我们的模型，以确定其在MNIST测试集上的准确性。我们调用该evaluate方法，该方法评估我们在eval_metric_ops参数
  中指定的指标model_fn。
  为了创建eval_input_fn，我们进行设置num_epochs=1，以便模型在一个时期的数据上评估指标并返回结果。我们还设置 shuffle=False按顺序
  迭代数据。'''
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()