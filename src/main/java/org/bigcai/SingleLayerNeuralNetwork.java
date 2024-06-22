package org.bigcai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * 单层神经网络
 *
 * 1958年，计算科学家Rosenblatt提出了由两个神经元组成的神经网络。
 * 他给它起了一个名字--“感知器”（Perceptron）（有的文献翻译成“感知机”，下文统一用“感知器”来指代）。
 * 感知器是当时首个可以学习的人工神经网络。Rosenblatt现场演示了其学习识别简单图像的过程，在当时的社会引起了轰动。
 *
 * Minsky在1969年出版了一本叫《Perceptron》的书，里面用详细的数学证明了感知器的弱点，
 * 尤其是感知器对XOR（异或）这样的简单分类任务都无法解决。
 * Minsky认为，如果将计算层增加到两层，计算量则过大，而且没有有效的学习算法。所以，他认为研究更深层的网络是没有价值的。
 *
 * 引用 https://www.cnblogs.com/subconscious/p/5058741.html#second
 */
public class SingleLayerNeuralNetwork {

    /**
     * 单层神经元列表
     */
    List<NeuralUnit> layer = new ArrayList<>();

    /**
     * 输入值缓存
     */
    List<BigDecimal> inputFeatureCache = new ArrayList<>();

    /**
     * 输出值缓存
     */
    List<BigDecimal> outputActivationCache = new ArrayList<>();

    public SingleLayerNeuralNetwork(List<NeuralUnit> initNeuralUnitList) {
        layer.addAll(initNeuralUnitList);
    }

    /**
     * 输入特征列表，特征列表要求与神经元的权重数目配对。
     * 每个神经元的权重数都是一样的。
     * 输出新的抽象特征（数量由神经元个数决定，跟输入特征数量可能不一样）
     *
     * @param features
     * @return
     */
    public List<BigDecimal> compute(List<BigDecimal> features) {
        inputFeatureCache.clear();
        for (BigDecimal feature : features) {
            inputFeatureCache.add(feature);
        }
        outputActivationCache.clear();
        for (NeuralUnit neuralUnitInLayer : layer ) {
            // 理论上是并发的
            BigDecimal activationVal = neuralUnitInLayer.compute(inputFeatureCache);
            outputActivationCache.add(activationVal);
        }
        return outputActivationCache;
    }
}

/**
 *
 * 让我们通过一个具体的例子来证明单层神经网络即使使用了非线性激活函数也无法解决某些非线性问题。
 * 我们来考虑一个简单的问题：XOR（异或）问题。
 *
 * XOR 问题是一个经典的非线性问题，其输入包含两个二进制数字，输出是这两个数字的异或结果。
 *
 * 我们知道 XOR 问题的真值表如下：
 *
 * | Input (X1, X2) | Output (Y) |
 * | -------------- | ---------- |
 * | (0, 0)         | 0          |
 * | (0, 1)         | 1          |
 * | (1, 0)         | 1          |
 * | (1, 1)         | 0          |
 *
 * 我们尝试使用单层神经网络来解决这个问题。我们定义输入特征为 (X1, X2)，输出为 Y。
 *
 * 单层神经网络的输出可以表示为：
 *
 * Y = σ(w1*X1 + w2*X2 + b)
 *
 * 其中，σ 是非线性激活函数，w1、w2 是权重，b 是偏置。
 *
 * 我们可以尝试使用 sigmoid 或者其他非线性激活函数，但是无论如何调整权重和偏置，
 * 单层神经网络无法找到一个边界将 (0, 0) 和 (1, 1) 分类到一个类别，同时将 (0, 1) 和 (1, 0) 分类到另一个类别。
 * 这是因为 XOR 问题的数据分布不是线性可分的。
 *
 * 因此，尽管单层神经网络使用了非线性激活函数，但由于其结构的限制，它无法解决 XOR 问题这样的非线性问题。
 * 要解决 XOR 问题，我们需要使用多层神经网络，例如多层感知器（MLP），它具有更高的表示能力，能够学习复杂的非线性关系。
 */
