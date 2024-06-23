package org.bigcai.function;

import org.bigcai.entity.MultiLayerNeuralNetwork;

import java.math.BigDecimal;
import java.util.List;

/**
 * ErrorFunction 损失函数
 *
 * 通过损失函数的继承关系可以看出，只有反向传播过程才需要用到这个度量误差，进行调参的工具。
 * 正向传播时不需要，反向传播才需要，这就是 1986年，Rumelhar和Hinton等人提出了反向传播（Backpropagation，BP）算法的伟大之处。
 *
 */
public abstract class ErrorFunction {

    /**
     * 不同的损失函数计算的损失值公式不一样（不过这个原函数，我们不会用到，只用到他的偏导数）
     *
     * @param outputList
     * @param actualList
     * @return
     */
    public abstract BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList);

    /**
     * 用于计算输出层的损失向量，也就是反向传播算法需要用到的首个误差项
     *
     * 正向传播时不需要，反向传播才需要，这也是 1986年，Rumelhar和Hinton等人提出了反向传播（Backpropagation，BP）算法的伟大之处。
     *
     * @param activationVal
     * @param actualValue
     * @return
     */
    public abstract List<BigDecimal> computeError(MultiLayerNeuralNetwork multiLayerNeuralNetwork,
                                                  List<BigDecimal> activationVal, BigDecimal actualValue);
}
