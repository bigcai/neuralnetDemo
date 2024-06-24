package org.bigcai.function.impl;

import org.bigcai.entity.MultiLayerNeuralNetwork;
import org.bigcai.function.ErrorFunction;
import org.bigcai.util.MathUtil;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class MeanSquareError extends ErrorFunction {

    /**
     * 均方差公式，用于计算特征列表的均方差（方差的一半，累加后求平均数）
     * <p>
     * (涉及计算 - 减法、乘法、除法) 这个公式用不到，但他的偏导数则会被频繁用到。
     *
     * @param outputList
     * @param actualList
     * @return
     */
    @Override
    public BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList) {
        BigDecimal halfSquareSum = new BigDecimal(0);
        for (int i = 0; i < outputList.size(); i++) {
            BigDecimal sub = MathUtil.subtract(actualList.get(i), outputList.get(i));
            BigDecimal squareSub = MathUtil.multiply(sub, sub);
            // 为了便于计算和简化反向传播的数学推导，有时我们会引入 1/2 的系数（0.5 就等于 1/2），这样在求导时可以消除平方项前的常数因子。
            // 比如 x平方求导后为 2X， 为了消除这个 2， 我们会故意只用 x 的一半（1/2 * x）来求导，得到的导数边是 x
            BigDecimal halfSquareSub = MathUtil.multiply(new BigDecimal("0.5"), squareSub);
            // 累加
            halfSquareSum = MathUtil.add(halfSquareSum, halfSquareSub);

        }
        BigDecimal n = new BigDecimal(outputList.size());
        return MathUtil.divide(halfSquareSum, n);
    }

    /**
     * （涉及计算 - 减法）计算输出层的损失函数向量
     *
     * @param estimatedValueList
     * @param actualValue
     * @return
     */
    @Override
    public List<BigDecimal> computeError(MultiLayerNeuralNetwork multiLayerNeuralNetwork,
                                         List<BigDecimal> estimatedValueList, BigDecimal actualValue) {
        List<BigDecimal> errorSource = new ArrayList<>();
        for (BigDecimal estimatedValue : estimatedValueList) {
            errorSource.add(MathUtil.subtract(estimatedValue, actualValue));
        }
        multiLayerNeuralNetwork.computeError(errorSource);
        return errorSource;
    }

}
