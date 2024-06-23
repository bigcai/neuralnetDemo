package org.bigcai.function.impl;

import org.bigcai.entity.MultiLayerNeuralNetwork;
import org.bigcai.function.ErrorFunction;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

public class MeanSquareError extends ErrorFunction {

    /**
     * 均方差公式，用于计算特征列表的均方差（方差的一半，累加后求平均数）
     *
     * 这个公式用不到，但他的偏导数则会被频繁用到。
     *
     * @param outputList
     * @param actualList
     * @return
     */
    @Override
    public BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList) {
        BigDecimal squareSum = new BigDecimal(0);
        for (int i = 0; i < outputList.size(); i++) {
            BigDecimal sub = outputList.get(i).subtract(actualList.get(i));
            BigDecimal squareSub = BigDecimal.valueOf(Math.pow(sub.doubleValue(), 2.0D));
            // 为了便于计算和简化反向传播的数学推导，有时我们会引入 1/2 的系数（0.5 就等于 1/2），这样在求导时可以消除平方项前的常数因子。
            // 比如 x平方求导后为 2X， 为了消除这个 2， 我们会故意只用 x 的一半（1/2 * x）来求导，得到的导数边是 x
            squareSub = squareSub.setScale(SCALE, RoundingMode.HALF_UP).multiply(new BigDecimal("0.5"));
            // 累加
            squareSum = squareSum.add(squareSub);

        }
        return squareSum.divide(new BigDecimal(outputList.size()), SCALE, RoundingMode.HALF_UP);
    }

    /**
     * 计算输出层的损失函数向量
     *
     * @param activationVal
     * @param actualValue
     * @return
     */
    @Override
    public List<BigDecimal> computeError(MultiLayerNeuralNetwork multiLayerNeuralNetwork,
                                         List<BigDecimal> activationVal, BigDecimal actualValue) {
        List<BigDecimal> errorSource = new ArrayList<>();
        for (BigDecimal estimatedValue : activationVal) {
            errorSource.add(estimatedValue.subtract(actualValue)
                    .setScale(SCALE, RoundingMode.HALF_UP));
        }
        multiLayerNeuralNetwork.computeError(errorSource);
        return errorSource;
    }

}
