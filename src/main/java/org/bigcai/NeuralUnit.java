package org.bigcai;

import java.math.BigDecimal;
import java.util.List;

/**
 * 神经元
 */
public class NeuralUnit {
    /**
     * 神经元的特征接收器（一个向量列表）
     */
    List<BigDecimal> inputFeatureVector;
    /**
     * 神经元的特征权重向量（一个向量列表）
     */
    List<BigDecimal> weightVector;
    /**
     * 求和
     */
    BigDecimal sum;

    /**
     * 激活函数值
     */
    BigDecimal ActivationFunctionValue;

    /**
     * 计算神经值
     */
    public BigDecimal compute(List<BigDecimal> inputFeatureVector) {
        // 求和
        // 计算激活值
        return ActivationFunctionValue;
    }


}
