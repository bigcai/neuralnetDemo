package org.bigcai.function;

import java.math.BigDecimal;

/**
 * ActivationFunction 激活函数
 */
public abstract class ActivationFunction {
    /**
     * 激活函数计算公式
     *
     * @param sum
     * @return
     */
    public abstract BigDecimal activationFunction(BigDecimal sum);

    /**
     * 用于协助计算损失值对权重的敏感度，计算出激活函数对权重向量的偏导
     *
     * @param activationVal
     * @return
     */
    public abstract BigDecimal computePartialDerivativeZOfActivation(BigDecimal activationVal);
}
