package org.bigcai.function;

import java.math.BigDecimal;

/**
 * ActivationFunction 激活函数
 */
public abstract class ActivationFunction {
    public abstract BigDecimal activationFunction(BigDecimal sum);
}
