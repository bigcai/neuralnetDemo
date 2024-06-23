package org.bigcai.function;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public abstract class ErrorFunction {

    /**
     * 不同的损失函数计算的损失值公式不一样（不过这个原函数，我们不会用到，只用到他的偏导数）
     * @param outputList
     * @param actualList
     * @return
     */
    public abstract BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList);

    /**
     * 用于计算损失值对权重的敏感度
     * @param activationVal
     * @return
     */
    public abstract BigDecimal computePartialDerivativeZ(BigDecimal activationVal);

    /**
     * 用于计算输出层的损失向量，也就是反向传播算法需要用到的首个误差项
     *
     * @param activationVal
     * @param actualValue
     * @return
     */
    public abstract List<BigDecimal> computeError(List<BigDecimal> activationVal, BigDecimal actualValue);
}
