package org.bigcai.function.impl;

import org.bigcai.function.ActivationFunction;
import org.bigcai.util.AdvancedMathUtil;
import org.bigcai.util.MathUtil;

import java.math.BigDecimal;
import java.math.RoundingMode;

import static org.bigcai.NeuralUnit.SCALE;

public class SigmoidActivation extends ActivationFunction {

    /**
     * (涉及计算 - 乘法、指数运算、加法、除法) 激活函数计算
     *
     * @param sum
     * @return
     */
    @Override
    public BigDecimal activationFunction(BigDecimal sum) {
        /*BigDecimal exp = AdvancedMathUtil.exp(
                MathUtil.multiply(new BigDecimal(-1), sum));*/
        sum = sum.multiply(new BigDecimal(-1))
                .setScale(SCALE, RoundingMode.HALF_UP);
        double dou = Math.exp(sum.doubleValue());
        BigDecimal exp = new BigDecimal(dou);
        return MathUtil.divide(new BigDecimal(1),
                MathUtil.add(new BigDecimal(1), exp));
    }

    /**
     * (涉及计算 - 乘法、减法) 根据损失函数来决定计算规则
     * (这个函数很妙，求偏导本质上是在求自变量的系数，他只用到因变量就可以解出这个系数)
     *
     * @param activationVal
     * @return
     */
    @Override
    public BigDecimal computePartialDerivativeZOfActivation(BigDecimal activationVal) {
        return MathUtil.multiply(activationVal,
                MathUtil.subtract(new BigDecimal(1), activationVal));
    }
}
