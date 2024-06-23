package org.bigcai.function.impl;

import org.bigcai.function.ActivationFunction;

import java.math.BigDecimal;
import java.math.RoundingMode;

import static org.bigcai.NeuralUnit.SCALE;

public class SigmoidActivation extends ActivationFunction {

    /**
     *  【涉及计算 - 乘法、指数运算、加法、除法】激活函数计算
     * @param sum
     * @return
     */
    @Override
    public BigDecimal activationFunction(BigDecimal sum) {
        sum = sum.multiply(new BigDecimal(-1))
                .setScale(SCALE, RoundingMode.HALF_UP);
        double dou = Math.exp(sum.doubleValue());
        BigDecimal exp;
        if (dou > 99999d) {
            exp = new BigDecimal(999d);
        } else if(dou < -0.99999d) {
            exp = new BigDecimal(-0.99999d);
        } else {
            exp = new BigDecimal(dou);
        }
        return new BigDecimal(1).divide(exp.add(new BigDecimal(1)), SCALE, RoundingMode.HALF_UP);
    }

    /**
     * 【涉及计算 - 乘法、减法】根据损失函数来决定计算规则
     * (这个函数很妙，求偏导本质上是在求自变量的系数，他只用到因变量就可以解出这个系数)
     *
     * @param activationVal
     * @return
     */
    @Override
    public BigDecimal computePartialDerivativeZOfActivation(BigDecimal activationVal) {
        return activationVal.multiply(new BigDecimal(1).subtract(activationVal));
    }
}
