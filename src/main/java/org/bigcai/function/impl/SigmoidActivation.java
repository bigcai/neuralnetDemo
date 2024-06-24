package org.bigcai.function.impl;

import org.bigcai.function.ActivationFunction;
import org.bigcai.util.ext.AdvancedMathUtil;
import org.bigcai.util.MathUtil;

import java.math.BigDecimal;

public class SigmoidActivation extends ActivationFunction {

    /**
     * (涉及计算 - 乘法、指数运算、加法、除法) 激活函数计算
     *
     * @param sum
     * @return
     */
    @Override
    public BigDecimal activationFunction(BigDecimal sum) {

        sum = MathUtil.multiply(new BigDecimal(-1), sum);
        // 低精度的计算方法
        BigDecimal exp = AdvancedMathUtil.exp(sum);
        // 高精度的计算方法
        //BigDecimal exp =  new BigDecimal(Math.exp(sum.doubleValue()));
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
