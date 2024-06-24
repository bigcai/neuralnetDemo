package org.bigcai.util;

import java.math.BigDecimal;

public class AdvancedMathUtil {
    /**
     * 泰勒级数的项数量，用于控制精度，默认为 20
     */
    private static final int TERMS = 20;
    // 使用泰勒展开式计算 e^x
    public static BigDecimal exp(BigDecimal x) {
        BigDecimal result = new BigDecimal("1.0");  // 第0项
        BigDecimal term = new BigDecimal("1.0");   // 当前项的值

        for (int i = 1; i < TERMS; i++) {
            term = MathUtil.multiply(term, MathUtil.divide(x, new BigDecimal(i))) ;   // 计算第i项的值
            result = MathUtil.add(result, term);  // 累加到结果中
        }
        return result;
    }


}
