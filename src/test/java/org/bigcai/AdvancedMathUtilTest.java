package org.bigcai;

import org.bigcai.util.AdvancedMathUtil;
import org.bigcai.util.MathUtil;

import java.math.BigDecimal;

public class AdvancedMathUtilTest {
    public static void main(String[] args) {
        BigDecimal x = new BigDecimal("-0.5"); // 需要计算的指数
        int terms = 20; // 控制精度的项数

        BigDecimal result = AdvancedMathUtil.exp(x);
        System.out.printf("exp(%f) ≈ %f\n", x, result);

        // 验证结果
        double expected = Math.exp(x.doubleValue());
        System.out.printf("Math.exp(%f) = %f\n", x, expected);
        System.out.printf("误差 = %e\n", Math.abs(result.doubleValue() - expected));
    }
}
