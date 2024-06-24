package org.bigcai.util;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class MathUtil {

    /**
     * 算数精度
     */
    public static final int SCALE = 6;

    public static BigDecimal subtract(BigDecimal subtrahend, BigDecimal subtract) {
        return subtrahend.subtract(subtract)
                .setScale(SCALE, RoundingMode.HALF_UP);
    }

    public static BigDecimal multiply(BigDecimal multiplied, BigDecimal multipliers) {
        return multiplied.multiply(multipliers)
                .setScale(SCALE, RoundingMode.HALF_UP);
    }

    public static BigDecimal add(BigDecimal additive, BigDecimal addition) {
        return additive.add(addition)
                .setScale(SCALE, RoundingMode.HALF_UP);
    }

    public static BigDecimal divide(BigDecimal dividend, BigDecimal divisor) {
        return dividend.divide(divisor, SCALE, RoundingMode.HALF_UP);
    }
}


