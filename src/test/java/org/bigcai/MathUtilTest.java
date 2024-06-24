package org.bigcai;

import org.bigcai.util.MathUtil;

import java.math.BigDecimal;

public class MathUtilTest {
    public static void main(String[] a) {
        System.out.println(MathUtil.add(new BigDecimal(1), new BigDecimal(1)));
        System.out.println(MathUtil.subtract(new BigDecimal(3), new BigDecimal(1)));
        System.out.println(MathUtil.multiply(new BigDecimal(2), new BigDecimal(1)));
        System.out.println(MathUtil.divide(new BigDecimal(2), new BigDecimal(1)));
    }
}
