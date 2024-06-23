package org.bigcai;

import org.bigcai.function.impl.MeanSquareError;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class MeanSquareErrorTest {
    public static void main(String[] args) {
        MeanSquareError meanSquareError = new MeanSquareError();
        List<BigDecimal> output = new ArrayList<>();
        // 预测值
        output.add(new BigDecimal(0.7122));
        List<BigDecimal> actual = new ArrayList<>();
        // 真实值
        actual.add(new BigDecimal(1));
        // 损失函数： 均方差1235
        BigDecimal res = meanSquareError.diff(output, actual);
        System.out.println(res);
    }
}
