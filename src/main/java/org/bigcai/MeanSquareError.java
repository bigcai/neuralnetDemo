package org.bigcai;

import org.bigcai.error.function.ErrorFunction;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

public class MeanSquareError extends ErrorFunction {
    @Override
    public BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList) {
        BigDecimal squareSum = new BigDecimal(0);
        for (int i = 0; i < outputList.size(); i++) {
            BigDecimal sub = outputList.get(i).subtract(actualList.get(i));
            double squareSub = Math.pow(sub.doubleValue(), 2.0D);
            squareSum = squareSum.add(new BigDecimal(squareSub));
            // 为了便于计算和简化反向传播的数学推导，有时我们会引入 1/2 的系数，这样在求导时可以消除平方项前的常数因子。
            // 比如 x平方求导后为 2X， 为了消除这个 2， 我们会故意只用 x 的一半（1/2 * x）来求导，得到的导数边是 x
            squareSum = squareSum.multiply(new BigDecimal(0.5));
        }
        return squareSum.divide(new BigDecimal(outputList.size()), SCALE, RoundingMode.HALF_UP);
    }
}
