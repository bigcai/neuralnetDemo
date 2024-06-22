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
        }
        return squareSum.divide(new BigDecimal(outputList.size()), SCALE, RoundingMode.HALF_UP);
    }
}
