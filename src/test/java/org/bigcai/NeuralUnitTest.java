package org.bigcai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class NeuralUnitTest {
    public static void main(String[] args) {
        NeuralUnit fooNeural1 = buildNeuralUnit1();

        NeuralUnit fooNeural2 = buildNeuralUnit2();

        List<BigDecimal> features = new ArrayList<>();
        features.add(new BigDecimal(0.5));
        features.add(new BigDecimal(0.1));

        BigDecimal activationVal = fooNeural1.compute(features);
        System.out.println(activationVal);

        BigDecimal activationVal2 = fooNeural2.compute(features);
        System.out.println(activationVal2);

    }

    private static NeuralUnit buildNeuralUnit2() {
        List<BigDecimal> weights2 = new ArrayList<>();
        weights2.add(new BigDecimal(0.3));
        weights2.add(new BigDecimal(0.4));
        NeuralUnit fooNeural2 = new NeuralUnit(weights2, new BigDecimal(0.2));
        return fooNeural2;
    }

    private static NeuralUnit buildNeuralUnit1() {
        List<BigDecimal> weights = new ArrayList<>();
        weights.add(new BigDecimal(0.1));
        weights.add(new BigDecimal(0.2));
        NeuralUnit fooNeural = new NeuralUnit(weights, new BigDecimal(0.1));
        return fooNeural;
    }
}
