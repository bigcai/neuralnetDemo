package org.bigcai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class SingleLayerNeuralNetworkTest {
    public static void main(String[] args) {
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = buildSingleLayerNeuralNetwork();

        List<BigDecimal> features = new ArrayList<>();
        features.add(new BigDecimal(0.5));
        features.add(new BigDecimal(0.1));

        List<BigDecimal> activationVal = singleLayerNeuralNetwork.compute(features);
        System.out.println(activationVal);
    }

    private static SingleLayerNeuralNetwork buildSingleLayerNeuralNetwork() {
        NeuralUnit fooNeural1 = buildNeuralUnit1();
        NeuralUnit fooNeural2 = buildNeuralUnit2();

        List<NeuralUnit> layer = new ArrayList<>();
        layer.add(fooNeural1);
        layer.add(fooNeural2);
        // 构建单层神经网络完成
        SingleLayerNeuralNetwork singleLayerNeuralNetwork = new SingleLayerNeuralNetwork(layer);
        return singleLayerNeuralNetwork;
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
