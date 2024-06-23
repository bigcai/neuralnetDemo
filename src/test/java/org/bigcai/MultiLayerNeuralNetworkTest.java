package org.bigcai;

import org.bigcai.entity.MultiLayerNeuralNetwork;
import org.bigcai.entity.SingleLayerNeuralNetwork;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class MultiLayerNeuralNetworkTest {

    public static void main(String[] args) {
        MultiLayerNeuralNetwork multiLayerNeuralNetwork = buildMultiLayerNeuralNetwork();

        List<BigDecimal> features = new ArrayList<>();
        features.add(new BigDecimal(0.5));
        features.add(new BigDecimal(0.1));

        List<BigDecimal> activationVal = multiLayerNeuralNetwork.compute(features);
        System.out.println(activationVal);
    }

    private static MultiLayerNeuralNetwork buildMultiLayerNeuralNetwork() {
        /** 第一层神经网络*/
        List<BigDecimal> weights1 = new ArrayList<>();
        weights1.add(new BigDecimal(0.1));
        weights1.add(new BigDecimal(0.2));
        NeuralUnit fooNeural1 = new NeuralUnit(weights1, new BigDecimal(0.1));

        List<BigDecimal> weights2 = new ArrayList<>();
        weights2.add(new BigDecimal(0.3));
        weights2.add(new BigDecimal(0.4));
        NeuralUnit fooNeural2 = new NeuralUnit(weights2, new BigDecimal(0.2));

        List<NeuralUnit> layer = new ArrayList<>();
        layer.add(fooNeural1);
        layer.add(fooNeural2);
        // 构建单层神经网络完成
        SingleLayerNeuralNetwork singleLayerNeuralNetwork1 = new SingleLayerNeuralNetwork(layer);

        /** 第二层神经网络*/
        List<BigDecimal> weights3 = new ArrayList<>();
        weights3.add(new BigDecimal(0.5));
        weights3.add(new BigDecimal(0.6));
        NeuralUnit fooNeural3 = new NeuralUnit(weights3, new BigDecimal(0.3));

        List<NeuralUnit> layer2 = new ArrayList<>();
        layer2.add(fooNeural3);
        // 构建单层神经网络完成
        SingleLayerNeuralNetwork singleLayerNeuralNetwork2 = new SingleLayerNeuralNetwork(layer2);

        /**  组装 2 个单层网络*/
        List<SingleLayerNeuralNetwork> multiLayer = new ArrayList<>();
        multiLayer.add(singleLayerNeuralNetwork1);
        multiLayer.add(singleLayerNeuralNetwork2);
        MultiLayerNeuralNetwork multiLayerNeuralNetwork = new MultiLayerNeuralNetwork(multiLayer);
        return multiLayerNeuralNetwork;
    }

}
