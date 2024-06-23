package org.bigcai;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

public class BackpropagationAlgorithmTest {
    public static void main(String[] args) {
        // 利用反向传播算法计算多层神经网络中每个神经元的误差值
        BackpropagationAlgorithm backpropagationAlgorithm = new BackpropagationAlgorithm();

        MultiLayerNeuralNetwork multiLayerNeuralNetwork = buildMultiLayerNeuralNetwork();
        for (int i = 0; i < 10000; i++) {
            trainModel(multiLayerNeuralNetwork, backpropagationAlgorithm);
        }


    }

    private static void trainModel(MultiLayerNeuralNetwork multiLayerNeuralNetwork, BackpropagationAlgorithm backpropagationAlgorithm) {

        List<BigDecimal> features = new ArrayList<>();
        features.add(new BigDecimal(0.5));
        features.add(new BigDecimal(0.1));
        BigDecimal actualValue = new BigDecimal(0.64);

        List<BigDecimal> activationVal = multiLayerNeuralNetwork.compute(features);
        System.out.println("BBBBBBBB训练前预估值：" + activationVal);

        /*// 打印更新前的权重
        System.out.println("=========打印更新前的权重=========");
        for (SingleLayerNeuralNetwork singleLayerNeuralNetwork: multiLayerNeuralNetwork.singleLayerNeuralNetworkList) {
            for (NeuralUnit neuralUnit: singleLayerNeuralNetwork.layer) {
                System.out.println(neuralUnit.weightVector);
            }
            System.out.println("==================");
        }*/
        // 更新神经元的误差项，打印误差项
        List<BigDecimal> errorSource = backpropagationAlgorithm.computeError(activationVal, actualValue);
        backpropagationAlgorithm.computeMultiNeuralNetworkError(multiLayerNeuralNetwork, errorSource);
        /*System.out.println("======打印误差项===============");

        for (SingleLayerNeuralNetwork singleLayerNeuralNetwork: multiLayerNeuralNetwork.singleLayerNeuralNetworkList) {
            System.out.println(singleLayerNeuralNetwork.errorOfErrorSourceLayer);
        }
        System.out.println("========================");*/

        // 更新前的权重
        backpropagationAlgorithm.updateMultiNeuralNetworkWeight(multiLayerNeuralNetwork);
        // 更新后的权重
        /*System.out.println("========更新后的权重==========");
        for (SingleLayerNeuralNetwork singleLayerNeuralNetwork: multiLayerNeuralNetwork.singleLayerNeuralNetworkList) {
            for (NeuralUnit neuralUnit: singleLayerNeuralNetwork.layer) {
                System.out.println(neuralUnit.weightVector);
            }
            System.out.println("==================");
        }*/

        activationVal = multiLayerNeuralNetwork.compute(features);
        System.out.println("EEEEEEEEE训练后预估值：" + activationVal);

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
