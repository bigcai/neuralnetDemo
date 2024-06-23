package org.bigcai;

import org.bigcai.entity.BackpropagationAlgorithm;
import org.bigcai.entity.MultiLayerNeuralNetwork;
import org.bigcai.entity.SingleLayerNeuralNetwork;
import org.bigcai.vo.DataForTrainVo;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

public class DataSetTrainTest {
    public static void main(String[] args) {
        // 利用反向传播算法计算多层神经网络中每个神经元的误差值
        BackpropagationAlgorithm backpropagationAlgorithm = new BackpropagationAlgorithm();
        MultiLayerNeuralNetwork multiLayerNeuralNetwork = buildMultiLayerNeuralNetwork();

        List<DataForTrainVo> dataForTrainVoList = new ArrayList<>();
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("0.5"), new BigDecimal("0.1"), new BigDecimal("0.2")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("0.3"), new BigDecimal("0.1"), new BigDecimal("0.2")));

        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("1.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("1.5"), new BigDecimal("2.1"), new BigDecimal("0.6")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("1.5"), new BigDecimal("2.1"), new BigDecimal("0.6")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("2.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("2.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("2.5"), new BigDecimal("1.1"), new BigDecimal("0.6")));

        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("0.8"), new BigDecimal("0.8"), new BigDecimal("0.9")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("0.9"), new BigDecimal("0.9"), new BigDecimal("0.9")));
        dataForTrainVoList.add(new DataForTrainVo(new BigDecimal("0.7"), new BigDecimal("0.7"), new BigDecimal("0.9")));

        for (int i = 0; i < 10000; i++) {
            trainModel(dataForTrainVoList, multiLayerNeuralNetwork, backpropagationAlgorithm);
        }

        for (int i = 0; i < dataForTrainVoList.size(); i++) {
            DataForTrainVo data = dataForTrainVoList.get(i);
            List<BigDecimal> estimateValue = multiLayerNeuralNetwork.compute(data.features);
            System.out.println("第 【" + i + "】 个 训练数据的预测");
            System.out.println(" 预估值：" + estimateValue);
            System.out.println(" 实际值：" + data.actualValue + "  loss value: " + data.actualValue.subtract(estimateValue.get(0)));
        }

        List<DataForTrainVo> dataForTestVoList = new ArrayList<>();
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("0.8"), new BigDecimal("0.1"), new BigDecimal("0.2")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("0.5"), new BigDecimal("0.1"), new BigDecimal("0.2")));

        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("2.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("3.5"), new BigDecimal("2.1"), new BigDecimal("0.6")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("4.5"), new BigDecimal("2.1"), new BigDecimal("0.6")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("5.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("7.5"), new BigDecimal("0.1"), new BigDecimal("0.6")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("4.5"), new BigDecimal("1.1"), new BigDecimal("0.6")));

        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("0.3"), new BigDecimal("0.3"), new BigDecimal("0.9")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("0.4"), new BigDecimal("0.4"), new BigDecimal("0.9")));
        dataForTestVoList.add(new DataForTrainVo(new BigDecimal("0.6"), new BigDecimal("0.6"), new BigDecimal("0.9")));

        System.out.println("===============测试数据集==============");
        for (int i = 0; i < dataForTestVoList.size(); i++) {
            DataForTrainVo data = dataForTestVoList.get(i);
            List<BigDecimal> estimateValue = multiLayerNeuralNetwork.compute(data.features);
            System.out.println("第 【" + i + "】 次 模型的预测测试");
            System.out.println("【测试】 预估值：" + estimateValue);
            System.out.println("【测试】  实际值：" + data.actualValue + "  loss value: " + data.actualValue.subtract(estimateValue.get(0)));
        }

    }

    private static void trainModel(List<DataForTrainVo> dataForTrainVoList, MultiLayerNeuralNetwork multiLayerNeuralNetwork, BackpropagationAlgorithm backpropagationAlgorithm) {
        for (DataForTrainVo data: dataForTrainVoList) {
            trainData(data, multiLayerNeuralNetwork, backpropagationAlgorithm);
        }
    }

    private static void trainData(DataForTrainVo dataForTrainVo,
                                  MultiLayerNeuralNetwork multiLayerNeuralNetwork, BackpropagationAlgorithm backpropagationAlgorithm) {
        List<BigDecimal> estimateVal = multiLayerNeuralNetwork.compute(dataForTrainVo.features);
        //System.out.println("BBBBBBBB训练前预估值：" + activationVal);

        // 更新神经元的误差项，打印误差项
        backpropagationAlgorithm.computeError(multiLayerNeuralNetwork, estimateVal, dataForTrainVo.actualValue);
        // 更新前的权重
        backpropagationAlgorithm.updateMultiNeuralNetworkWeight(multiLayerNeuralNetwork);

        estimateVal = multiLayerNeuralNetwork.compute(dataForTrainVo.features);
        //System.out.println("EEEEEEEEE训练后预估值：" + activationVal);
    }

    private static List<BigDecimal> computeError(List<BigDecimal> activationVal, BigDecimal actualValue) {
        List<BigDecimal> errorSource = new ArrayList<>();
        for (BigDecimal estimatedValue : activationVal) {
            errorSource.add(estimatedValue.subtract(actualValue).setScale(SCALE, RoundingMode.HALF_UP));
        }
        //System.out.println("loss value: " + errorSource);
        return errorSource;
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
