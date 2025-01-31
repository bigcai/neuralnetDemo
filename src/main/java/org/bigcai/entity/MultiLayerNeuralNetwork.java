package org.bigcai.entity;

import org.bigcai.Layer;
import org.bigcai.NeuralUnit;
import org.bigcai.entity.helper.ErrorComputer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Minsky说过单层神经网络无法解决异或问题。但是当增加一个计算层以后，两层神经网络不仅可以解决异或问题，而且具有非常好的非线性分类效果。不过两层神经网络的计算是一个问题，没有一个较好的解法。
 * <p>
 * 　　1986年，Rumelhar和 Hinton 等人提出了反向传播（Backpropagation，BP）算法，解决了两层神经网络所需要的复杂计算量问题，从而带动了业界使用两层神经网络研究的热潮。目前，大量的教授神经网络的教材，都是重点介绍两层（带一个隐藏层）神经网络的内容。
 * <p>
 * 　　这时候的Hinton还很年轻，30年以后，正是他重新定义了神经网络，带来了神经网络复苏的又一春。
 * <p>
 *    Hinton 是 加拿大多伦多大学的 Geoffery Hinton教授
 * <p>
 *    -------------------------------
 * <p>
 *    神经网络仍然存在若干的问题：尽管使用了BP算法，一次神经网络的训练仍然耗时太久，而且困扰训练优化的一个问题就是局部最优解问题，这使得神经网络的优化较为困难。同时，隐藏层的节点数需要调参，这使得使用不太方便，工程和研究人员对此多有抱怨。
 * <p>
 * 　　90年代中期，由Vapnik等人发明的SVM（Support Vector Machines，支持向量机）算法诞生，很快就在若干个方面体现出了对比神经网络的优势：无需调参；高效；全局最优解。基于以上种种理由，SVM迅速打败了神经网络算法成为主流。
 * <p>
 *    引用 <a href="https://www.cnblogs.com/subconscious/p/5058741.html#second">神经网络相关资料</a>
 */
public class MultiLayerNeuralNetwork extends Layer implements ErrorComputer {

    List<SingleLayerNeuralNetwork> singleLayerNeuralNetworkList = new ArrayList<>();

    List<BigDecimal> errorSource = new ArrayList<>();

    public MultiLayerNeuralNetwork(List<SingleLayerNeuralNetwork> initSingleLayerNeuralNetworkList) {
        for (int i = 0; i < initSingleLayerNeuralNetworkList.size(); i++) {
            SingleLayerNeuralNetwork singleLayer = initSingleLayerNeuralNetworkList.get(i);

            if(i+1 < initSingleLayerNeuralNetworkList.size()) {
                // 如果不是最后一层网络，需要标记下一层网络的指针，用于后续反向传播用到
                singleLayer.setNextLayer(initSingleLayerNeuralNetworkList.get(i+1));
                for (NeuralUnit neuralUnit : singleLayer.getNeuralLayer()) {
                    neuralUnit.setNextLayer(singleLayer.getNextLayer());
                }
            }
            singleLayerNeuralNetworkList.add(singleLayer);
        }
    }

    public List<BigDecimal> compute(List<BigDecimal> features) {
        readyCompute(features);

        // 迭代每个网络层的激活值向量
        List<BigDecimal> currentInputFeature = inputFeatureCache;
        List<BigDecimal> activationValues = null;
        for (SingleLayerNeuralNetwork singleLayerInNetwork : singleLayerNeuralNetworkList ) {
            // 理论上是并发的
            activationValues = singleLayerInNetwork.compute(currentInputFeature);
            currentInputFeature = activationValues;
        }

        // cache result
        outputActivationCache = activationValues;
        return activationValues;
    }

    @Override
    public void computeError(List<BigDecimal> errorSource) {
        this.setErrorSource(errorSource);

        //  虚拟层， 只有一个神经元，误差向量是【输出层的损失函数向量】
        SingleLayerNeuralNetwork lastLayer = new SingleLayerNeuralNetwork(null);
        lastLayer.layerError = errorSource;

        // 既然是反向传播，那当然是从最后一层神经元开始计算了
        for (int i = this.singleLayerNeuralNetworkList.size() - 1; i >= 0; i--) {
            SingleLayerNeuralNetwork singleLayer = this.singleLayerNeuralNetworkList.get(i);
            singleLayer.computeError(lastLayer.layerError);
            lastLayer = singleLayer;
        }
    }

    public List<SingleLayerNeuralNetwork> getSingleLayerNeuralNetworkList() {
        return singleLayerNeuralNetworkList;
    }

    public void setSingleLayerNeuralNetworkList(List<SingleLayerNeuralNetwork> singleLayerNeuralNetworkList) {
        this.singleLayerNeuralNetworkList = singleLayerNeuralNetworkList;
    }

    public List<BigDecimal> getErrorSource() {
        return errorSource;
    }

    public void setErrorSource(List<BigDecimal> errorSource) {
        this.errorSource = errorSource;
    }


}
