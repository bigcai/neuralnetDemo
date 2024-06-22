package org.bigcai;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

/**
 * 反向传播算法
 * <p>
 * ## 简单理解为 3 个步骤：
 * 1. 计算出当前的误差项： 利用 【上一个的带权重误差项】x 和 【当前输出（激活值）的损失函数（可能带有惩罚项）】k,计算【出当前误差项】y ---  这是最重要的一部，也是最难理解的一部，他包含了数学偏导的知识
 * 2. 计算权重梯度： 利用【当前神经元权重的系数（也就是当前神经元输入值，也就是当前神经元激活值）】k 和 【当前神经元的误差项】x，计算【当前神经元的权重梯度】y。  --- 这是准备结束的一部
 * 3. 计算新权重：  利用 【当前神经元各位旧的权重】k、【学习率】k、【权重梯度】x，计算当前神经元各位旧权重的【新权重】y。 --- 结束的一部
 * <p>
 * ## 算法作用：
 * 1.计算【整个网络】的误差项矩阵（神经元矩阵相对应）。
 * 2.计算【每个神经元】的新权重向量。
 */
public class BackpropagationAlgorithm {

    /**
     * 进行一轮权重更新
     * @param multiLayer
     */
    public void updateMultiNeuralNetworkWeight(MultiLayerNeuralNetwork multiLayer) {
        // 既然是反向传播，那当然是从最后一层神经元开始计算了
        for (int i = multiLayer.singleLayerNeuralNetworkList.size() - 1; i >= 0; i--) {
            SingleLayerNeuralNetwork singleLayer = multiLayer.singleLayerNeuralNetworkList.get(i);
            for (int neuralUnitIndex = 0; neuralUnitIndex < singleLayer.layer.size(); neuralUnitIndex++) {
                NeuralUnit neuralUnit = singleLayer.layer.get(neuralUnitIndex);
                BigDecimal currentNeuralUnitError = singleLayer.errorOfErrorSourceLayer.get(neuralUnitIndex);
                List<BigDecimal> gradientVector = computeNeuralUnitGradientVector(neuralUnit, currentNeuralUnitError);
                // 更新神经元权重
                updateWeight(neuralUnit, gradientVector, multiLayer.learnRate);
            }

        }
    }

    private void updateWeight(NeuralUnit neuralUnit, List<BigDecimal> gradientVector, BigDecimal learnRate) {
        for (int i = 0; i < neuralUnit.weightVector.size(); i++) {
            BigDecimal oldWeight = neuralUnit.weightVector.get(i);
            BigDecimal newWeight = oldWeight.subtract(learnRate.multiply(gradientVector.get(i)));
            newWeight = newWeight.setScale(SCALE, RoundingMode.HALF_UP);
            neuralUnit.weightVector.set(i, newWeight);
        }
    }

    // 计算制定神经元的梯度，即 损失函数对神经元权重的敏感度。
    private List<BigDecimal> computeNeuralUnitGradientVector(NeuralUnit neuralUnit, BigDecimal currentNeuralUnitError) {
        List<BigDecimal> neuralUnitGradientVector = new ArrayList<>();
        for (BigDecimal inputFeature : neuralUnit.inputFeatureVector) {
            BigDecimal neuralUnitGradient = inputFeature.multiply(currentNeuralUnitError);
            neuralUnitGradientVector.add(neuralUnitGradient);
        }
        return neuralUnitGradientVector;
    }

    /**
     * 计算出该【当前神经元的误差项】： 利用【误差来源层的误差向量】和当前神经元的【输出激活值】，计算出该【当前神经元的误差项】， 这个值会被用来计算【当前神经元的梯度向量】
     * <p>
     * 【当前神经元的梯度向量】，表示了当前神经元的权重变化对误差的敏感度。
     */
    private BigDecimal computeCurrentNeuralUnitError(int neuralUnitIndex, NeuralUnit currentNeuralUnit, SingleLayerNeuralNetwork errorSourceLayer) {
        BigDecimal currentNeuralUnitError = new BigDecimal(0);
        BigDecimal activationVal = currentNeuralUnit.activationFunctionValue;
        BigDecimal partialDerivativeZ = computePartialDerivativeZ(activationVal);
        List<BigDecimal> weightOfErrorSourceLayerByIndex = readWeightByIndex(neuralUnitIndex, errorSourceLayer);
        List<BigDecimal> errorOfErrorSourceLayer = errorSourceLayer.errorOfErrorSourceLayer;
        for (int i = 0; i < errorOfErrorSourceLayer.size(); i++) {
            BigDecimal weight = weightOfErrorSourceLayerByIndex.get(i);
            // 每一个误差来源层的输入权重 * 权重所在的来源层神经的误差项 * 当前神经元的激活值
            BigDecimal item = weight.multiply(errorOfErrorSourceLayer.get(i)).multiply(partialDerivativeZ);
            item = item.setScale(SCALE, RoundingMode.HALF_UP);
            currentNeuralUnitError = currentNeuralUnitError.add(item);
        }
        return currentNeuralUnitError;
    }

    // TODO 根据损失函数来决定计算规则
    private BigDecimal computePartialDerivativeZ(BigDecimal activationVal) {
        return activationVal.multiply(new BigDecimal(1).subtract(activationVal));
    }

    private List<BigDecimal> readWeightByIndex(int neuralUnitIndex, SingleLayerNeuralNetwork errorSourceLayer) {
        List<BigDecimal> weightOfErrorSourceLayerByIndex = new ArrayList<>();
        if (errorSourceLayer.layer.size() == 0) {
            for (int i = 0; i < errorSourceLayer.errorOfErrorSourceLayer.size(); i++) {
                // 如果没有神经元，说明这个是输出层，输出层只有误差向量，那么他的权重都为 1
                weightOfErrorSourceLayerByIndex.add(new BigDecimal(1));
            }
        } else {
            for (int i = 0; i < errorSourceLayer.layer.size(); i++) {
                // 如果有神经元，说明这个是隐藏层，隐藏层的权重列表要根据 neuralUnitIndex 从每个神经元中获取
                NeuralUnit oneNeuralUnit = errorSourceLayer.layer.get(i);
                weightOfErrorSourceLayerByIndex.add(oneNeuralUnit.weightVector.get(neuralUnitIndex));
            }
        }
        return weightOfErrorSourceLayerByIndex;
    }

    public void computeMultiNeuralNetworkError(MultiLayerNeuralNetwork multiLayer, List<BigDecimal> errorSource) {
        //  虚拟层， 只有一个神经元，误差是

        SingleLayerNeuralNetwork lastLay = new SingleLayerNeuralNetwork(null);
        lastLay.errorOfErrorSourceLayer = errorSource;
        // 既然是反向传播，那当然是从最后一层神经元开始计算了
        for (int i = multiLayer.singleLayerNeuralNetworkList.size() - 1; i >= 0; i--) {
            SingleLayerNeuralNetwork singleLayer = multiLayer.singleLayerNeuralNetworkList.get(i);
            for (int neuralUnitIndex = 0; neuralUnitIndex < singleLayer.layer.size(); neuralUnitIndex++) {
                NeuralUnit currentNeuralUnit = singleLayer.layer.get(neuralUnitIndex);
                BigDecimal currentNeuralUnitError = computeCurrentNeuralUnitError(neuralUnitIndex, currentNeuralUnit, lastLay);
                singleLayer.errorOfErrorSourceLayer.add(currentNeuralUnitError);
            }
            lastLay = singleLayer;
        }
    }
}

/**
 * 附加题
 * 题目：输入层输入的值是激活函数计算后的值，我理解应该求损失函数对 a 的敏感度，对 a 做偏导才对，为什么是对 z 做偏导？
 * 答案： 这是因为反向传播的目的是通过计算梯度来更新权重和偏置。而在神经网络中，权重和偏置直接影响的是 z （也就是加权后的值，而不是对 z 激活后的激活值 a）
 * 我们需要计算的是损失函数 error 对这些参数 w、b 的敏感度, 就需要通过对 z 求偏导即可。
 * <p>
 * 这里对 偏导的理解，可以看成是 求 自变量因变量之间的系数，这个系数也反映了映射的敏感度
 * <p>
 * 附加题
 * 题目：为什么反向传播微调神经元的权重的量，跟变化率和学习率有关系，数学证明过程，以及如何理解学习率
 */
