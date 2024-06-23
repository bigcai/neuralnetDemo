package org.bigcai.entity;

import org.bigcai.NeuralUnit;
import org.bigcai.function.impl.MeanSquareError;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import static org.bigcai.NeuralUnit.SCALE;

/**
 * 反向传播算法
 *
 *  【理解】 反向传播算法本身就是一个求解 “当前权重矩阵对损失函数的敏感度” 的过程。
 *
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
public class BackpropagationAlgorithm extends MeanSquareError{

    /**
     * 学习率
     */
    BigDecimal learnRate = new BigDecimal("0.2");

    /**
     * 进行一轮权重更新（会用到梯度向量和学习率）
     *
     * @param multiLayer
     */
    public void updateMultiNeuralNetworkWeight(MultiLayerNeuralNetwork multiLayer) {
        // 既然是反向传播，那当然是从最后一层神经元开始计算了
        for (int i = multiLayer.singleLayerNeuralNetworkList.size() - 1; i >= 0; i--) {

            SingleLayerNeuralNetwork singleLayer = multiLayer.singleLayerNeuralNetworkList.get(i);
            for (int neuralUnitIndex = 0; neuralUnitIndex < singleLayer.neuralLayer.size(); neuralUnitIndex++) {
                NeuralUnit neuralUnit = singleLayer.neuralLayer.get(neuralUnitIndex);
                // 获取当前神经元的误差项（损失值对权重向量积的敏感度）
                BigDecimal currentNeuralUnitError = singleLayer.layerError.get(neuralUnitIndex);
                // 计算梯度向量，即损失值对每一个权重的敏感度
                List<BigDecimal> gradientVector = computeNeuralUnitGradientVector(neuralUnit, currentNeuralUnitError);
                // 更新神经元权重
                updateWeight(neuralUnit, gradientVector, this.learnRate);
            }
        }
    }

    /**
     * 计算制定神经元的梯度向量，即 损失函数对神经元每个权重的敏感度。
     *
     * @param neuralUnit
     * @param currentNeuralUnitError
     * @return
     */
    private List<BigDecimal> computeNeuralUnitGradientVector(NeuralUnit neuralUnit, BigDecimal currentNeuralUnitError) {
        List<BigDecimal> neuralUnitGradientVector = new ArrayList<>();
        for (BigDecimal inputFeature : neuralUnit.getInputFeatureVector()) {
            BigDecimal neuralUnitGradient = inputFeature.multiply(currentNeuralUnitError);
            neuralUnitGradientVector.add(neuralUnitGradient);
        }
        return neuralUnitGradientVector;
    }


    /**
     * 这个公式需要通过泰勒展开证明得到，通过对损失函数进行泰勒展开得到一条递推公式，最终推导出一条令损失值递降的办法，
     * 就是令学习率大于 0 并且 还要乘上损失函数对权重的偏导（也就是所谓的梯度向量）。
     *
     * @param neuralUnit
     * @param gradientVector
     * @param learnRate
     */
    private void updateWeight(NeuralUnit neuralUnit, List<BigDecimal> gradientVector, BigDecimal learnRate) {
        for (int i = 0; i < neuralUnit.getWeightVector().size(); i++) {
            BigDecimal oldWeight = neuralUnit.getWeightVector().get(i);
            BigDecimal newWeight = oldWeight.subtract(learnRate.multiply(gradientVector.get(i)));
            newWeight = newWeight.setScale(SCALE, RoundingMode.HALF_UP);
            neuralUnit.getWeightVector().set(i, newWeight);
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
