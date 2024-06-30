package org.bigcai;

import org.bigcai.entity.SingleLayerNeuralNetwork;
import org.bigcai.entity.helper.ErrorComputer;
import org.bigcai.function.impl.SigmoidActivation;
import org.bigcai.util.MathUtil;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/**
 * 神经元
 *
 * 【理解】 从这个结构看出，神经元的本质就是激活函数（一个可以调参的非线性函数）
 *   进一步推导出，一个神经网络本身就是一个可以调参的非线性函数
 *   进一步推导出，一个神经网络训练过程本身就是在给定标签数据集下，通过对非线性函数权重调参，求最小值的过程。
 */
public class NeuralUnit extends SigmoidActivation implements ErrorComputer {

    /**
     * 当前神经元位于网络层的位置
     */
    public int layerPositionIndex;

    public String name = "no NeuralUnit name";

    /**
     * 神经元的特征接收器（一个向量列表, 也可以理解为是上层输出的激活值）
     */
    List<BigDecimal> inputFeatureVector = new ArrayList<>();

    /**
     * 激活函数值
     */
    BigDecimal activationFunctionValue;

    /**
     * 权重向量积
     */
    BigDecimal sumZ = new BigDecimal(0);

    /**
     * 神经元的特征权重向量（一个向量列表）
     */
    List<BigDecimal> weightVector = new ArrayList<>();

    /**
     * 该神经元的误差项
     */
    BigDecimal neuralUnitErrorItem = new BigDecimal(0);

    /**
     * 下一层神经网络的指针
     */
    SingleLayerNeuralNetwork nextLayer;

    /**
     * 初始化神经元
     */
    public NeuralUnit(List<BigDecimal> weightVectorInit, BigDecimal offset) {

        for (BigDecimal weightInit : weightVectorInit) {
            weightVector.add(weightInit);
        }
        // 偏置 b 也是权重
        weightVector.add(offset);
    }

    public NeuralUnit(String name, List<BigDecimal> weightVectorInit, BigDecimal offset) {
        this.name = name;
        for (BigDecimal weightInit : weightVectorInit) {
            weightVector.add(weightInit);
        }
        // 偏置 b 也是权重
        weightVector.add(offset);
    }



    /**
     * 计算神经值
     */
    public BigDecimal compute(final List<BigDecimal> inputFeatures) {
        // 神经元读取数据
        readFeature(inputFeatures);

        // (涉及计算 - 乘法、加法) 向量积
        sumZ = new BigDecimal(0);
        for (int i = 0; i < inputFeatureVector.size(); i++) {
            sumZ = MathUtil.add(sumZ,
                    MathUtil.multiply(inputFeatureVector.get(i), weightVector.get(i)));
        }

        // 计算激活值，并输出
        activationFunctionValue = this.activationFunction(sumZ);
        return activationFunctionValue;
    }

    /**
     * -------------------------------------------------------------/
     * <p>
     * /**
     * 读取特征值到神经元缓存中
     *
     * @param inputFeatures
     */
    private void readFeature(final List<BigDecimal> inputFeatures) {
        inputFeatureVector.clear();
        for (BigDecimal feature : inputFeatures) {
            inputFeatureVector.add(feature);
        }

        // 偏置 b 的特征固定为 1. (如果把偏置也看成一个权重，那么它是特殊的，特殊之处在于它的特征值 feature 永远为 1)
        inputFeatureVector.add(new BigDecimal(1));
        // 校验输入值是否跟权重数量匹配
        if (inputFeatureVector.size() != weightVector.size()) {
            throw new RuntimeException(
                    "输入特征值的数量为" + inputFeatureVector.size()
                            + " 与本神经元的权重数量" + weightVector.size()
                            + "不相等，无法配对");
        }
    }

    public List<BigDecimal> getInputFeatureVector() {
        return inputFeatureVector;
    }

    public void setInputFeatureVector(List<BigDecimal> inputFeatureVector) {
        this.inputFeatureVector = inputFeatureVector;
    }

    public BigDecimal getActivationFunctionValue() {
        return activationFunctionValue;
    }

    public void setActivationFunctionValue(BigDecimal activationFunctionValue) {
        this.activationFunctionValue = activationFunctionValue;
    }

    public BigDecimal getSumZ() {
        return sumZ;
    }

    public void setSumZ(BigDecimal sumZ) {
        this.sumZ = sumZ;
    }

    public List<BigDecimal> getWeightVector() {
        return weightVector;
    }

    public void setWeightVector(List<BigDecimal> weightVector) {
        this.weightVector = weightVector;
    }

    public int getLayerPositionIndex() {
        return layerPositionIndex;
    }

    public void setLayerPositionIndex(int layerPositionIndex) {
        this.layerPositionIndex = layerPositionIndex;
    }

    @Override
    public void computeError(List<BigDecimal> errorSource) {
        /**
         * 计算出 Z 偏导（每种损失函数的计算方式都不一样）
         */
        BigDecimal partialDerivativeZ = this.computePartialDerivativeZOfActivation(this.activationFunctionValue);
        // 读取当了神经元（通过序号指定）与上一次神经元链接的权重。
        List<BigDecimal> weightOfErrorSourceLayerByIndex = readWeightByIndex(errorSource);

        this.neuralUnitErrorItem = new BigDecimal(0);
        for (int i = 0; i < errorSource.size(); i++) {
            BigDecimal layerErrorElement = errorSource.get(i);
            // 每一个误差来源层的输入权重
            BigDecimal weight = weightOfErrorSourceLayerByIndex.get(i);
            // 权重所在的来源层神经的误差项
            // (涉及计算 - 乘法) 对上一个层误差的贡献度 【需要重点理解】
            BigDecimal contributionOfError = MathUtil.multiply(weight, layerErrorElement);
            // (涉及计算 - 乘法) 每一个误差来源层的输入权重 * 权重所在的来源层神经的误差项 * 当前神经元的激活值
            BigDecimal item = MathUtil.multiply(contributionOfError, partialDerivativeZ);
            // (涉及计算 - 加法) 累加
            this.neuralUnitErrorItem = MathUtil.add(this.neuralUnitErrorItem, item);
        }
    }

    /**
     * 读取用于计算误差项的权重
     *
     * @return
     */
    private List<BigDecimal> readWeightByIndex(List<BigDecimal> errorSource) {
        List<BigDecimal> weightOfErrorSourceLayerByIndex = new ArrayList<>();
        if (this.nextLayer == null || this.nextLayer.getNeuralLayer().isEmpty()) {
            for (int i = 0; i < errorSource.size(); i++) {
                // 如果没有神经元，说明这个是输出层，输出层只有误差向量，那么他的权重都为 1
                weightOfErrorSourceLayerByIndex.add(new BigDecimal(1));
            }
        } else {
            for (int i = 0; i < this.nextLayer.getNeuralLayer().size(); i++) {
                // 如果有神经元，说明这个是隐藏层，隐藏层的权重列表要根据 neuralUnitIndex 从每个神经元中获取
                NeuralUnit neuralUnitFromNextLayer = this.nextLayer.getNeuralLayer().get(i);
                // 获取下一层网络与当前神经元链接的权重
                BigDecimal weightWithCurrentNeuralUnit = neuralUnitFromNextLayer.getWeightVector().get(this.layerPositionIndex);
                weightOfErrorSourceLayerByIndex.add(weightWithCurrentNeuralUnit);
            }
        }
        return weightOfErrorSourceLayerByIndex;
    }

    public BigDecimal getNeuralUnitErrorItem() {
        return neuralUnitErrorItem;
    }

    public void setNeuralUnitErrorItem(BigDecimal neuralUnitErrorItem) {
        this.neuralUnitErrorItem = neuralUnitErrorItem;
    }

    public SingleLayerNeuralNetwork getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(SingleLayerNeuralNetwork nextLayer) {
        this.nextLayer = nextLayer;
    }
}
