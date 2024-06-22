package org.bigcai;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/**
 * 神经元
 */
public class NeuralUnit {
    public static final int SCALE = 4;
    /**
     * 神经元的特征接收器（一个向量列表, 也可以理解为是上层输出的激活值）
     */
    List<BigDecimal> inputFeatureVector = new ArrayList<>();
    /**
     * 神经元的特征权重向量（一个向量列表）
     */
    List<BigDecimal> weightVector = new ArrayList<>();
    /**
     * 求和
     */
    BigDecimal sum = new BigDecimal(0);

    /**
     * 激活函数值
     */
    BigDecimal activationFunctionValue;

    /**
     * 初始化神经元
     */
    public NeuralUnit(List<BigDecimal> weightVectorInit, BigDecimal offset) {

        for (BigDecimal weightInit: weightVectorInit) {
            weightInit = weightInit.setScale(SCALE, RoundingMode.HALF_UP);
            weightVector.add(weightInit);
        }
        // 偏置 b 也是权重
        offset = offset.setScale(SCALE, RoundingMode.HALF_UP);
        weightVector.add(offset);
    }

    /**
     * 计算神经值
     */
    public BigDecimal compute(final List<BigDecimal> inputFeatures) {
        // 神经元读取数据
        readFeature(inputFeatures);

        // 求和
        for (int i = 0; i < inputFeatureVector.size(); i++) {
            sum = sum.add(inputFeatureVector.get(i).multiply(weightVector.get(i)));
        }

        // 计算激活值，并输出
        activationFunctionValue = activationFunction(sum);
        activationFunctionValue.setScale(SCALE);
        return activationFunctionValue;
    }

    private BigDecimal activationFunction(BigDecimal sum) {
        BigDecimal exp = new BigDecimal(Math.exp(-1 * sum.doubleValue()));
        return new BigDecimal(1).divide(exp.add(new BigDecimal(1)), SCALE, RoundingMode.HALF_UP);
    }


    /**-------------------------------------------------------------/

    /**
     * 读取特征值到神经元
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
        if(inputFeatureVector.size() != weightVector.size()) {
            throw new RuntimeException(
                    "输入特征值的数量为" + inputFeatureVector.size()
                    + " 与本神经元的权重数量" + weightVector.size()
                    + "不相等，无法配对");
        }
    }


}
