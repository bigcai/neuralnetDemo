package org.bigcai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Layer 这是一个抽象的层的概念，
 * 每一个层包含了输入层和输出层缓存。
 * 以及层缓存的使用方法
 */
public class Layer {

    /**
     * 输入值缓存
     */
    public List<BigDecimal> inputFeatureCache = new ArrayList<>();

    /**
     * 输出值缓存
     */
    public List<BigDecimal> outputActivationCache = new ArrayList<>();

    public void readyCompute(List<BigDecimal> features) {
        inputFeatureCache.clear();
        for (BigDecimal feature : features) {
            inputFeatureCache.add(feature);
        }
        outputActivationCache.clear();
    }

}
