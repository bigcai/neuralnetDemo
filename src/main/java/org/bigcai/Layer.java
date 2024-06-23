package org.bigcai;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

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
