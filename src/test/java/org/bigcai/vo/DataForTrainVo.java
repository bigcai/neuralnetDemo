package org.bigcai.vo;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class DataForTrainVo {
    public List<BigDecimal> features = new ArrayList<>();

    public BigDecimal actualValue;

    public DataForTrainVo(BigDecimal feat1, BigDecimal feat2, BigDecimal actualValue) {
        features.add(feat1);
        features.add(feat2);
        this.actualValue = actualValue;
    }
}
