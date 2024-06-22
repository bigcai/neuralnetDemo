package org.bigcai.error.function;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public abstract class ErrorFunction {
    public abstract BigDecimal diff(List<BigDecimal> outputList, List<BigDecimal> actualList);
}
