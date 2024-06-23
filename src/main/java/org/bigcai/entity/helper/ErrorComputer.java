package org.bigcai.entity.helper;

import java.math.BigDecimal;
import java.util.List;

public interface ErrorComputer {
    public void computeError(List<BigDecimal> errorSource);
}
