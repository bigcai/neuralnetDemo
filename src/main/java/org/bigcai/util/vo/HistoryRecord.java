package org.bigcai.util.vo;

import java.util.ArrayList;
import java.util.List;

public class HistoryRecord {
    private String name;
    private List<Float> oldWeights = new ArrayList<>();
    private List<Float> newWeights = new ArrayList<>();

    public List<Float> getOldWeights() {
        return oldWeights;
    }

    public void setOldWeights(List<Float> oldWeights) {
        this.oldWeights = oldWeights;
    }

    public List<Float> getNewWeights() {
        return newWeights;
    }

    public void setNewWeights(List<Float> newWeights) {
        this.newWeights = newWeights;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public HistoryRecord(String name) {
        this.name = name;
    }
}
