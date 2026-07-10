package com.antonio.visionsector.network.dto;

import com.google.gson.annotations.SerializedName;

public class ClassEntryDTO {

    @SerializedName("calssId")
    private Long moduleId;

    @SerializedName("name")
    public String name;
    @SerializedName("sampleCount")
    public Integer sampleCount;

    @SerializedName("vectorBase64")
    public String vectorBase64;

    public ClassEntryDTO() {
    }

    public Long getModuleId() {
        return moduleId;
    }

    public void setModuleId(Long moduleId) {
        this.moduleId = moduleId;
    }

    public String getName() {
        return name;
    }

    public Integer getSampleCount() {
        return sampleCount;
    }

    public String getVectorBase64() {
        return vectorBase64;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setSampleCount(Integer sampleCount) {
        this.sampleCount = sampleCount;
    }

    public void setVectorBase64(String vectorBase64) {
        this.vectorBase64 = vectorBase64;
    }
}
