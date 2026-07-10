package com.antonio.visionsector.network.dto;

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class ModuleBundleDTO {

    @SerializedName("moduleId")
    private Long moduleId;
    @SerializedName("moduleName")
    public String moduleName;

    @SerializedName("visibility")
    public String visibility;
    @SerializedName("encoderFileName")
    public String encoderFileName;
    @SerializedName("statsFileName")
    public String statsFileName;

    @SerializedName("rademacherSeed")
    public Long rademacherSeed;

    @SerializedName("classes")
    public List<ClassEntryDTO> classes;

    public ModuleBundleDTO() {
    }

    public Long getModuleId() {
        return moduleId;
    }

    public void setModuleId(Long moduleId) {
        this.moduleId = moduleId;
    }

    public String getModuleName() {
        return moduleName;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
    }

    public String getVisibility() {
        return visibility;
    }

    public void setVisibility(String visibility) {
        this.visibility = visibility;
    }

    public String getEncoderFileName() {
        return encoderFileName;
    }

    public void setEncoderFileName(String encoderFileName) {
        this.encoderFileName = encoderFileName;
    }

    public String getStatsFileName() {
        return statsFileName;
    }

    public void setStatsFileName(String statsFileName) {
        this.statsFileName = statsFileName;
    }

    public Long getRademacherSeed() {
        return rademacherSeed;
    }

    public void setRademacherSeed(Long rademacherSeed) {
        this.rademacherSeed = rademacherSeed;
    }

    public List<ClassEntryDTO> getClasses() {
        return classes;
    }

    public void setClasses(List<ClassEntryDTO> classes) {
        this.classes = classes;
    }
}
