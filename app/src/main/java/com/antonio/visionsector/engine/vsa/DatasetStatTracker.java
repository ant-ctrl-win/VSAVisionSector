package com.antonio.visionsector.engine.vsa;

import android.content.Context;
import java.io.DataInputStream;
import java.io.IOException;

public class DatasetStatTracker {
    private final int dimensions;
    private final double[] mean;
    private final double[] stdDev;

    public DatasetStatTracker(int dimensions, double[] mean, double[] stdDev) {
        this.dimensions = dimensions;
        this.mean = mean;
        this.stdDev = stdDev;
    }

    public double[] standardize(double[] continuousVector) {
        double[] standardized = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            standardized[i] = (continuousVector[i] - mean[i]) / stdDev[i];
        }
        return standardized;
    }

    public void standardizeInPlace(double[] vector) {
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (vector[i] - mean[i]) / stdDev[i];
        }
    }

    public int getDimensions() {
        return dimensions;
    }

    public double[] getMean() {
        return mean;
    }

    public double[] getStdDev() {
        return stdDev;
    }

    // Metodo specifico per Android per caricare il file binario dagli Assets
    public static DatasetStatTracker importFromAndroidAssets(Context context, String fileName) throws IOException {
        try (DataInputStream dis = new DataInputStream(context.getAssets().open(fileName))) {
            int dims = dis.readInt();
            double[] loadedMean = new double[dims];
            double[] loadedStd = new double[dims];

            for (int i = 0; i < dims; i++) {
                loadedMean[i] = dis.readDouble();
            }
            for (int i = 0; i < dims; i++) {
                loadedStd[i] = dis.readDouble();
            }
            return new DatasetStatTracker(dims, loadedMean, loadedStd);
        }



    }
}