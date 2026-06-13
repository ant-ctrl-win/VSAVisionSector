package com.antonio.visionsector;

import java.util.Random;

public class BottleneckBridge {
//    private final int inputDim = 1280;
//    private final int outputDim = 256;
//
//    // Matrice di pesi casuali fissi [256][1280]
//    private final float[][] projectionMatrix;
//
//    public BottleneckBridge(long seed) {
//        projectionMatrix = new float[outputDim][inputDim];
//        Random random = new Random(seed);
//
//        // Inizializzazione di Xavier/Glorot approssimata
//        float limit = (float) Math.sqrt(6.0 / (inputDim + outputDim));
//        for (int i = 0; i < outputDim; i++) {
//            for (int j = 0; j < inputDim; j++) {
//                // Valori casuali tra -limit e +limit
//                projectionMatrix[i][j] = (random.nextFloat() * 2 * limit) - limit;
//            }
//        }
//    }
//
//    /**
//     * Comprime una cella 1280-dim in una 256-dim.
//     */
//    public void reduceCell(float[] spatialGrid, int startOffset, float[] out256) {
//        for (int i = 0; i < outputDim; i++) {
//            float sum = 0.0f;
//            for (int j = 0; j < inputDim; j++) {
//                sum += spatialGrid[startOffset + j] * projectionMatrix[i][j];
//            }
//            // Applichiamo una ReLU opzionale per mantenere la sparsità
//            out256[i] = Math.max(0.0f, sum);
//        }
//    }
private final int inputDim = 1280;
    private final int outputDim = 256;

    // Matrice di proiezione statica [256][1280]
    private final float[][] projectionMatrix;

    public BottleneckBridge(long seed) {
        projectionMatrix = new float[outputDim][inputDim];
        Random random = new Random(seed);

        // Inizializzazione bilanciata per mantenere le magnitudini stabili
        float limit = (float) Math.sqrt(6.0 / (inputDim + outputDim));
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                projectionMatrix[i][j] = (random.nextFloat() * 2 * limit) - limit;
            }
        }
    }

    /**
     * Comprime il tensore CNN originale in un vettore ridotto.
     * * @param spatialGrid L'array piatto restituito dalla CNN (es. 62720 float)
     * @param startOffset L'offset della cella desiderata (es. la cella [3,3])
     * @param out256 Buffer pre-allocato di 256 float dove scrivere il risultato
     */
    public void reduceCell(float[] spatialGrid, int startOffset, float[] out256) {
        for (int i = 0; i < outputDim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < inputDim; j++) {
                sum += spatialGrid[startOffset + j] * projectionMatrix[i][j];
            }
            out256[i] = Math.max(0.0f, sum);
        }
    }

}