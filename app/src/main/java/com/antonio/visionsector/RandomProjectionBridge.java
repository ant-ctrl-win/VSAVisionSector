package com.antonio.visionsector;

import java.util.Random;

public class RandomProjectionBridge {
    private final int cnnFeatureSize;
    private final int vsaDimensions;

    // Invertito da [1280][10000] a [10000][1280] per garantire la Cache Locality!
    private final boolean[][] projectionMatrix;

    public RandomProjectionBridge(int cnnFeatureSize, long seed) {
        this.cnnFeatureSize = cnnFeatureSize;          // 1280
        this.vsaDimensions = 10000;                    // 10000
        this.projectionMatrix = new boolean[vsaDimensions][cnnFeatureSize]; // Mantieni layout ottimizzato [10000][1280]

        Random random = new Random(seed);

        // RISPETTIAMO L'ORDINE ORIGINALE DI ESTRAZIONE DEL SEED!
        for (int i = 0; i < cnnFeatureSize; i++) {
            for (int d = 0; d < vsaDimensions; d++) {
                boolean casuale = random.nextBoolean(); // Estrae nell'ordine corretto

                // Salviamo in modo trasposto nella matrice ottimizzata per la cache
                projectionMatrix[d][i] = casuale;
            }
        }
    }

    /**
     * Versione ottimizzata: Accetta un array di riutilizzo per azzerare la Garbage Collection
     */
    public double[] projectToContinuous(float[] cnnFeatures, double[] reuseVector) {
        double sumOfSquares = 0.0;

        for (int d = 0; d < vsaDimensions; d++) {
            double sum = 0.0;

            // Estraiamo il riferimento alla riga: calcolo sequenziale in cache L1
            boolean[] row = projectionMatrix[d];

            for (int i = 0; i < cnnFeatureSize; i++) {
                if (row[i]) {
                    sum += cnnFeatures[i];
                } else {
                    sum -= cnnFeatures[i];
                }
            }
            reuseVector[d] = sum;
            sumOfSquares += sum * sum;
        }

        // Normalizzazione L2
        double magnitude = Math.sqrt(sumOfSquares);
        if (magnitude > 1e-9) {
            for (int d = 0; d < vsaDimensions; d++) {
                reuseVector[d] /= magnitude;
            }
        }
        return reuseVector;
    }

    /**
     * Estrae e proietta una singola cella spaziale direttamente dall'array gigante della CNN.
     * Usa un offset per evitare di instanziare nuovi array float[] e affaticare il Garbage Collector.
     */
    // vsaDimensions
    // cnnFeatureSize

    public void projectSpatialCellToContinuous(float[] spatialGrid, int startOffset, double[] outContinuous) {
        int[] activeIndices = new int[cnnFeatureSize];
        float[] activeValues = new float[cnnFeatureSize];
        int activeCount = 0;

        for (int i = 0; i < cnnFeatureSize; i++) {
            float val = spatialGrid[startOffset + i];
            if (val != 0.0f) { // Consideriamo solo i filtri che hanno visto qualcosa
                activeIndices[activeCount] = i;
                activeValues[activeCount] = val;
                activeCount++;
            }
        }

        // 2. PROIEZIONE IPERDIMENSIONALE VELOCE
        // Eseguiamo il calcolo solo sulle feature reali (risparmio dell'80% di cicli CPU)
        for (int j = 0; j < 10000; j++) {
            boolean[] basis = projectionMatrix[j];
            double sum = 0.0;

            for (int k = 0; k < activeCount; k++) {
                int idx = activeIndices[k];
                if (basis[idx]) {
                    sum += activeValues[k];
                } else {
                    sum -= activeValues[k];
                }
            }
            outContinuous[j] = sum;
        }
    }


    public void projectFeatureVector(float[] features, double[] outContinuous) {
        int[] activeIndices = new int[cnnFeatureSize];
        float[] activeValues = new float[cnnFeatureSize];
        int activeCount = 0;

        for (int i = 0; i < cnnFeatureSize; i++) {
            float val = features[i];
            if (val != 0.0f) {
                activeIndices[activeCount] = i;
                activeValues[activeCount] = val;
                activeCount++;
            }
        }

        // 2. PROIEZIONE VELOCE
        for (int j = 0; j < 10000; j++) {
            boolean[] basis = projectionMatrix[j];
            double sum = 0.0;
            for (int k = 0; k < activeCount; k++) {
                int idx = activeIndices[k];
                if (basis[idx]) sum += activeValues[k];
                else sum -= activeValues[k];
            }
            outContinuous[j] = sum;
        }
    }

    /** Normalizza L2 un vettore continuo in-place */
    public void normalizeL2(double[] vector) {
        double sumSq = 0;
        for (int i = 0; i < vsaDimensions; i++) sumSq += vector[i] * vector[i];
        double norm = Math.sqrt(sumSq);
        if (norm > 1e-9) {
            for (int i = 0; i < vsaDimensions; i++) vector[i] /= norm;
        }
    }


    /**
     * Versione ottimizzata di Binarize: Scrive direttamente nel VsaVector pre-allocato
     */
    public void binarizeTo(double[] continuousVector, long[] reuseBlocks) {
        // Pulisce l'array dei blocchi impostando tutto a zero
        java.util.Arrays.fill(reuseBlocks, 0L);

        for (int d = 0; d < vsaDimensions; d++) {
            if (continuousVector[d] > 0) {
                int blockIndex = d / 64;
                int bitPosition = d % 64;
                reuseBlocks[blockIndex] |= (1L << bitPosition);
            }
        }
    }
}