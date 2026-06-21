package com.antonio.visionsector.engine.vsa;

/**
 * Rappresenta un Ipervettore Bipolare (MAP-B) compresso in un array di bit (long).
 */
public class VsaVector {

    // 10000 dimensioni divise per 64 bit = 157 long (l'ultimo array avrà dei bit inutilizzati, ma va bene)
    public static final int DIMENSIONS = 10000;
    private static final int ARRAY_SIZE = (DIMENSIONS + 63) / 64;

    private long[] blocks;

    /**
     * Costruisce un vettore BSC a partire da un array di blocchi a 64 bit.
     */
    public VsaVector(long[] blocks) {
        if (blocks.length != ARRAY_SIZE) {
            throw new IllegalArgumentException("Dimensione array errata");
        }
        this.blocks = blocks;
    }

    /**
     * OPERAZIONE 1: BINDING (Associazione)
     * In BSC, il binding è l'operatore logico XOR.
     */
    public VsaVector bind(VsaVector other) {
        long[] result = new long[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            // Un singolo ciclo di clock del processore esegue il binding di 64 dimensioni!
            result[i] = this.blocks[i] ^ other.blocks[i];
        }
        return new VsaVector(result);
    }

    /**
     * LA MEMORIA CLEAN-UP: Distanza di Hamming
     * Calcola quanti bit sono diversi tra questo vettore e un altro.
     */
    public int calculateHammingDistance(VsaVector other) {
        int distance = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            // Troviamo i bit discordanti con lo XOR
            long differences = this.blocks[i] ^ other.blocks[i];

            // MAGIA HARDWARE: POPCNT
            // Il processore (ARM o x86) conta i bit a '1' a livello di silicio.
            distance += Long.bitCount(differences);
        }
        return distance;
    }

    /**
     * Ritorna la Similarità del Coseno equivalente (utile per paragoni statistici e Z-Score).
     * Formula: 1 - 2 * (Hamming / Dimensioni)
     */
    public double getCosineSimilarity(VsaVector other) {
        int h = calculateHammingDistance(other);
        return 1.0 - 2.0 * ((double) h / DIMENSIONS);
    }

    public long[] getBlocks() {
        return blocks;
    }

    /**
     * Setter ottimizzato "Zero-Allocation".
     * Invece di cambiare il puntatore dell'array, sovrascrive fisicamente
     * i 157 long esistenti in memoria, evitando di risvegliare il Garbage Collector.
     */
    public void setBlocks(long[] newBlocks) {
        if (newBlocks != null && newBlocks.length == 157) {
            System.arraycopy(newBlocks, 0, this.blocks, 0, 157);
        }
    }
}