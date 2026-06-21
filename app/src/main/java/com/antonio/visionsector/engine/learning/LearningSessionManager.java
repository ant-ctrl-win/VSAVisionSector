package com.antonio.visionsector.engine.learning;

import android.util.Log;

import java.util.Arrays;

/**
 * Gestisce una sessione di apprendimento multi‑shot continua via ImageAnalysis.
 *
 * Flusso:
 *   1. L'utente preme "+", inserisce il nome, startLearning() avvia la sessione.
 *   2. Per ogni frame in cui l'SSD trova un box, addFrame() accumula il vettore
 *      VSA (L2‑normalizzato) tramite bundling additivo.
 *   3. Raggiunto il target, lo stato diventa PROCESSING e getFinalBundleAndReset()
 *      restituisce l'archetipo bundlato.
 *   4. L'utente può annullare con cancel() o forzare il completamento con finish().
 *
 * Thread safety: tutti i metodi pubblici sono synchronized.
 */
public class LearningSessionManager {

    private static final String TAG = "VSA_LearningSession";

    public enum State {
        IDLE,
        COLLECTING,
        PROCESSING,
        CANCELLED
    }

    private State currentState = State.IDLE;
    private String pendingClassName;
    private int framesCollected;
    private final int targetFrames;
    private final int vsaDimensions;

    private final double[] bundleVector;

    public LearningSessionManager(int vsaDimensions, int targetFrames) {
        this.vsaDimensions = vsaDimensions;
        this.targetFrames = Math.max(1, targetFrames);
        this.bundleVector = new double[vsaDimensions];
    }

    /**
     * Avvia la sessione di apprendimento.
     * @param className nome della classe da apprendere
     */
    public synchronized void startLearning(String className) {
        if (className == null || className.trim().isEmpty()) {
            Log.e(TAG, "Impossibile avviare il learning: nome classe non valido.");
            return;
        }
        this.pendingClassName = className.trim();
        this.framesCollected = 0;
        Arrays.fill(bundleVector, 0.0);
        this.currentState = State.COLLECTING;
        Log.d(TAG, "Sessione di apprendimento avviata per: " + pendingClassName
                + " (target=" + targetFrames + " shot)");
    }

    /**
     * Aggiunge un vettore VSA L2‑normalizzato all'accumulatore.
     * @param normalizedContinuousVSA vettore double[vsaDimensions] già proiettato e normalizzato L2
     * @return true se il target di frame è stato raggiunto e la sessione è PROCESSING
     */
    public synchronized boolean addFrame(double[] normalizedContinuousVSA) {
        if (currentState != State.COLLECTING) return false;
        if (normalizedContinuousVSA == null || normalizedContinuousVSA.length != vsaDimensions) {
            return false;
        }

        for (int i = 0; i < vsaDimensions; i++) {
            bundleVector[i] += normalizedContinuousVSA[i];
        }

        framesCollected++;
        Log.d(TAG, "Frame raccolto: " + framesCollected + "/" + targetFrames
                + " per " + pendingClassName);

        if (framesCollected >= targetFrames) {
            this.currentState = State.PROCESSING;
            Log.i(TAG, "Target raggiunto per '" + pendingClassName + "' — pronto per il salvataggio");
            return true;
        }
        return false;
    }

    /**
     * Forza il completamento anticipato. Usa i sample già raccolti (se almeno 1).
     * @return il vettore bundlato (non normalizzato), o null se nessun sample
     */
    public synchronized double[] finish() {
        if (framesCollected == 0 || currentState == State.CANCELLED) {
            Log.w(TAG, "finish() chiamato ma nessun frame raccolto");
            currentState = State.CANCELLED;
            return null;
        }
        Log.i(TAG, "Sessione '" + pendingClassName + "' completata manualmente con "
                + framesCollected + " shot");
        currentState = State.PROCESSING;
        return getFinalBundleAndReset();
    }

    /**
     * Restituisce il vettore bundlato e resetta lo stato.
     * Da chiamare dopo che addFrame() ha restituito true.
     */
    public synchronized double[] getFinalBundleAndReset() {
        if (framesCollected == 0) {
            Log.w(TAG, "getFinalBundleAndReset: nessun frame raccolto");
            reset();
            return null;
        }
        double[] finalResult = bundleVector.clone();
        String name = pendingClassName;
        int collected = framesCollected;
        reset();
        Log.i(TAG, "Bundle finale creato per '" + name + "' (" + collected + " shot)");
        return finalResult;
    }

    public synchronized void cancel() {
        if (currentState == State.CANCELLED) return;
        String name = pendingClassName;
        reset();
        currentState = State.CANCELLED;
        Log.d(TAG, "Sessione annullata per '" + name + "'");
    }

    public synchronized void reset() {
        this.currentState = State.IDLE;
        this.pendingClassName = null;
        this.framesCollected = 0;
        Arrays.fill(bundleVector, 0.0);
    }

    // ── Getters ────────────────────────────────────────────────────────

    public synchronized State getCurrentState() { return currentState; }
    public synchronized int getFramesCollected() { return framesCollected; }
    public synchronized int getTargetFrames() { return targetFrames; }
    public synchronized String getPendingClassName() { return pendingClassName; }

    /**
     * Restituisce il progresso come frazione [0.0, 1.0].
     */
    public synchronized float getProgress() {
        if (targetFrames <= 0) return 0f;
        return Math.min(1f, (float) framesCollected / targetFrames);
    }
}
