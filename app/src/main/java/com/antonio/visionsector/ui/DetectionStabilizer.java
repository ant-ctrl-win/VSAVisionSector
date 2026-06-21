package com.antonio.visionsector.ui;

import android.graphics.RectF;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Filtro di smussamento per la stabilizzazione temporale e spaziale
 * delle predizioni dell'Intelligenza Artificiale nei flussi video.
 */
public class DetectionStabilizer {

    // --- PARAMETRI DI TUNING ---
    // Quanto è "reattivo" il box (1.0 = nessun filtro, 0.1 = molto lento e fluido)
    private static final float SMOOTHING_FACTOR = 0.3f;

    // Quanti frame consecutivi usiamo per il Majority Voting della classe
    private static final int HISTORY_SIZE = 5;

    // Quanti frame di "pazienza" diamo se l'oggetto scompare prima di toglierlo dall'UI
    private static final int MAX_MISSING_FRAMES = 6;

    // Scarta box che coprono più dell'80% del frame (falsi positivi dell'SSD)
    private static final float MAX_BOX_AREA_RATIO = 0.80f;

    // Scarta box la cui area è < 1% del frame (rumore)
    private static final float MIN_BOX_AREA_RATIO = 0.01f;

    // Stato interno
    private RectF currentSmoothedBox = null;
    private LinkedList<String> classHistory = new LinkedList<>();
    private int missingFramesCount = 0;

    public static class StabilizedResult {
        public RectF box;
        public String className;
        public boolean isVisible;

        public StabilizedResult(RectF box, String className, boolean isVisible) {
            this.box = box;
            this.className = className;
            this.isVisible = isVisible;
        }
    }

    /**
     * Da chiamare ad ogni frame, anche quando l'AI non trova nulla (passando null).
     */
    public StabilizedResult update(RectF newBox, String newClassName) {

        // Sanity check: scarta box anomali (falsi positivi o rumore)
        if (newBox != null && newClassName != null) {
            RectF clamped = new RectF(
                    Math.max(0f, Math.min(1f, newBox.left)),
                    Math.max(0f, Math.min(1f, newBox.top)),
                    Math.max(0f, Math.min(1f, newBox.right)),
                    Math.max(0f, Math.min(1f, newBox.bottom))
            );
            float area = clamped.width() * clamped.height();
            if (area > MAX_BOX_AREA_RATIO || area < MIN_BOX_AREA_RATIO) {
                newBox = null;
                newClassName = null;
            } else {
                newBox = clamped;
            }
        }

        // CASO 1: L'IA non ha trovato l'oggetto in questo frame
        if (newBox == null || newClassName == null) {
            missingFramesCount++;
            if (missingFramesCount >= MAX_MISSING_FRAMES) {
                // Sparito da troppo tempo, resettiamo tutto
                reset();
                return new StabilizedResult(null, null, false);
            } else {
                // Coasting: è sparito in questo frame, ma lo manteniamo a schermo per inerzia
                String votedClass = getMajorityClass();
                return new StabilizedResult(currentSmoothedBox, votedClass, true);
            }
        }

        // CASO 2: L'IA ha trovato l'oggetto
        missingFramesCount = 0; // Resettiamo il contatore sparizioni

        // 1. Stabilizzazione Spaziale (EMA)
        if (currentSmoothedBox == null) {
            // Primo frame, accettiamo il box così com'è
            currentSmoothedBox = new RectF(newBox);
        } else {
            // Calcoliamo la distanza tra il nuovo box e il vecchio.
            // Se l'utente muove il telefono di scatto, resettiamo per evitare "l'effetto fantasma".
            float centerDistX = Math.abs(currentSmoothedBox.centerX() - newBox.centerX());
            float centerDistY = Math.abs(currentSmoothedBox.centerY() - newBox.centerY());

            // Se si muove di oltre il 20% dello schermo di scatto, resetta lo smoothing
            if (centerDistX > 0.2f || centerDistY > 0.2f) {
                currentSmoothedBox = new RectF(newBox);
            } else {
                // Applicazione Media Mobile Esponenziale (Smussamento)
                currentSmoothedBox.left = (newBox.left * SMOOTHING_FACTOR) + (currentSmoothedBox.left * (1 - SMOOTHING_FACTOR));
                currentSmoothedBox.top = (newBox.top * SMOOTHING_FACTOR) + (currentSmoothedBox.top * (1 - SMOOTHING_FACTOR));
                currentSmoothedBox.right = (newBox.right * SMOOTHING_FACTOR) + (currentSmoothedBox.right * (1 - SMOOTHING_FACTOR));
                currentSmoothedBox.bottom = (newBox.bottom * SMOOTHING_FACTOR) + (currentSmoothedBox.bottom * (1 - SMOOTHING_FACTOR));
            }
        }

        // 2. Stabilizzazione Semantica (Majority Voting)
        classHistory.addLast(newClassName);
        if (classHistory.size() > HISTORY_SIZE) {
            classHistory.removeFirst();
        }

        String mostFrequentClass = getMajorityClass();

        return new StabilizedResult(currentSmoothedBox, mostFrequentClass, true);
    }

    /**
     * Trova la classe che è apparsa più spesso negli ultimi N frame.
     */
    private String getMajorityClass() {
        if (classHistory.isEmpty()) return null;

        Map<String, Integer> counts = new HashMap<>();
        for (String c : classHistory) {
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }

        String bestClass = null;
        int maxCount = -1;
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                bestClass = entry.getKey();
            }
        }
        return bestClass;
    }

    public void reset() {
        currentSmoothedBox = null;
        classHistory.clear();
        missingFramesCount = 0;
    }
}