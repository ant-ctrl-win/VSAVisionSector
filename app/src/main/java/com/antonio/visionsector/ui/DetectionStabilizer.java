package com.antonio.visionsector.ui;

import android.graphics.RectF;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Filtro di smussamento per la stabilizzazione temporale e spaziale
 * delle predizioni dell'Intelligenza Artificiale nei flussi video.
 */
public class DetectionStabilizer {

    // --- SOGLIE DI CLASSIFICAZIONE (Z‑Score su distribuzione nulla) ---
    private static final double SIGMA_NULL = 1.0 / Math.sqrt(10000.0);
    private static final double Z_THRESHOLD = 3.0;
    private static final double Z_MARGIN = 0.5;

    // --- MARKER INTERNI PER IL MAJORITY VOTING ---
    private static final String MARKER_UNKNOWN = "__UNKNOWN__";
    private static final String MARKER_AMBIGUOUS = "__AMBIGUOUS__";

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

    // ── Tipi ────────────────────────────────────────────────────────

    public enum Verdict {
        UNKNOWN,      // maxZScore < Z_THRESHOLD
        AMBIGUOUS,    // margin top1Z-top2Z < Z_MARGIN
        MATCH_FOUND   // classificazione sicura
    }

    public static class SimilarityResult {
        public final String className;
        public final double similarity;

        public SimilarityResult(String className, double similarity) {
            this.className = className;
            this.similarity = similarity;
        }
    }

    public static class PredictionVerdict {
        public final Verdict verdict;
        public final String className;
        public final double zScore;

        private PredictionVerdict(Verdict verdict, String className, double zScore) {
            this.verdict = verdict;
            this.className = className;
            this.zScore = zScore;
        }

        public static PredictionVerdict unknown(double zScore) {
            return new PredictionVerdict(Verdict.UNKNOWN, null, zScore);
        }

        public static PredictionVerdict ambiguous(double zScore) {
            return new PredictionVerdict(Verdict.AMBIGUOUS, null, zScore);
        }

        public static PredictionVerdict match(String className, double zScore) {
            return new PredictionVerdict(Verdict.MATCH_FOUND, className, zScore);
        }
    }

    public static class StabilizedResult {
        public RectF box;
        public String className;
        public boolean isVisible;
        public Verdict verdict;

        public StabilizedResult(RectF box, String className, boolean isVisible, Verdict verdict) {
            this.box = box;
            this.className = className;
            this.isVisible = isVisible;
            this.verdict = verdict;
        }
    }

    // ── Classificazione ─────────────────────────────────────────────

    /**
     * Valuta la predizione in base a Z‑Score sulla distribuzione nulla (random BSC).
     * @param results lista di {@link SimilarityResult} ordinata per similarity decrescente
     * @param totalArchetypes numero totale di archetipi nel database
     * @return il verdetto con Z‑Score e eventuale classe associata
     */
    public static PredictionVerdict evaluatePrediction(List<SimilarityResult> results, int totalArchetypes) {
        if (results == null || results.isEmpty()) {
            return PredictionVerdict.unknown(0.0);
        }

        SimilarityResult top1 = results.get(0);
        double top1Z = top1.similarity / SIGMA_NULL;

        if (top1Z < Z_THRESHOLD) {
            return PredictionVerdict.unknown(top1Z);
        }

        if (totalArchetypes > 1 && results.size() > 1) {
            SimilarityResult top2 = results.get(1);
            double top2Z = top2.similarity / SIGMA_NULL;
            double marginZ = top1Z - top2Z;
            if (marginZ < Z_MARGIN) {
                return PredictionVerdict.ambiguous(top1Z);
            }
        }

        return PredictionVerdict.match(top1.className, top1Z);
    }

    // ── Stabilizzazione ─────────────────────────────────────────────

    /**
     * Da chiamare ad ogni frame, anche quando l'AI non trova nulla (passando null).
     */
    public StabilizedResult update(RectF newBox, Verdict verdict, String className) {

        // Sanity check: scarta box anomali (falsi positivi o rumore)
        if (newBox != null) {
            RectF clamped = new RectF(
                    Math.max(0f, Math.min(1f, newBox.left)),
                    Math.max(0f, Math.min(1f, newBox.top)),
                    Math.max(0f, Math.min(1f, newBox.right)),
                    Math.max(0f, Math.min(1f, newBox.bottom))
            );
            float area = clamped.width() * clamped.height();
            if (area > MAX_BOX_AREA_RATIO || area < MIN_BOX_AREA_RATIO) {
                newBox = null;
            } else {
                newBox = clamped;
            }
        }

        // CASO 1: L'IA non ha trovato l'oggetto in questo frame
        if (newBox == null) {
            missingFramesCount++;
            if (missingFramesCount >= MAX_MISSING_FRAMES) {
                // Sparito da troppo tempo, resettiamo tutto
                reset();
                return new StabilizedResult(null, null, false, Verdict.UNKNOWN);
            } else {
                // Coasting: è sparito in questo frame, ma lo manteniamo a schermo per inerzia
                String votedClass = getMajorityClass();
                Verdict coastVerdict = (votedClass != null) ? Verdict.MATCH_FOUND : Verdict.UNKNOWN;
                return new StabilizedResult(currentSmoothedBox, votedClass, true, coastVerdict);
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
        // UNKNOWN e AMBIGUOUS non inquinano il majority voting
        if (verdict == Verdict.MATCH_FOUND && className != null) {
            classHistory.addLast(className);
        } else if (verdict == Verdict.UNKNOWN) {
            classHistory.addLast(MARKER_UNKNOWN);
        } else if (verdict == Verdict.AMBIGUOUS) {
            classHistory.addLast(MARKER_AMBIGUOUS);
        }
        if (classHistory.size() > HISTORY_SIZE) {
            classHistory.removeFirst();
        }

        String mostFrequentClass = getMajorityClass();

        // Determina il verdetto di uscita: se il majority voting dà una classe, è MATCH_FOUND
        Verdict effectiveVerdict;
        String effectiveClass;
        if (mostFrequentClass != null) {
            effectiveVerdict = Verdict.MATCH_FOUND;
            effectiveClass = mostFrequentClass;
        } else {
            // Nessuna classe votata: usa il verdetto corrente grezzo
            effectiveVerdict = verdict;
            effectiveClass = (verdict == Verdict.MATCH_FOUND) ? className : null;
        }

        return new StabilizedResult(currentSmoothedBox, effectiveClass, true, effectiveVerdict);
    }

    /**
     * Trova la classe che è apparsa più spesso negli ultimi N frame,
     * escludendo UNKNOWN e AMBIGUOUS dal conteggio.
     */
    private String getMajorityClass() {
        if (classHistory.isEmpty()) return null;

        Map<String, Integer> counts = new HashMap<>();
        for (String c : classHistory) {
            if (MARKER_UNKNOWN.equals(c) || MARKER_AMBIGUOUS.equals(c)) continue;
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
