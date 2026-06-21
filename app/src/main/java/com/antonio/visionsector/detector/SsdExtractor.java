package com.antonio.visionsector.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectionResult;

import java.util.List;

/**
 * SSD object detector using MediaPipe Tasks Vision con accelerazione GPU.
 *
 * Model: MobileNetV2-SSD (Class-Agnostic), trained with MediaPipe Model Maker.
 * Input:  float32[1, 256, 256, 3]
 * Output: boxes  float32[1, 12276, 4]   (raw anchor offsets)
 *         scores float32[1, 12276, 2]   (index 1 = object confidence)
 *
 * All anchor decoding, NMS, and resize are handled internally by ObjectDetector.
 *
 * Name‑clash: com.google.mediapipe.tasks.core.Delegate (enum: CPU/GPU) e
 * org.tensorflow.lite.Delegate (interfaccia) NON coincidono. In questo file
 * importiamo solo il Delegate MediaPipe; in altri file usare fully‑qualified name.
 */
public class SsdExtractor {

    private static final String TAG = "SsdDetector";
    private static final float SCORE_THRESHOLD = 0.3f;

    private final ObjectDetector detector;
    private final boolean isUsingGpu;

    /**
     * Creates the extractor loading the model from app assets.
     * Tenta GPU; in caso di fallimento ricade su CPU.
     *
     * @param context   application context
     * @param modelPath asset path, e.g. "mobilenet_ssd.tflite"
     */
    public SsdExtractor(Context context, String modelPath) throws Exception {
        boolean gpuOk = false;
        ObjectDetector det;

        // ── Tentativo GPU ─────────────────────────────────────────────
        try {
            BaseOptions gpuOpts = BaseOptions.builder()
                    .setModelAssetPath(modelPath)
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.GPU)
                    .build();

            ObjectDetector.ObjectDetectorOptions opts =
                    ObjectDetector.ObjectDetectorOptions.builder()
                            .setBaseOptions(gpuOpts)
                            .setMaxResults(1)
                            .setScoreThreshold(SCORE_THRESHOLD)
                            .setRunningMode(RunningMode.IMAGE)
                            .build();

            det = ObjectDetector.createFromOptions(context, opts);
            gpuOk = true;
            Log.i(TAG, "SSD avviato con GPU (MediaPipe Delegate.GPU) — model: " + modelPath);
        } catch (Exception gpuError) {
            Log.w(TAG, "GPU SSD non disponibile: " + gpuError.getMessage() + " — ricado su CPU");

            BaseOptions cpuOpts = BaseOptions.builder()
                    .setModelAssetPath(modelPath)
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                    .build();

            ObjectDetector.ObjectDetectorOptions opts =
                    ObjectDetector.ObjectDetectorOptions.builder()
                            .setBaseOptions(cpuOpts)
                            .setMaxResults(1)
                            .setScoreThreshold(SCORE_THRESHOLD)
                            .setRunningMode(RunningMode.IMAGE)
                            .build();

            det = ObjectDetector.createFromOptions(context, opts);
            Log.i(TAG, "SSD avviato con CPU — model: " + modelPath);
        }

        this.detector = det;
        this.isUsingGpu = gpuOk;
    }

    /**
     * Runs detection on a bitmap frame.
     *
     * @param bitmap input frame at any resolution
     * @return bounding box normalized to [0.0, 1.0] relative to bitmap dimensions,
     *         or null if no object exceeds the confidence threshold
     */
    public RectF detectObject(Bitmap bitmap) {
        int imgW = bitmap.getWidth();
        int imgH = bitmap.getHeight();

        MPImage mpImage = new BitmapImageBuilder(bitmap).build();
        ObjectDetectionResult result = detector.detect(mpImage);
        List<Detection> detections = result.detections();

        if (detections == null || detections.isEmpty()) {
            return null;
        }

        RectF bbox = detections.get(0).boundingBox();

        return new RectF(
                bbox.left   / imgW,
                bbox.top    / imgH,
                bbox.right  / imgW,
                bbox.bottom / imgH
        );
    }

    public boolean isUsingGpu() {
        return isUsingGpu;
    }

    /** Releases native resources. */
    public void close() {
        try {
            detector.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing detector", e);
        }
    }
}
