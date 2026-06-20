package com.antonio.visionsector;

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
 * SSD object detector using MediaPipe Tasks Vision.
 *
 * Model: MobileNetV2-SSD (Class-Agnostic), trained with MediaPipe Model Maker.
 * Input:  float32[1, 256, 256, 3]
 * Output: boxes  float32[1, 12276, 4]   (raw anchor offsets)
 *         scores float32[1, 12276, 2]   (index 1 = object confidence)
 *
 * All anchor decoding, NMS, and resize are handled internally by ObjectDetector.
 */
public class SsdExtractor {

    private static final String TAG = "SsdExtractor";
    private static final float SCORE_THRESHOLD = 0.3f;

    private final ObjectDetector detector;

    /**
     * Creates the extractor loading the model from app assets.
     *
     * @param context   application context
     * @param modelPath asset path, e.g. "mobilenet_ssd.tflite"
     */
    public SsdExtractor(Context context, String modelPath) throws Exception {
        BaseOptions baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .build();

        ObjectDetector.ObjectDetectorOptions options =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setMaxResults(1)
                        .setScoreThreshold(SCORE_THRESHOLD)
                        .setRunningMode(RunningMode.IMAGE)
                        .build();

        detector = ObjectDetector.createFromOptions(context, options);
        Log.d(TAG, "ObjectDetector ready — model: " + modelPath);
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

    /** Releases native resources. */
    public void close() {
        try {
            detector.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing detector", e);
        }
    }
}
