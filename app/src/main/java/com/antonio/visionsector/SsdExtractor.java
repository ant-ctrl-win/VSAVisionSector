package com.antonio.visionsector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Wrapper per MobileNet SSD TFLite — solo localizzazione (ignora le classi).
 * Input:  uint8 [0-255], 300x300x3
 * Output: RectF normalizzato [0.0-1.0] sul frame originale, o null se nessuna detection.
 */
public class SsdExtractor {

    private static final int INPUT_SIZE = 300;
    private static final float CONFIDENCE_THRESHOLD = 0.5f;

    private final Interpreter tflite;
    private ByteBuffer inputBuffer;

    private float[][][] detectionBoxes;
    private float[][] detectionClasses;
    private float[][] detectionScores;
    private float[] numDetections;

    public SsdExtractor(Context context, String modelPath) throws Exception {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        tflite = new Interpreter(loadModelFile(context, modelPath), options);

        inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] shape0 = tflite.getOutputTensor(0).shape();
        int maxDet = shape0[1];

        detectionBoxes = new float[1][maxDet][4];
        detectionClasses = new float[1][maxDet];
        detectionScores = new float[1][maxDet];
        numDetections = new float[1];
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
        android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel ch = is.getChannel();
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    /**
     * Rileva l'oggetto con la confidenza piu alta nel frame.
     * @return RectF normalizzato [0.0-1.0] sulle dimensioni originali, o null.
     */
    public RectF detectObject(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        convertBitmapToByteBuffer(resized);
        resized.recycle();

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, detectionBoxes);
        outputs.put(1, detectionClasses);
        outputs.put(2, detectionScores);
        outputs.put(3, numDetections);
        tflite.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);

        int bestIdx = -1;
        float bestScore = CONFIDENCE_THRESHOLD;
        int num = Math.min((int) numDetections[0], detectionScores[0].length);

        for (int i = 0; i < num; i++) {
            if (detectionScores[0][i] > bestScore) {
                bestScore = detectionScores[0][i];
                bestIdx = i;
            }
        }

        if (bestIdx < 0) return null;

        float ymin = detectionBoxes[0][bestIdx][0];
        float xmin = detectionBoxes[0][bestIdx][1];
        float ymax = detectionBoxes[0][bestIdx][2];
        float xmax = detectionBoxes[0][bestIdx][3];

        return new RectF(xmin, ymin, xmax, ymax);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        inputBuffer.rewind();
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            inputBuffer.put((byte) ((pixel >> 16) & 0xFF));
            inputBuffer.put((byte) ((pixel >> 8) & 0xFF));
            inputBuffer.put((byte) (pixel & 0xFF));
        }
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
}
