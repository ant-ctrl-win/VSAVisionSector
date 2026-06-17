package com.antonio.visionsector;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Encoder MobileNetV2 per estrazione feature globali.
 * Input:  224x224 float32 [-1.0, 1.0]
 * Output: float[1280]
 */
public class FeatureEncoder {

    private static final int INPUT_SIZE = 224;
    private static final int FEATURE_DIM = 1280;

    private final Interpreter tflite;

    public FeatureEncoder(Context context, String modelPath) throws Exception {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        tflite = new Interpreter(loadModelFile(context, modelPath), options);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
        android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel ch = is.getChannel();
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    /**
     * Estrae il vettore float[1280] da un crop quadrato (già letterboxato).
     */
    public float[] extract(Bitmap crop) {
        Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        resized.recycle();

        for (int pixel : pixels) {
            inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f);
            inputBuffer.putFloat((((pixel >> 8) & 0xFF) - 127.5f) / 127.5f);
            inputBuffer.putFloat(((pixel & 0xFF) - 127.5f) / 127.5f);
        }

        float[][] output = new float[1][FEATURE_DIM];
        tflite.run(inputBuffer, output);

        return output[0];
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
}
