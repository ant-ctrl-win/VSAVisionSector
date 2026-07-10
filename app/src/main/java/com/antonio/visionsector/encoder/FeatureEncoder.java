package com.antonio.visionsector.encoder;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FeatureEncoder {

    private static final String TAG = "FeatureEncoder";

    private static final int INPUT_SIZE = 224;

    private final int featureDim;
    private Interpreter tflite;
    private GpuDelegate gpuDelegate = null;
    private boolean isUsingGpu = false;

    public FeatureEncoder(Context context, String modelPath) throws Exception {
        Interpreter.Options options = new Interpreter.Options();

        try {
            CompatibilityList compatList = new CompatibilityList();

            if (compatList.isDelegateSupportedOnThisDevice()) {
                GpuDelegate.Options gpuOptions = compatList.getBestOptionsForThisDevice();
                gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);

                gpuDelegate = new GpuDelegate(gpuOptions);
                options.addDelegate(gpuDelegate);

                isUsingGpu = true;
                Log.i(TAG, "TFLite FeatureEncoder: GPU Delegate inizializzato con successo.");
            } else {
                throw new UnsupportedOperationException("GPU non supportata dalla CompatibilityList.");
            }
        } catch (Exception e) {
            Log.w(TAG, "Fallback su CPU (4 thread). Motivo: " + e.getMessage());
            options.setNumThreads(4);
            isUsingGpu = false;
        }

        MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
        tflite = new Interpreter(modelBuffer, options);

        int[] outputShape = tflite.getOutputTensor(0).shape();
        if (outputShape.length >= 2 && outputShape[1] > 0) {
            featureDim = outputShape[1];
        } else {
            featureDim = 1280;
            Log.w(TAG, "Forma output non riconosciuta, default a " + featureDim);
        }
        Log.i(TAG, "FeatureEncoder: dimensione feature = " + featureDim);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
        AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel ch = is.getChannel();
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    public float[] extract(Bitmap crop) {
        if (crop == null) return new float[featureDim];

        Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        resized.recycle();

        for (int pixel : pixels) {
            inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f);
            inputBuffer.putFloat((((pixel >> 8)  & 0xFF) - 127.5f) / 127.5f);
            inputBuffer.putFloat((( pixel        & 0xFF) - 127.5f) / 127.5f);
        }

        float[][] output = new float[1][featureDim];
        tflite.run(inputBuffer, output);

        return output[0];
    }

    public int getFeatureDimension() {
        return featureDim;
    }

    public void close() {
        try {
            if (tflite != null) {
                tflite.close();
            }
            if (gpuDelegate != null) {
                gpuDelegate.close();
            }
            Log.d(TAG, "Risorse dell'interprete TFLite rilasciate correttamente.");
        } catch (Exception e) {
            Log.e(TAG, "Errore durante la chiusura di FeatureEncoder", e);
        }
    }

    public boolean isUsingGpu() {
        return isUsingGpu;
    }
}
