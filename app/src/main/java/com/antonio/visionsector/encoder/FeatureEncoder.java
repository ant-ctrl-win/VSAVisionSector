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
    private static final int FEATURE_DIM = 1280;

    private Interpreter tflite;
    private GpuDelegate gpuDelegate = null;

    // FIX 1: Tolto il "final" e inizializzato a false.
    // Verrà impostato a true solo se l'inizializzazione GPU va a buon fine.
    private boolean isUsingGpu = false;

    /**
     * Costruttore dell'Encoder con supporto all'accelerazione GPU hardware.
     */
    public FeatureEncoder(Context context, String modelPath) throws Exception {
        Interpreter.Options options = new Interpreter.Options();

        try {
            // Controlla se la GPU del dispositivo è compatibile e supportata
            CompatibilityList compatList = new CompatibilityList();

            if (compatList.isDelegateSupportedOnThisDevice()) {
                // Estrae le migliori impostazioni GPU per il telefono corrente
                GpuDelegate.Options gpuOptions = compatList.getBestOptionsForThisDevice();
                gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);

                // FIX 2: Creazione corretta dell'istanza GpuDelegate (senza metodi statici inventati)
                gpuDelegate = new GpuDelegate(gpuOptions);
                options.addDelegate(gpuDelegate);

                isUsingGpu = true; // Successo!
                Log.i(TAG, "TFLite FeatureEncoder: GPU Delegate inizializzato con successo.");
            } else {
                throw new UnsupportedOperationException("GPU non supportata dalla CompatibilityList.");
            }
        } catch (Exception e) {
            Log.w(TAG, "Fallback su CPU (4 thread). Motivo: " + e.getMessage());
            options.setNumThreads(4);
            isUsingGpu = false; // Assicuriamoci che il flag sia false in caso di errore
        }

        MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
        tflite = new Interpreter(modelBuffer, options);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
        AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel ch = is.getChannel();
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    /**
     * Estrae il vettore di feature [1280] da un crop quadrato.
     */
    public float[] extract(Bitmap crop) {
        if (crop == null) return new float[FEATURE_DIM];

        // 1. Ridimensionamento a 224x224 bilineare
        Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);

        // 2. Allocazione del ByteBuffer diretto (224 * 224 * 3 canali * 4 byte per float)
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        resized.recycle();

        // 3. Normalizzazione dei pixel nel range [-1.0, 1.0]
        for (int pixel : pixels) {
            inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f); // R
            inputBuffer.putFloat((((pixel >> 8)  & 0xFF) - 127.5f) / 127.5f); // G
            inputBuffer.putFloat((( pixel        & 0xFF) - 127.5f) / 127.5f); // B
        }

        // 4. Esecuzione dell'inferenza accelerata
        float[][] output = new float[1][FEATURE_DIM];
        tflite.run(inputBuffer, output);

        return output[0]; // Ritorna il vettore float[1280]
    }

    /**
     * Rilascia le risorse native per evitare Memory Leak.
     */
    public void close() {
        try {
            if (tflite != null) {
                tflite.close();
            }
            if (gpuDelegate != null) {
                gpuDelegate.close(); // È fondamentale chiudere anche la GPU
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