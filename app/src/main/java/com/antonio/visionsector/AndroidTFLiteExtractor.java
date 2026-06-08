package com.antonio.visionsector;
import android.content.Context;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class AndroidTFLiteExtractor {
    private Interpreter tflite;
    private ByteBuffer imgData;

    // NUOVO BUFFER SPAZIALE: 7 * 7 * 1280 = 62720 float
    private static final int GRID_SIZE = 7;
    private static final int FEATURE_DIM = 1280;
    private static final int TOTAL_FEATURES = GRID_SIZE * GRID_SIZE * FEATURE_DIM;

    // Buffer per ricevere i dati grezzi dalla CNN
    private ByteBuffer outputBuffer;

    // Array riutilizzabile per passare i dati alla MainActivity senza innescare il Garbage Collector
    private float[] spatialFeaturesArray;

    public AndroidTFLiteExtractor(Context context, String modelPath) throws Exception {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4); // Sfruttiamo i core del J3

        tflite = new Interpreter(loadModelFile(context, modelPath), options);

        // Input: 1 immagine RGB da 224x224 (224 * 224 * 3 * 4 bytes per float)
        imgData = ByteBuffer.allocateDirect(224 * 224 * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());

        // Output: Buffer per 62.720 float (62720 * 4 bytes)
        outputBuffer = ByteBuffer.allocateDirect(TOTAL_FEATURES * 4);
        outputBuffer.order(ByteOrder.nativeOrder());

        spatialFeaturesArray = new float[TOTAL_FEATURES];
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
        android.content.res.AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }

    /**
     * Estrae l'intera griglia spaziale. Restituisce un array piatto di 62.720 elementi.
     * La struttura logica è: [Cella 1 (1280)], [Cella 2 (1280)], ..., [Cella 49 (1280)]
     */
    public float[] extractSpatialFeatures(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        convertBitmapToByteBuffer(resizedBitmap);

        // Pulisce il buffer di output prima della nuova inferenza
        outputBuffer.rewind();

        // Esegue la rete neurale!
        tflite.run(imgData, outputBuffer);

        // Copia i dati dal ByteBuffer di TensorFlow al nostro array float Java
        outputBuffer.rewind();
        outputBuffer.asFloatBuffer().get(spatialFeaturesArray);

        return spatialFeaturesArray;
    }

    /**
     * Estrazione Globale Classica: Prende un'immagine (presumibilmente già ritagliata),
     * la ridimensiona a 224x224 e la passa nella CNN per estrarre il vettore puro da 1280.
     */
    public float[] extractGlobalFeatures(Bitmap croppedBitmap) {
        // 1. Ridimensionamento per la MobileNetV2 (224x224)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, 224, 224, true);

        // 2. Preparazione del ByteBuffer (224 * 224 * 3 canali * 4 byte per float)
        java.nio.ByteBuffer inputBuffer = java.nio.ByteBuffer.allocateDirect(224 * 224 * 3 * 4);
        inputBuffer.order(java.nio.ByteOrder.nativeOrder());

        // 3. Estrazione dei pixel
        int[] intValues = new int[224 * 224];
        resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224);

        // 4. Normalizzazione nell'intervallo [-1.0, 1.0] (Standard MobileNet)
        int pixel = 0;
        for (int i = 0; i < 224; i++) {
            for (int j = 0; j < 224; j++) {
                int val = intValues[pixel++];
                // Normalizzazione: (valore_colore - 127.5) / 127.5
                inputBuffer.putFloat((((val >> 16) & 0xFF) - 127.5f) / 127.5f); // Canale Rosso
                inputBuffer.putFloat((((val >> 8) & 0xFF) - 127.5f) / 127.5f);  // Canale Verde
                inputBuffer.putFloat(((val & 0xFF) - 127.5f) / 127.5f);         // Canale Blu
            }
        }

        // 5. Esecuzione dell'Inferenza TFLite
        // La rete classica restituisce un array bidimensionale [1][1280]
        float[][] outputArray = new float[1][1280];
        tflite.run(inputBuffer, outputArray);

        // 6. Restituiamo il vettore piatto
        return outputArray[0];
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) return;
        imgData.rewind();
        int[] intValues = new int[224 * 224];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                final int val = intValues[pixel++];
                // Normalizzazione [-1, 1] tipica della MobileNetV2
                imgData.putFloat((((val >> 16) & 0xFF) - 127.5f) / 127.5f);
                imgData.putFloat((((val >> 8) & 0xFF) - 127.5f) / 127.5f);
                imgData.putFloat(((val & 0xFF) - 127.5f) / 127.5f);
            }
        }
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }
}