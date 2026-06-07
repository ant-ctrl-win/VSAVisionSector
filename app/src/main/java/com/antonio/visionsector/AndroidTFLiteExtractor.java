package com.antonio.visionsector;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class AndroidTFLiteExtractor {

    public static final int CNN_FEATURE_SIZE = 1280;
    private static final int INPUT_IMAGE_SIZE = 224;
    private static final int BYTES_PER_CHANNEL = 4; // float32
    private static final int CHANNELS_PER_PIXEL = 3; // RGB

    private Interpreter interpreter;

    public AndroidTFLiteExtractor(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Runtime.getRuntime().availableProcessors());

        // Carica il modello direttamente dalla cartella assets
        this.interpreter = new Interpreter(loadModelFromAssets(context), options);
    }

    private MappedByteBuffer loadModelFromAssets(Context context) throws IOException {
        // Legge direttamente il file .tflite posizionato in src/main/assets
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd("mobilenet_vsa.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public float[] extractFeatures(Bitmap bitmap) {
        Bitmap resizedBitmap;
        if (bitmap.getWidth() != INPUT_IMAGE_SIZE || bitmap.getHeight() != INPUT_IMAGE_SIZE) {
            resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        } else {
            resizedBitmap = bitmap;
        }

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                1 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * CHANNELS_PER_PIXEL * BYTES_PER_CHANNEL
        );
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        inputBuffer.rewind();
        for (int pixelValue : intValues) {
            float r = ((pixelValue >> 16) & 0xFF);
            float g = ((pixelValue >> 8) & 0xFF);
            float b = (pixelValue & 0xFF);

            inputBuffer.putFloat((r / 127.5f) - 1.0f);
            inputBuffer.putFloat((g / 127.5f) - 1.0f);
            inputBuffer.putFloat((b / 127.5f) - 1.0f);
        }

        float[][] outputFeatures = new float[1][CNN_FEATURE_SIZE];
        interpreter.run(inputBuffer, outputFeatures);

        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle();
        }

        return outputFeatures[0];
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}