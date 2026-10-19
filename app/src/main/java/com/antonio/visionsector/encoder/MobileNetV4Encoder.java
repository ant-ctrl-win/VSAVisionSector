package com.antonio.visionsector.encoder;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public final class MobileNetV4Encoder implements AutoCloseable {

    private static final String MODEL_ASSET = "mobilenetv4_conv_small_encoder.tflite";

    public static final int INPUT_SIZE = 224;
    public static final int INPUT_CHANNELS = 3;
    public static final int INPUT_ELEMENT_COUNT = INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS;
    public static final int EMBEDDING_DIM = 960;

    private static final int[] EXPECTED_INPUT_SHAPE = {
            1, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS
    };
    private static final int[] EXPECTED_OUTPUT_SHAPE = {1, EMBEDDING_DIM};

    private final Interpreter interpreter;

    public MobileNetV4Encoder(Context context) throws IOException {
        if (context == null) {
            throw new IllegalArgumentException("context must not be null");
        }

        Interpreter newInterpreter = new Interpreter(loadModel(context));
        try {
            newInterpreter.allocateTensors();
            validateContract(newInterpreter);
        } catch (RuntimeException exception) {
            newInterpreter.close();
            throw exception;
        }
        this.interpreter = newInterpreter;
    }

    public float[] encode(float[] nhwcInput) {
        if (nhwcInput == null) {
            throw new IllegalArgumentException("nhwcInput must not be null");
        }
        if (nhwcInput.length != INPUT_ELEMENT_COUNT) {
            throw new IllegalArgumentException(
                    "nhwcInput must contain " + INPUT_ELEMENT_COUNT
                            + " floats (" + INPUT_SIZE + "x" + INPUT_SIZE + "x"
                            + INPUT_CHANNELS + "), got " + nhwcInput.length);
        }

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_ELEMENT_COUNT * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : nhwcInput) {
            inputBuffer.putFloat(value);
        }
        inputBuffer.rewind();

        float[][] output = new float[1][EMBEDDING_DIM];
        interpreter.run(inputBuffer, output);
        return output[0];
    }

    @Override
    public void close() {
        interpreter.close();
    }

    private static MappedByteBuffer loadModel(Context context) throws IOException {
        try (AssetFileDescriptor descriptor = context.getAssets().openFd(MODEL_ASSET);
             FileInputStream stream = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = stream.getChannel()) {
            return channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(),
                    descriptor.getDeclaredLength());
        }
    }

    private static void validateContract(Interpreter interpreter) {
        if (interpreter.getInputTensorCount() != 1) {
            throw new IllegalStateException(
                    "Expected exactly 1 input tensor, got " + interpreter.getInputTensorCount());
        }
        if (interpreter.getOutputTensorCount() != 1) {
            throw new IllegalStateException(
                    "Expected exactly 1 output tensor, got " + interpreter.getOutputTensorCount());
        }

        Tensor input = interpreter.getInputTensor(0);
        Tensor output = interpreter.getOutputTensor(0);

        if (!Arrays.equals(input.shape(), EXPECTED_INPUT_SHAPE)) {
            throw new IllegalStateException(
                    "Unexpected input shape " + Arrays.toString(input.shape())
                            + ", expected " + Arrays.toString(EXPECTED_INPUT_SHAPE));
        }
        if (!Arrays.equals(output.shape(), EXPECTED_OUTPUT_SHAPE)) {
            throw new IllegalStateException(
                    "Unexpected output shape " + Arrays.toString(output.shape())
                            + ", expected " + Arrays.toString(EXPECTED_OUTPUT_SHAPE));
        }
        if (input.dataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "Unexpected input data type " + input.dataType() + ", expected FLOAT32");
        }
        if (output.dataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "Unexpected output data type " + output.dataType() + ", expected FLOAT32");
        }
    }
}
