package com.antonio.vsatest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.antonio.visionsector.encoder.MobileNetV4Encoder;

import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class MobileNetV4EncoderRuntimeTest {

    private static final String TAG = "MobileNetV4Runtime";
    private static final int INPUT_LENGTH = 224 * 224 * 3;
    private static final int EMBEDDING_DIM = 960;
    private static final double MAX_ALLOWED_DIFF = 1e-6;

    private static float[] buildSyntheticInput() {
        float[] input = new float[INPUT_LENGTH];
        for (int i = 0; i < input.length; i++) {
            input[i] = ((i * 31) % 256) / 255f;
        }
        return input;
    }

    private static double l2Norm(float[] values) {
        double sum = 0.0;
        for (float value : values) {
            sum += (double) value * value;
        }
        return Math.sqrt(sum);
    }

    private static float min(float[] values) {
        float result = Float.POSITIVE_INFINITY;
        for (float value : values) {
            if (value < result) {
                result = value;
            }
        }
        return result;
    }

    private static float max(float[] values) {
        float result = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            if (value > result) {
                result = value;
            }
        }
        return result;
    }

    private static double mean(float[] values) {
        double sum = 0.0;
        for (float value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    @Test
    public void encodeSyntheticInputTwice() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        float[] input = buildSyntheticInput();

        try (MobileNetV4Encoder encoder = new MobileNetV4Encoder(context)) {
            long startFirst = System.nanoTime();
            float[] first = encoder.encode(input);
            long firstInferenceNanos = System.nanoTime() - startFirst;

            long startSecond = System.nanoTime();
            float[] second = encoder.encode(input);
            long secondInferenceNanos = System.nanoTime() - startSecond;

            assertNotNull(first);
            assertNotNull(second);
            assertEquals(EMBEDDING_DIM, first.length);
            assertEquals(EMBEDDING_DIM, second.length);

            for (int i = 0; i < first.length; i++) {
                assertTrue("non-finite first output at index " + i,
                        Float.isFinite(first[i]));
            }
            for (int i = 0; i < second.length; i++) {
                assertTrue("non-finite second output at index " + i,
                        Float.isFinite(second[i]));
            }

            double norm = l2Norm(first);
            assertTrue("L2 norm must be > 0, was " + norm, norm > 0.0);

            double maxAbsDiff = 0.0;
            for (int i = 0; i < first.length; i++) {
                double diff = Math.abs((double) first[i] - (double) second[i]);
                if (diff > maxAbsDiff) {
                    maxAbsDiff = diff;
                }
            }
            assertTrue("max abs diff " + maxAbsDiff + " exceeds " + MAX_ALLOWED_DIFF,
                    maxAbsDiff <= MAX_ALLOWED_DIFF);

            String diagnostics = "outputLength=" + first.length
                    + " min=" + min(first)
                    + " max=" + max(first)
                    + " mean=" + mean(first)
                    + " l2Norm=" + norm
                    + " firstInferenceMs=" + (firstInferenceNanos / 1_000_000.0)
                    + " secondInferenceMs=" + (secondInferenceNanos / 1_000_000.0)
                    + " maxAbsDiff=" + maxAbsDiff;

            Log.i(TAG, diagnostics);
            System.out.println(TAG + " " + diagnostics);
        }
    }
}
