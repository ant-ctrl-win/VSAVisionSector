package com.antonio.visionsector;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "VSA_TwoStage";
    private static final int CAMERA_PERMISSION_CODE = 101;
    private static final String USER_ARCHETYPES_FILE = "archetype_map.bin";
    private static final int FEATURE_DIM = 1280;
    private static final int VSA_DIM = 10000;

    private SsdExtractor ssdExtractor;
    private FeatureEncoder encoder;
    private RandomProjectionBridge vsaBridge;
    private float[] blackPaddingSignature;

    private Map<String, double[]> archetypesMAP;
    private Map<String, VsaVector> archetypesBSC;

    private double[] continuousBuffer;
    private long[] bscBlocksBuffer;
    private VsaVector reusableQueryVector;

    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private PreviewView viewFinder;
    private TextView tvVerdict;
    private TextView tvConfidence;
    private FloatingActionButton fabLearn;
    private TrackingOverlayView trackingOverlay;

    private boolean isProcessingFrame = false;
    private boolean captureNextFrameForLearning = false;
    private String pendingLearningName = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFinder = findViewById(R.id.viewFinder);
        tvVerdict = findViewById(R.id.tvVerdict);
        tvConfidence = findViewById(R.id.tvConfidence);
        fabLearn = findViewById(R.id.fabLearn);
        trackingOverlay = findViewById(R.id.trackingOverlay);

        fabLearn.setEnabled(false);
        fabLearn.setOnClickListener(v -> showNameInputDialog());
        fabLearn.setOnLongClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setTitle("Pulisci Archetipi")
                    .setMessage("Cancellare tutti gli oggetti appresi?")
                    .setPositiveButton("Si, cancella tutto", (d, w) -> {
                        inferenceExecutor.execute(() -> {
                            clearArchetypes();
                            runOnUiThread(() -> {
                                Toast.makeText(this, "Archetipi cancellati", Toast.LENGTH_SHORT).show();
                                tvVerdict.setText("Tabula Rasa. Premi + per imparare.");
                                trackingOverlay.updateDetections(Collections.emptyList(), 1, 1);
                            });
                        });
                    })
                    .setNegativeButton("No", null)
                    .show();
            return true;
        });

        inferenceExecutor.execute(() -> {
            try {
                long start = System.currentTimeMillis();

                ssdExtractor = new SsdExtractor(this, "mobilenet_ssd.tflite");
                encoder = new FeatureEncoder(this, "mobilenet_vsa.tflite");

                Bitmap black = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
                new Canvas(black).drawColor(Color.BLACK);
                blackPaddingSignature = encoder.extract(black);
                black.recycle();

                vsaBridge = new RandomProjectionBridge(FEATURE_DIM, 42L);

                archetypesMAP = new HashMap<>();
                archetypesBSC = new HashMap<>();

                continuousBuffer = new double[VSA_DIM];
                bscBlocksBuffer = new long[157];
                reusableQueryVector = new VsaVector(new long[157]);

                java.io.File oldFile = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
                if (oldFile.exists()) {
                    oldFile.delete();
                    Log.d(TAG, "Archetipi precedenti cancellati — test pulito");
                }

                loadAllArchetypesMAP();
                if (!archetypesMAP.isEmpty()) {
                    for (Map.Entry<String, double[]> e : archetypesMAP.entrySet()) {
                        archetypesBSC.put(e.getKey(), binarize(e.getValue()));
                    }
                    Log.d(TAG, "Motore Two-Stage avviato con " + archetypesMAP.size() + " archetipi");
                } else {
                    Log.d(TAG, "Motore Two-Stage avviato vuoto (Tabula Rasa)");
                }

                long dur = System.currentTimeMillis() - start;
                Log.d(TAG, "Modelli caricati in " + dur + " ms");

                runOnUiThread(() -> {
                    fabLearn.setEnabled(true);
                    checkCameraPermissionAndStart();
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore caricamento modelli", e);
                runOnUiThread(() -> tvVerdict.setText("Errore modelli. Controlla Logcat."));
            }
        });
    }

    private void checkCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            tvVerdict.setText("Permesso Fotocamera Negato.");
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(inferenceExecutor, imageProxy -> {
                    if (isProcessingFrame) {
                        imageProxy.close();
                        return;
                    }
                    isProcessingFrame = true;
                    try {
                        Bitmap frame = imageProxy.toBitmap();
                        int rotation = imageProxy.getImageInfo().getRotationDegrees();

                        if (captureNextFrameForLearning) {
                            captureNextFrameForLearning = false;
                            runOnUiThread(() -> triggerOneShotLearningFlow(frame, rotation));
                        } else if (!archetypesBSC.isEmpty()) {
                            processCameraFrame(frame, rotation);
                        }

                    } catch (Exception e) {
                        Log.e(TAG, "Errore analisi frame", e);
                    } finally {
                        imageProxy.close();
                        isProcessingFrame = false;
                    }
                });

                CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
                provider.unbindAll();
                provider.bindToLifecycle(this, selector, preview, analysis);

                tvVerdict.setText("Inquadra un oggetto...");
                tvConfidence.setText("Two-Stage Cascade attivo");

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Errore CameraX", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void processCameraFrame(Bitmap frame, int rotationDegrees) {
        try {
            long t0 = System.currentTimeMillis();

            RectF boxNorm = ssdExtractor.detectObject(frame);
            if (boxNorm == null) {
                runOnUiThread(() -> {
                    tvVerdict.setText("Ricerca in corso...");
                    trackingOverlay.updateDetections(Collections.emptyList(), 1, 1);
                });
                return;
            }

            Log.d(TAG, "SSD box norm=[" + String.format("%.2f", boxNorm.left) + "," +
                    String.format("%.2f", boxNorm.top) + " " + String.format("%.2f", boxNorm.right) +
                    "x" + String.format("%.2f", boxNorm.bottom) + "] rot=" + rotationDegrees);

            int origW = frame.getWidth();
            int origH = frame.getHeight();
            RectF boxPx = new RectF(
                    boxNorm.left * origW,
                    boxNorm.top * origH,
                    boxNorm.right * origW,
                    boxNorm.bottom * origH
            );

            if (boxPx.width() < 20 || boxPx.height() < 20) return;

            Bitmap square = letterboxCrop(frame, boxPx);
            if (square == null) return;

            long tCNN = System.currentTimeMillis();
            float[] features = encoder.extract(square);
            square.recycle();
            long timeCNN = System.currentTimeMillis() - tCNN;

            applyDebinding(features, boxPx);

            long tVSA = System.currentTimeMillis();
            vsaBridge.projectFeatureVector(features, continuousBuffer);
            binarizeToBlocks(continuousBuffer, bscBlocksBuffer);
            reusableQueryVector.setBlocks(bscBlocksBuffer);

            String bestMatch = null;
            double bestSim = 0.35;
            StringBuilder diag = new StringBuilder("VSA sim: ");

            for (Map.Entry<String, VsaVector> entry : archetypesBSC.entrySet()) {
                double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
                diag.append(entry.getKey()).append("=").append(String.format("%.3f", sim)).append(" ");
                if (sim > bestSim) {
                    bestSim = sim;
                    bestMatch = entry.getKey();
                }
            }
            Log.d(TAG, diag.toString());

            long timeVSA = System.currentTimeMillis() - tVSA;
            long total = System.currentTimeMillis() - t0;

            String verdict = (bestMatch != null)
                    ? bestMatch.replace("_", " ") + " (" + String.format("%.2f", bestSim) + ")"
                    : "Nessun target";
            String conf = "SSD+" + timeCNN + "ms VSA:" + timeVSA + "ms TOT:" + total + "ms";
            Log.d(TAG, "Box=[" + (int)boxPx.left + "," + (int)boxPx.top + " " + (int)boxPx.width() + "x" + (int)boxPx.height() + "] match=" + bestMatch + " sim=" + String.format("%.3f", bestSim));

            final String fVerdict = verdict;
            final String fConf = conf;
            final int fRot = rotationDegrees;
            final String fMatch = bestMatch;
            final double fSim = bestSim;

            runOnUiThread(() -> {
                tvVerdict.setText(fVerdict);
                tvConfidence.setText(fConf);
                if (fMatch != null) {
                    RectF rotBox = rotateBoxForDisplay(boxPx, origW, origH, fRot);
                    int dispW = (fRot == 90 || fRot == 270) ? origH : origW;
                    int dispH = (fRot == 90 || fRot == 270) ? origW : origH;
                    trackingOverlay.updateDetections(
                            Collections.singletonList(new TrackingOverlayView.DetectedItem(
                                    fMatch.replace("_", " "),
                                    (int) rotBox.left, (int) rotBox.top,
                                    (int) rotBox.right, (int) rotBox.bottom, fSim)),
                            dispW, dispH);
                } else {
                    trackingOverlay.updateDetections(Collections.emptyList(), origW, origH);
                }
            });

            frame.recycle();

        } catch (Exception e) {
            Log.e(TAG, "Errore pipeline two-stage", e);
        }
    }

    private RectF rotateBoxForDisplay(RectF box, int frameW, int frameH, int rotationDegrees) {
        switch (rotationDegrees) {
            case 90:
                return new RectF(frameH - box.bottom, box.left, frameH - box.top, box.right);
            case 180:
                return new RectF(frameW - box.right, frameH - box.bottom, frameW - box.left, frameH - box.top);
            case 270:
                return new RectF(box.top, frameW - box.right, box.bottom, frameW - box.left);
            default:
                return new RectF(box);
        }
    }

    private Bitmap letterboxCrop(Bitmap source, RectF box) {
        int left = Math.max(0, (int) box.left);
        int top = Math.max(0, (int) box.top);
        int right = Math.min(source.getWidth(), (int) box.right);
        int bottom = Math.min(source.getHeight(), (int) box.bottom);

        int cropW = right - left;
        int cropH = bottom - top;
        if (cropW <= 0 || cropH <= 0) return null;

        Bitmap crop = Bitmap.createBitmap(source, left, top, cropW, cropH);

        int squareSize = Math.max(cropW, cropH);
        Bitmap square = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(square);
        canvas.drawColor(Color.BLACK);

        int offsetX = (squareSize - cropW) / 2;
        int offsetY = (squareSize - cropH) / 2;
        canvas.drawBitmap(crop, offsetX, offsetY, null);

        crop.recycle();
        return square;
    }

    private void applyDebinding(float[] features, RectF box) {
        if (blackPaddingSignature == null) return;
        float objArea = box.width() * box.height();
        float maxSide = Math.max(box.width(), box.height());
        float sqArea = maxSide * maxSide;
        if (sqArea <= 0) return;
        float paddingRatio = 1.0f - (objArea / sqArea);
        if (paddingRatio > 0.10f) {
            float beforeMean = 0;
            for (int i = 0; i < FEATURE_DIM; i++) beforeMean += features[i];
            beforeMean /= FEATURE_DIM;

            for (int i = 0; i < FEATURE_DIM; i++) {
                features[i] = Math.max(0.0f, features[i] - (blackPaddingSignature[i] * paddingRatio));
            }

            float afterMean = 0;
            for (int i = 0; i < FEATURE_DIM; i++) afterMean += features[i];
            afterMean /= FEATURE_DIM;

            Log.d(TAG, "Debind: padding=" + String.format("%.1f", paddingRatio*100) +
                    "% mean " + String.format("%.4f", beforeMean) + "→" + String.format("%.4f", afterMean));
        }
    }

    private void binarizeToBlocks(double[] continuous, long[] blocks) {
        java.util.Arrays.fill(blocks, 0L);
        for (int d = 0; d < VSA_DIM; d++) {
            if (continuous[d] > 0) {
                blocks[d / 64] |= (1L << (d % 64));
            }
        }
    }

    private VsaVector binarize(double[] continuous) {
        long[] blocks = new long[157];
        binarizeToBlocks(continuous, blocks);
        return new VsaVector(blocks);
    }

    private void showNameInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("One-Shot Learning");
        builder.setMessage("Inserisci il nome del nuovo oggetto:");

        final EditText input = new EditText(this);
        input.setHint("Es. Vite_M5");
        builder.setView(input);

        builder.setPositiveButton("Cattura e Impara", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (!name.isEmpty()) {
                pendingLearningName = name;
                captureNextFrameForLearning = true;
            } else {
                Toast.makeText(MainActivity.this, "Il nome non puo essere vuoto", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("Annulla", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private void triggerOneShotLearningFlow(Bitmap sample, int rotationDegrees) {
        tvVerdict.setText("Apprendimento in corso...");
        tvConfidence.setText("Two-Stage Learning...");
        learnNewCustomObject(pendingLearningName, sample, rotationDegrees);
    }

    private void learnNewCustomObject(String className, Bitmap sampleFrame, int rotationDegrees) {
        inferenceExecutor.execute(() -> {
            try {
                long start = System.currentTimeMillis();

                RectF boxNorm = ssdExtractor.detectObject(sampleFrame);
                if (boxNorm == null) {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this,
                            "Nessun oggetto rilevato. Inquadra meglio.", Toast.LENGTH_SHORT).show());
                    return;
                }

                int origW = sampleFrame.getWidth();
                int origH = sampleFrame.getHeight();
                RectF boxPx = new RectF(
                        boxNorm.left * origW,
                        boxNorm.top * origH,
                        boxNorm.right * origW,
                        boxNorm.bottom * origH
                );
                if (boxPx.width() < 20 || boxPx.height() < 20) {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this,
                            "Oggetto troppo piccolo.", Toast.LENGTH_SHORT).show());
                    return;
                }

                Bitmap square = letterboxCrop(sampleFrame, boxPx);
                if (square == null) return;

                float[] features = encoder.extract(square);
                square.recycle();

                applyDebinding(features, boxPx);

                double[] mapVec = new double[VSA_DIM];
                vsaBridge.projectFeatureVector(features, mapVec);
                vsaBridge.normalizeL2(mapVec);

                archetypesMAP.put(className, mapVec);
                archetypesBSC.put(className, binarize(mapVec));
                saveAllArchetypesMAP();

                long duration = System.currentTimeMillis() - start;
                Log.d(TAG, "Appreso '" + className + "' in " + duration + " ms box=" +
                        (int)boxPx.width() + "x" + (int)boxPx.height() +
                        " padRatio=" + String.format("%.0f%%",
                        100f*(1f-(boxPx.width()*boxPx.height())/
                        (Math.max(boxPx.width(),boxPx.height())*Math.max(boxPx.width(),boxPx.height())))));

                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "Appreso: " + className, Toast.LENGTH_SHORT).show();
                    tvVerdict.setText(className.replace("_", " "));
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore apprendimento two-stage", e);
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "Errore: " + e.getMessage(), Toast.LENGTH_SHORT).show());
            }
        });
    }

    private synchronized void saveAllArchetypesMAP() {
        try (java.io.FileOutputStream fos = openFileOutput(USER_ARCHETYPES_FILE, Context.MODE_PRIVATE);
             java.io.DataOutputStream dos = new java.io.DataOutputStream(fos)) {

            dos.writeInt(archetypesMAP.size());
            dos.writeInt(VSA_DIM);

            for (Map.Entry<String, double[]> entry : archetypesMAP.entrySet()) {
                dos.writeUTF(entry.getKey());
                for (double v : entry.getValue()) dos.writeDouble(v);
            }
        } catch (Exception e) {
            Log.e(TAG, "Errore salvataggio MAP", e);
        }
    }

    private void loadAllArchetypesMAP() {
        java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
        if (!file.exists()) return;

        try (java.io.FileInputStream fis = new java.io.FileInputStream(file);
             java.io.DataInputStream dis = new java.io.DataInputStream(fis)) {

            int numClasses = dis.readInt();
            int dims = dis.readInt();
            if (dims != VSA_DIM) {
                Log.w(TAG, "Dimensione MAP incompatibile, cancello");
                file.delete();
                return;
            }

            for (int i = 0; i < numClasses; i++) {
                String name = dis.readUTF();
                double[] vec = new double[VSA_DIM];
                for (int d = 0; d < VSA_DIM; d++) vec[d] = dis.readDouble();
                archetypesMAP.put(name, vec);
            }
        } catch (Exception e) {
            Log.e(TAG, "Errore caricamento MAP, cancello", e);
            file.delete();
        }
    }

    private synchronized void clearArchetypes() {
        archetypesMAP.clear();
        archetypesBSC.clear();
        java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
        if (file.exists()) file.delete();
        Log.d(TAG, "Archetipi cancellati — pronti per nuovo training");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (ssdExtractor != null) ssdExtractor.close();
        if (encoder != null) encoder.close();
        inferenceExecutor.shutdown();
    }
}
