package com.antonio.visionsector.ui;

import com.antonio.visionsector.detector.SsdExtractor;
import com.antonio.visionsector.encoder.FeatureEncoder;
import com.antonio.visionsector.engine.vsa.RandomProjectionBridge;
import com.antonio.visionsector.engine.vsa.VsaVector;
import com.antonio.visionsector.engine.learning.LearningSessionManager;

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
import android.widget.ProgressBar;
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

import com.antonio.visionsector.R;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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

    private LearningSessionManager learningSessionManager;
    private DetectionStabilizer stabilizer = new DetectionStabilizer();
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
    private ProgressBar progressLearning;
    private FloatingActionButton fabLearn;
    private TrackingOverlayView trackingOverlay;

    private boolean isProcessingFrame = false;
    private String gpuConfigLabel = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        learningSessionManager = new LearningSessionManager(VSA_DIM, 8);

        viewFinder = findViewById(R.id.viewFinder);
        tvVerdict = findViewById(R.id.tvVerdict);
        tvConfidence = findViewById(R.id.tvConfidence);
        progressLearning = findViewById(R.id.progressLearning);
        fabLearn = findViewById(R.id.fabLearn);
        trackingOverlay = findViewById(R.id.trackingOverlay);

        if (progressLearning != null) {
            progressLearning.setMax(100);
            progressLearning.setVisibility(android.view.View.GONE);
        }

        fabLearn.setEnabled(false);
        fabLearn.setOnClickListener(v -> {
            LearningSessionManager.State st = learningSessionManager.getCurrentState();
            if (st == LearningSessionManager.State.COLLECTING) {
                inferenceExecutor.execute(() -> finishLearningEarly());
            } else {
                showNameInputDialog();
            }
        });
        fabLearn.setOnLongClickListener(v -> {
            LearningSessionManager.State st = learningSessionManager.getCurrentState();
            if (st == LearningSessionManager.State.COLLECTING) {
                inferenceExecutor.execute(() -> cancelLearning());
                return true;
            }
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

                gpuConfigLabel = "SSD:" + (ssdExtractor.isUsingGpu() ? "GPU" : "CPU")
                        + " Enc:" + (encoder.isUsingGpu() ? "GPU" : "CPU");
                Log.i(TAG, "Configurazione delegate: " + gpuConfigLabel);

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
                    tvConfidence.setText(gpuConfigLabel);
                    checkCameraPermissionAndStart();
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore caricamento modelli", e);
                runOnUiThread(() -> tvVerdict.setText("Errore modelli. Controlla Logcat."));
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // Permessi e Camera
    // ═══════════════════════════════════════════════════════════════════

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
                        if (frame != null) {
                            processCameraFrame(frame, imageProxy.getImageInfo().getRotationDegrees());
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
                tvConfidence.setText(gpuConfigLabel);

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Errore CameraX", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pipeline Two‑Stage (inferenza e apprendimento)
    // ═══════════════════════════════════════════════════════════════════

    private void processCameraFrame(Bitmap frame, int rotationDegrees) {
        try {
            long t0 = System.currentTimeMillis();
            int origW = frame.getWidth();
            int origH = frame.getHeight();

            RectF boxNorm = ssdExtractor.detectObject(frame);
            String bestMatch = null;
            double bestSim = 0.35;
            long timeCNN = 0;
            long timeVSA = 0;

            if (boxNorm != null && isBoxSane(boxNorm)) {
                RectF cropBoxNorm = expandBox(boxNorm, 0.20f);

                RectF boxPx = new RectF(
                        cropBoxNorm.left * origW,
                        cropBoxNorm.top * origH,
                        cropBoxNorm.right * origW,
                        cropBoxNorm.bottom * origH
                );

                if (boxPx.width() >= 20 && boxPx.height() >= 20) {
                    Bitmap square = letterboxCrop(frame, boxPx);
                    if (square != null) {
                        long tCNN = System.currentTimeMillis();
                        float[] features = encoder.extract(square);
                        square.recycle();
                        timeCNN = System.currentTimeMillis() - tCNN;

                        applyDebinding(features, boxPx);

                        long tVSA = System.currentTimeMillis();
                        vsaBridge.projectFeatureVector(features, continuousBuffer);

                        // ── BIVIO: APPRENDIMENTO vs INFERENZA ──────────
                        if (learningSessionManager.getCurrentState() == LearningSessionManager.State.COLLECTING) {

                            vsaBridge.normalizeL2(continuousBuffer);

                            boolean isDone = learningSessionManager.addFrame(continuousBuffer);
                            int collected = learningSessionManager.getFramesCollected();
                            int target = learningSessionManager.getTargetFrames();
                            String pName = learningSessionManager.getPendingClassName();
                            float prog = learningSessionManager.getProgress();

                            runOnUiThread(() -> {
                                if (progressLearning != null) {
                                    progressLearning.setVisibility(android.view.View.VISIBLE);
                                    progressLearning.setProgress((int)(prog * 100));
                                }
                                tvVerdict.setText("Learning: " + pName);
                                tvConfidence.setText("Shot " + collected + "/" + target + " — muovi leggermente");
                                trackingOverlay.updateDetections(Collections.emptyList(), 1, 1);
                            });

                            if (isDone) {
                                double[] finalBundle = learningSessionManager.getFinalBundleAndReset();
                                saveBundleAsArchetype(pName, finalBundle);
                            }

                            frame.recycle();
                            return;

                        } else {
                            // ── INFERENZA STANDARD ─────────────────────
                            if (!archetypesBSC.isEmpty()) {
                                binarizeToBlocks(continuousBuffer, bscBlocksBuffer);
                                reusableQueryVector.setBlocks(bscBlocksBuffer);

                                for (Map.Entry<String, VsaVector> entry : archetypesBSC.entrySet()) {
                                    double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
                                    if (sim > bestSim) {
                                        bestSim = sim;
                                        bestMatch = entry.getKey();
                                    }
                                }
                                timeVSA = System.currentTimeMillis() - tVSA;
                            }
                        }
                    }
                }
            } else if (learningSessionManager.getCurrentState() == LearningSessionManager.State.COLLECTING) {
                runOnUiThread(() -> tvConfidence.setText("Oggetto perso — inquadralo di nuovo"));
            }

            long total = System.currentTimeMillis() - t0;

            // ── STABILIZZAZIONE E UI (solo se non in collecting) ───────
            if (learningSessionManager.getCurrentState() != LearningSessionManager.State.COLLECTING) {
                DetectionStabilizer.StabilizedResult result = stabilizer.update(boxNorm, bestMatch);

                final String fVerdict;
                final String fConf;
                final List<TrackingOverlayView.DetectedItem> fItems;

                // 0. Calcoliamo le dimensioni del frame ROTATO (Ecco le variabili che mancavano!)
                final int rotatedFrameW = (rotationDegrees == 90 || rotationDegrees == 270) ? origH : origW;
                final int rotatedFrameH = (rotationDegrees == 90 || rotationDegrees == 270) ? origW : origH;

                if (result.isVisible && result.box != null && result.className != null) {
                    fVerdict = result.className.replace("_", " ") + " (" + String.format("%.2f", bestSim) + ")";
                    fConf = gpuConfigLabel + " | SSD+" + timeCNN + "ms VSA:" + timeVSA + "ms TOT:" + total + "ms";

                    // 1. Convertiamo in pixel sul frame ORIGINALE
                    RectF rawPxBox = new RectF(
                            result.box.left * origW,
                            result.box.top * origH,
                            result.box.right * origW,
                            result.box.bottom * origH
                    );

                    // 2. Ruotiamo il box
                    RectF rotBox = rotateBoxForDisplay(rawPxBox, origW, origH, rotationDegrees);

                    // 3. LA GEOMETRIA DEL FILL_CENTER
                    int viewW = viewFinder.getWidth();
                    int viewH = viewFinder.getHeight();

                    float scaleX = (float) viewW / rotatedFrameW;
                    float scaleY = (float) viewH / rotatedFrameH;

                    // FILL_CENTER scala usando il lato che richiede l'ingrandimento maggiore
                    float maxScale = Math.max(scaleX, scaleY);

                    // Calcoliamo le dimensioni del frame della telecamera una volta scalato
                    float scaledFrameW = rotatedFrameW * maxScale;
                    float scaledFrameH = rotatedFrameH * maxScale;

                    // Calcoliamo l'offset (quanto frame viene tagliato fuori dallo schermo)
                    float offsetX = (scaledFrameW - viewW) / 2f;
                    float offsetY = (scaledFrameH - viewH) / 2f;

                    // 4. Applichiamo la scala e l'offset al box ruotato
                    RectF finalUiBox = new RectF(
                            (rotBox.left * maxScale) - offsetX,
                            (rotBox.top * maxScale) - offsetY,
                            (rotBox.right * maxScale) - offsetX,
                            (rotBox.bottom * maxScale) - offsetY
                    );

                    // Inoltriamo i pixel esatti e calcolati all'Overlay
                    fItems = Collections.singletonList(new TrackingOverlayView.DetectedItem(
                            result.className.replace("_", " "),
                            finalUiBox.left, finalUiBox.top,
                            finalUiBox.right, finalUiBox.bottom, bestSim));

                } else if (archetypesBSC.isEmpty()) {
                    fVerdict = "Tabula Rasa. Premi + per imparare.";
                    fConf = gpuConfigLabel;
                    fItems = Collections.emptyList();
                } else {
                    fVerdict = "Ricerca in corso...";
                    fConf = gpuConfigLabel + " | SSD+" + timeCNN + "ms TOT:" + total + "ms";
                    fItems = Collections.emptyList();
                }

                runOnUiThread(() -> {
                    tvVerdict.setText(fVerdict);
                    tvConfidence.setText(fConf);
                    trackingOverlay.updateDetections(fItems, viewFinder.getWidth(), viewFinder.getHeight());
                });
            }

            frame.recycle();

        } catch (Exception e) {
            Log.e(TAG, "Errore pipeline two-stage", e);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Multi‑shot Learning
    // ═══════════════════════════════════════════════════════════════════

    private void showNameInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Multi‑Shot Learning");
        builder.setMessage("Inserisci il nome del nuovo oggetto e ruotalo lentamente:");

        final EditText input = new EditText(this);
        input.setHint("Es. Vite_M5");
        builder.setView(input);

        builder.setPositiveButton("Avvia Acquisizione", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (!name.isEmpty()) {
                learningSessionManager.startLearning(name);
                runOnUiThread(() -> {
                    if (progressLearning != null) {
                        progressLearning.setVisibility(android.view.View.VISIBLE);
                        progressLearning.setProgress(0);
                    }
                    fabLearn.setImageResource(android.R.drawable.ic_media_pause);
                    Toast.makeText(MainActivity.this,
                            "Inquadra l'oggetto. Premi + per terminare.", Toast.LENGTH_LONG).show();
                });
            } else {
                Toast.makeText(MainActivity.this, "Il nome non può essere vuoto", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("Annulla", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private void finishLearningEarly() {
        double[] finalBundle = learningSessionManager.finish();
        if (finalBundle != null) {
            saveBundleAsArchetype(learningSessionManager.getPendingClassName(), finalBundle);
        } else {
            runOnUiThread(() -> {
                if (progressLearning != null) progressLearning.setVisibility(android.view.View.GONE);
                fabLearn.setImageResource(android.R.drawable.ic_input_add);
                Toast.makeText(MainActivity.this, "Nessun sample raccolto", Toast.LENGTH_SHORT).show();
                tvVerdict.setText("Inquadra un oggetto...");
                tvConfidence.setText(gpuConfigLabel);
            });
        }
    }

    private void cancelLearning() {
        learningSessionManager.cancel();
        runOnUiThread(() -> {
            if (progressLearning != null) progressLearning.setVisibility(android.view.View.GONE);
            fabLearn.setImageResource(android.R.drawable.ic_input_add);
            Toast.makeText(MainActivity.this, "Apprendimento annullato", Toast.LENGTH_SHORT).show();
            tvVerdict.setText("Inquadra un oggetto...");
            tvConfidence.setText(gpuConfigLabel);
        });
    }

    private void saveBundleAsArchetype(String className, double[] bundle) {
        if (className == null || bundle == null) return;

        vsaBridge.normalizeL2(bundle);

        archetypesMAP.put(className, bundle);
        VsaVector archetypeBSC = binarize(bundle);
        archetypesBSC.put(className, archetypeBSC);
        saveAllArchetypesMAP();

        int activeBits = 0;
        for (int i = 0; i < 157; i++) {
            activeBits += Long.bitCount(archetypeBSC.getBlocks()[i]);
        }
        Log.i(TAG, "ARCHETIPO CREATO: " + className + " | Bit attivi: " + activeBits + "/10000");

        runOnUiThread(() -> {
            if (progressLearning != null) progressLearning.setVisibility(android.view.View.GONE);
            fabLearn.setImageResource(android.R.drawable.ic_input_add);
            Toast.makeText(MainActivity.this, "Appreso: " + className, Toast.LENGTH_SHORT).show();
            tvVerdict.setText(className.replace("_", " "));
            tvConfidence.setText(gpuConfigLabel);
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // Utility geometriche e VSA
    // ═══════════════════════════════════════════════════════════════════

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

    private RectF expandBox(RectF box, float ratio) {
        float w = box.width() * ratio;
        float h = box.height() * ratio;
        return new RectF(
                Math.max(0f, box.left - w), Math.max(0f, box.top - h),
                Math.min(1f, box.right + w), Math.min(1f, box.bottom + h)
        );
    }

    private boolean isBoxSane(RectF box) {
        if (box == null) return false;
        float area = box.width() * box.height();
        if (area < 0) area = -area;
        return area >= 0.01f && area <= 0.80f;
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
            for (int i = 0; i < FEATURE_DIM; i++) {
                features[i] = Math.max(0.0f, features[i] - (blackPaddingSignature[i] * paddingRatio));
            }
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

    // ═══════════════════════════════════════════════════════════════════
    // Persistenza archetipi
    // ═══════════════════════════════════════════════════════════════════

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

    // ═══════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════

    @Override
    protected void onDestroy() {
        super.onDestroy();
        learningSessionManager.cancel();
        if (ssdExtractor != null) ssdExtractor.close();
        if (encoder != null) encoder.close();
        inferenceExecutor.shutdown();
    }
}
