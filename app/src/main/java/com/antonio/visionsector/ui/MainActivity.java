package com.antonio.visionsector.ui;

import com.antonio.visionsector.detector.SsdExtractor;
import com.antonio.visionsector.encoder.FeatureEncoder;
import com.antonio.visionsector.engine.vsa.RandomProjectionBridge;
import com.antonio.visionsector.engine.vsa.VsaVector;
import com.antonio.visionsector.engine.vsa.DatasetStatTracker;
import com.antonio.visionsector.engine.learning.LearningSessionManager;


import com.antonio.visionsector.network.RetrofitClient;
import com.antonio.visionsector.network.dto.ModuleBundleDTO;
import com.antonio.visionsector.utils.LocalStorageManager;

import android.Manifest;
import android.app.ProgressDialog;
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

import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    // Tag per identificare facilmente i nostri log nella console di Android Studio
    private static final String TAG = "VSHUB_NETWORK_TEST";
//    private static final String TAG = "VSA_TwoStage";
    private static final int CAMERA_PERMISSION_CODE = 101;
    private static final String USER_ARCHETYPES_FILE = "archetype_map.bin";
    private static final int VSA_DIM = 10000;
    private static final int FILE_MAGIC = 0x56534132;
    private static final int DEFAULT_SAMPLE_COUNT = 8;

    private int featureDim;

    private LearningSessionManager learningSessionManager;
    private DetectionStabilizer stabilizer = new DetectionStabilizer();
    private SsdExtractor ssdExtractor;
    private FeatureEncoder encoder;
    private RandomProjectionBridge vsaBridge;
    private float[] blackPaddingSignature;

    private Map<String, double[]> archetypesMAP;
    private Map<String, VsaVector> archetypesBSC;
    private Map<String, Integer> sampleCounts;

    private double[] continuousBuffer;
    private long[] bscBlocksBuffer;
    private VsaVector reusableQueryVector;

    private DatasetStatTracker statsTracker;
    private int frameCounter;

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

        testCloudConnection();

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
                featureDim = encoder.getFeatureDimension();

                gpuConfigLabel = "SSD:" + (ssdExtractor.isUsingGpu() ? "GPU" : "CPU")
                        + " Enc:" + (encoder.isUsingGpu() ? "GPU" : "CPU");
                Log.i(TAG, "Configurazione delegate: " + gpuConfigLabel);
                Log.i(TAG, "Feature dimension dal modello: " + featureDim);

                Bitmap black = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
                new Canvas(black).drawColor(Color.BLACK);
                blackPaddingSignature = encoder.extract(black);
                black.recycle();

                vsaBridge = new RandomProjectionBridge(featureDim, 42L);

                try {
                    statsTracker = DatasetStatTracker.importFromAndroidAssets(this, "vsa_stats.bin");
                    Log.i(TAG, "Statistiche Z‑Score caricate: " + statsTracker.getDimensions() + " dimensioni");
                } catch (Exception e) {
                    statsTracker = null;
                    Log.w(TAG, "vsa_stats.bin non trovato o illeggibile — Z‑Score disabilitato: " + e.getMessage());
                }

                archetypesMAP = new HashMap<>();
                archetypesBSC = new HashMap<>();
                sampleCounts = new HashMap<>();

                continuousBuffer = new double[VSA_DIM];
                bscBlocksBuffer = new long[157];
                reusableQueryVector = new VsaVector(new long[157]);

                java.io.File oldFile = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
                if (oldFile.exists()) {
                    oldFile.delete();
                    Log.d(TAG, "Archetipi precedenti cancellati — partenza pulita");
                }

                loadBuiltInArchetypes();
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

    private void processCameraFrame(Bitmap frame, int rotationDegrees) {
        try {
            long t0 = System.currentTimeMillis();
            int origW = frame.getWidth();
            int origH = frame.getHeight();

            RectF boxNorm = ssdExtractor.detectObject(frame);
            double bestSim = 0;
            String bestMatch = null;
            DetectionStabilizer.PredictionVerdict predVerdict = DetectionStabilizer.PredictionVerdict.unknown(0.0);
            long timeCNN = 0;
            long timeVSA = 0;

            if (boxNorm != null && isBoxSane(boxNorm)) {
                RectF cropBoxNorm = expandBox(boxNorm, 0.03f);

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

                        if (learningSessionManager.getCurrentState() == LearningSessionManager.State.COLLECTING) {

                            vsaBridge.projectFeatureVector(features, continuousBuffer);
                            vsaBridge.normalizeL2(continuousBuffer);
                            if (statsTracker != null) {
                                statsTracker.standardizeInPlace(continuousBuffer);
                            }

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
                                int nFrames = learningSessionManager.getFramesCollected();
                                double[] finalBundle = learningSessionManager.getFinalBundleAndReset();
                                saveBundleAsArchetype(pName, finalBundle, nFrames);
                            }

                            frame.recycle();
                            return;

                        } else {
                            if (!archetypesBSC.isEmpty()) {
                                vsaBridge.projectFeatureVector(features, continuousBuffer);
                                vsaBridge.normalizeL2(continuousBuffer);
                                if (statsTracker != null) {
                                    statsTracker.standardizeInPlace(continuousBuffer);
                                }
                                binarizeToBlocks(continuousBuffer, bscBlocksBuffer);
                                reusableQueryVector.setBlocks(bscBlocksBuffer);

                                if (++frameCounter % 100 == 0) {
                                    double pct = VsaVector.bitOnePercentage(bscBlocksBuffer);
                                    Log.i(TAG, "Bit‑1 query frame#" + frameCounter + ": " + String.format("%.1f%%", pct));
                                }

                                List<DetectionStabilizer.SimilarityResult> simResults = new ArrayList<>();
                                for (Map.Entry<String, VsaVector> entry : archetypesBSC.entrySet()) {
                                    double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
                                    simResults.add(new DetectionStabilizer.SimilarityResult(entry.getKey(), sim));
                                }
                                simResults.sort((a, b) -> Double.compare(b.similarity, a.similarity));

                                if (simResults.size() >= 2) {
                                    Log.d("VSA_DIAGNOSTIC",
                                            "Top1=" + simResults.get(0).className + "(" + String.format("%.4f", simResults.get(0).similarity) + ")"
                                                    + " Top2=" + simResults.get(1).className + "(" + String.format("%.4f", simResults.get(1).similarity) + ")"
                                                    + " Delta=" + String.format("%.4f", simResults.get(0).similarity - simResults.get(1).similarity));
                                } else if (simResults.size() == 1) {
                                    Log.d("VSA_DIAGNOSTIC",
                                            "Top1=" + simResults.get(0).className + "(" + String.format("%.4f", simResults.get(0).similarity) + ")"
                                                    + " Top2=N/A Delta=N/A");
                                }

                                predVerdict = DetectionStabilizer.evaluatePrediction(simResults, archetypesBSC.size());
                                bestSim = simResults.isEmpty() ? 0 : simResults.get(0).similarity;
                                bestMatch = predVerdict.className;
                                timeVSA = System.currentTimeMillis() - tVSA;
                            }
                        }
                    }
                }
            } else if (learningSessionManager.getCurrentState() == LearningSessionManager.State.COLLECTING) {
                runOnUiThread(() -> tvConfidence.setText("Oggetto perso — inquadralo di nuovo"));
            }

            long total = System.currentTimeMillis() - t0;

            if (learningSessionManager.getCurrentState() != LearningSessionManager.State.COLLECTING) {
                DetectionStabilizer.StabilizedResult result = stabilizer.update(boxNorm, predVerdict.verdict, bestMatch);

                final String fVerdict;
                final String fConf;
                final List<TrackingOverlayView.DetectedItem> fItems;

                final int rotatedFrameW = (rotationDegrees == 90 || rotationDegrees == 270) ? origH : origW;
                final int rotatedFrameH = (rotationDegrees == 90 || rotationDegrees == 270) ? origW : origH;

                if (result.isVisible && result.box != null) {
                    switch (result.verdict) {
                        case UNKNOWN:
                            fVerdict = "Sconosciuto";
                            break;
                        case AMBIGUOUS:
                            fVerdict = "Avvicinati / Ruota oggetto";
                            break;
                        case MATCH_FOUND:
                        default:
                            if (result.className != null) {
                                double confidence = Math.min(100.0, (predVerdict.zScore / 3.0) * 100.0);
                                fVerdict = result.className.replace("_", " ") + " (" + String.format("%.0f%%", confidence) + ")";
                            } else {
                                fVerdict = "Ricerca in corso...";
                            }
                            break;
                    }
                    fConf = gpuConfigLabel + " | SSD+" + timeCNN + "ms VSA:" + timeVSA + "ms TOT:" + total + "ms";

                    RectF rawPxBox = new RectF(
                            result.box.left * origW,
                            result.box.top * origH,
                            result.box.right * origW,
                            result.box.bottom * origH
                    );

                    RectF rotBox = rotateBoxForDisplay(rawPxBox, origW, origH, rotationDegrees);

                    int viewW = viewFinder.getWidth();
                    int viewH = viewFinder.getHeight();

                    float scaleX = (float) viewW / rotatedFrameW;
                    float scaleY = (float) viewH / rotatedFrameH;

                    float maxScale = Math.max(scaleX, scaleY);

                    float scaledFrameW = rotatedFrameW * maxScale;
                    float scaledFrameH = rotatedFrameH * maxScale;

                    float offsetX = (scaledFrameW - viewW) / 2f;
                    float offsetY = (scaledFrameH - viewH) / 2f;

                    RectF finalUiBox = new RectF(
                            (rotBox.left * maxScale) - offsetX,
                            (rotBox.top * maxScale) - offsetY,
                            (rotBox.right * maxScale) - offsetX,
                            (rotBox.bottom * maxScale) - offsetY
                    );

                    String displayClass = result.className != null ? result.className.replace("_", " ") : "?";
                    fItems = Collections.singletonList(new TrackingOverlayView.DetectedItem(
                            displayClass,
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
        String className = learningSessionManager.getPendingClassName();
        int nFrames = learningSessionManager.getFramesCollected();
        double[] finalBundle = learningSessionManager.finish();
        if (finalBundle != null) {
            saveBundleAsArchetype(className, finalBundle, nFrames);
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

    private void saveBundleAsArchetype(String className, double[] bundle, int framesCollected) {
        if (className == null || bundle == null) return;

        vsaBridge.normalizeL2(bundle);
        int nNew = Math.max(1, framesCollected);

        double[] existing = archetypesMAP.get(className);
        if (existing != null) {
            int nOld = sampleCounts.containsKey(className) ? sampleCounts.get(className) : DEFAULT_SAMPLE_COUNT;
            double[] combined = new double[VSA_DIM];
            for (int i = 0; i < VSA_DIM; i++) {
                combined[i] = existing[i] * nOld + bundle[i] * nNew;
            }
            vsaBridge.normalizeL2(combined);
            archetypesMAP.put(className, combined);
            sampleCounts.put(className, nOld + nNew);
        } else {
            archetypesMAP.put(className, bundle);
            sampleCounts.put(className, nNew);
        }

        VsaVector archetypeBSC = binarize(archetypesMAP.get(className));
        archetypesBSC.put(className, archetypeBSC);
        saveAllArchetypesMAP();

        double bitPct = archetypeBSC.getBitOnePercentage();
        int totalShots = sampleCounts.get(className);
        Log.i(TAG, "ARCHETIPO " + (existing != null ? "AGGIORNATO" : "CREATO") + ": "
                + className + " | Shot: " + totalShots + " | Bit‑1: " + String.format("%.1f%%", bitPct));

        runOnUiThread(() -> {
            if (progressLearning != null) progressLearning.setVisibility(android.view.View.GONE);
            fabLearn.setImageResource(android.R.drawable.ic_input_add);
            Toast.makeText(MainActivity.this, "Appreso: " + className, Toast.LENGTH_SHORT).show();
            tvVerdict.setText(className.replace("_", " "));
            tvConfidence.setText(gpuConfigLabel);
        });
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
            for (int i = 0; i < featureDim; i++) {
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

    private synchronized void saveAllArchetypesMAP() {
        try (java.io.FileOutputStream fos = openFileOutput(USER_ARCHETYPES_FILE, Context.MODE_PRIVATE);
             java.io.DataOutputStream dos = new java.io.DataOutputStream(fos)) {

            dos.writeInt(FILE_MAGIC);
            dos.writeInt(archetypesMAP.size());
            dos.writeInt(VSA_DIM);

            for (Map.Entry<String, double[]> entry : archetypesMAP.entrySet()) {
                dos.writeUTF(entry.getKey());
                int count = sampleCounts.containsKey(entry.getKey())
                        ? sampleCounts.get(entry.getKey()) : DEFAULT_SAMPLE_COUNT;
                dos.writeInt(count);
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

            int firstInt = dis.readInt();

            if (firstInt == FILE_MAGIC) {
                int numClasses = dis.readInt();
                int dims = dis.readInt();
                if (dims != VSA_DIM) { file.delete(); return; }
                for (int i = 0; i < numClasses; i++) {
                    String name = dis.readUTF();
                    int sampleCount = dis.readInt();
                    double[] vec = new double[VSA_DIM];
                    for (int d = 0; d < VSA_DIM; d++) vec[d] = dis.readDouble();
                    archetypesMAP.put(name, vec);
                    sampleCounts.put(name, sampleCount);
                }
            } else {
                int numClasses = firstInt;
                int dims = dis.readInt();
                if (dims != VSA_DIM) { file.delete(); return; }
                for (int i = 0; i < numClasses; i++) {
                    String name = dis.readUTF();
                    double[] vec = new double[VSA_DIM];
                    for (int d = 0; d < VSA_DIM; d++) vec[d] = dis.readDouble();
                    archetypesMAP.put(name, vec);
                    sampleCounts.put(name, DEFAULT_SAMPLE_COUNT);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Errore caricamento MAP, cancello", e);
            file.delete();
        }
    }

    private void loadBuiltInArchetypes() {
        try (java.io.DataInputStream dis = new java.io.DataInputStream(
                getAssets().open("vsa_archetypes.bin"))) {
            int firstInt = dis.readInt();
            int numClasses;
            int dims;
            if (firstInt == FILE_MAGIC) {
                numClasses = dis.readInt();
                dims = dis.readInt();
            } else {
                numClasses = firstInt;
                dims = dis.readInt();
            }
            if (dims != VSA_DIM) {
                Log.w(TAG, "vsa_archetypes.bin ha dimensione " + dims + " invece di " + VSA_DIM);
                return;
            }
            for (int i = 0; i < numClasses; i++) {
                String name = dis.readUTF();
                int sampleCount = 0;
                try {
                    sampleCount = dis.readInt();
                } catch (Exception e) {
                    sampleCount = DEFAULT_SAMPLE_COUNT;
                }
                double[] vec = new double[VSA_DIM];
                for (int d = 0; d < VSA_DIM; d++) vec[d] = dis.readDouble();
                archetypesMAP.put(name, vec);
                sampleCounts.put(name, sampleCount);
            }
            Log.i(TAG, "Caricati " + numClasses + " archetipi predefiniti da vsa_archetypes.bin");
        } catch (Exception e) {
            Log.w(TAG, "vsa_archetypes.bin non trovato o illeggibile: " + e.getMessage());
        }
    }

    private synchronized void clearArchetypes() {
        archetypesMAP.clear();
        archetypesBSC.clear();
        sampleCounts.clear();
        java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
        if (file.exists()) file.delete();
        Log.d(TAG, "Archetipi cancellati — pronti per nuovo training");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        learningSessionManager.cancel();
        if (ssdExtractor != null) ssdExtractor.close();
        if (encoder != null) encoder.close();
        inferenceExecutor.shutdown();
    }



    private void testCloudConnection() {
        Log.d(TAG, "Avvio del test di connessione al Cloud...");

        RetrofitClient.getApi().getAllModules().enqueue(new Callback<List<ModuleBundleDTO>>() {

            @Override
            public void onResponse(Call<List<ModuleBundleDTO>> call, Response<List<ModuleBundleDTO>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<ModuleBundleDTO> listaModuli = response.body();

                    Log.d(TAG, "CONNESSO! Ricevuti " + listaModuli.size() + " moduli dal database.");

                    for (ModuleBundleDTO modulo : listaModuli) {
                        Log.d(TAG, " -> Modulo ID: " + modulo.moduleName + " | Visibilità: " + modulo.visibility);
                    }

                    Toast.makeText(MainActivity.this, "Connesso a VSHub Cloud!", Toast.LENGTH_SHORT).show();

                    aggiornaInterfacciaModuli(listaModuli);
                } else {
                    Log.e(TAG, "Il server ha risposto, ma con un errore. Codice: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<List<ModuleBundleDTO>> call, Throwable t) {
                Log.e(TAG, "ERRORE CRITICO DI CONNESSIONE: " + t.getMessage());
                Toast.makeText(MainActivity.this, "Impossibile raggiungere il server Cloud", Toast.LENGTH_LONG).show();
            }
        });
    }

    private void aggiornaInterfacciaModuli(List<ModuleBundleDTO> moduliDalServer) {
        Set<Long> moduliLocali = LocalStorageManager.getDownloadedModuleIds(MainActivity.this);

        for (ModuleBundleDTO mod : moduliDalServer) {
            if (moduliLocali.contains(mod.getModuleId())) {
                Log.d(TAG, "Modulo " + mod.getModuleName() + " è disponibile LOCALMENTE.");
            } else {
                Log.d(TAG, "Modulo " + mod.getModuleName() + " è solo sul SERVER.");
            }
        }
    }

    private void scaricaModuloDalServer(Long moduleId) {
        ProgressDialog progress = new ProgressDialog(this);
        progress.setMessage("Scaricamento modulo VSA in corso...");
        progress.show();

        RetrofitClient.getApi().getModuleBundle(moduleId).enqueue(new Callback<ModuleBundleDTO>() {
            @Override
            public void onResponse(Call<ModuleBundleDTO> call, Response<ModuleBundleDTO> response) {
                progress.dismiss();
                if (response.isSuccessful() && response.body() != null) {
                    ModuleBundleDTO bundle = response.body();

                    boolean salvato = LocalStorageManager.saveModuleBundle(MainActivity.this, bundle);

                    if (salvato) {
                        Toast.makeText(MainActivity.this, "Modulo " + bundle.getModuleName() + " scaricato con successo!", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(MainActivity.this, "Errore nel salvataggio locale", Toast.LENGTH_SHORT).show();
                    }
                }
            }

            @Override
            public void onFailure(Call<ModuleBundleDTO> call, Throwable t) {
                progress.dismiss();
                Toast.makeText(MainActivity.this, "Errore di rete: " + t.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }

    private void attivaModuloSelezionato(Long moduleId) {
        ModuleBundleDTO moduloAttivo = LocalStorageManager.loadLocalModuleBundle(this, moduleId);
        if (moduloAttivo != null) {
            Log.d(TAG, "Inizializzazione motore VSA per il modulo: " + moduloAttivo.getModuleName());

            Toast.makeText(this, "Modulo " + moduloAttivo.getModuleName() + " Attivato!", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Impossibile caricare il modulo dal disco", Toast.LENGTH_SHORT).show();
        }
    }

}
