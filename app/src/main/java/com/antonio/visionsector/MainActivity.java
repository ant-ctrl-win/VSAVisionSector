package com.antonio.visionsector;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "VSA_Pipeline";
    private static final int CAMERA_PERMISSION_CODE = 101;
    private static final String USER_ARCHETYPES_FILE = "user_custom_archetypes.bin";

    // Motore VSA e Rete
    private AndroidTFLiteExtractor cnnExtractor;
    private RandomProjectionBridge vsaBridge;
    private DatasetStatTracker statTracker;
    private Map<String, VsaVector> archetypes;

    // Buffer ottimizzati anti-Garbage Collector (Sub-800ms sul J3)
    private double[] continuousBuffer;
    private double[] standardizedBuffer;
    private long[] bscBlocksBuffer;

    // Threading e UI
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private PreviewView viewFinder;
    private TextView tvVerdict;
    private TextView tvConfidence;
    private FloatingActionButton fabLearn;

    // Controllo di Stato per la Fotocamera Live
    private boolean isProcessingFrame = false;
    private boolean captureNextFrameForLearning = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFinder = findViewById(R.id.viewFinder);
        tvVerdict = findViewById(R.id.tvVerdict);
        tvConfidence = findViewById(R.id.tvConfidence);
        fabLearn = findViewById(R.id.fabLearn);

        // Disabilitato di default finché il motore non è pronto
        fabLearn.setEnabled(false);

        // Click sul pulsante per attivare l'apprendimento
        fabLearn.setOnClickListener(v -> showNameInputDialog());

        // Caricamento asincrono iniziale
        inferenceExecutor.execute(() -> {
            try {
                long startTime = System.currentTimeMillis();

                cnnExtractor = new AndroidTFLiteExtractor(this);
                vsaBridge = new RandomProjectionBridge(1280, 42L);
                statTracker = DatasetStatTracker.importFromAndroidAssets(this, "vsa_stats.bin");

                continuousBuffer = new double[10000];
                standardizedBuffer = new double[10000];
                bscBlocksBuffer = new long[157];

                // Carica base da assets
                archetypes = loadArchetypesFromAndroidAssets(this, "vsa_archetypes.bin");

                // Carica ed esegue il merge degli archetipi salvati in locale dall'utente
                Map<String, VsaVector> userCustomClasses = loadUserArchetypesFromStorage();
                if (!userCustomClasses.isEmpty()) {
                    archetypes.putAll(userCustomClasses);
                    Log.d(TAG, "Fusi in RAM " + userCustomClasses.size() + " archetipi utente preesistenti.");
                }

                long duration = System.currentTimeMillis() - startTime;
                Log.d(TAG, "Motore Neuro-Simbolico pronto in: " + duration + " ms");

                runOnUiThread(() -> {
                    fabLearn.setEnabled(true);
                    checkCameraPermissionAndStart();
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore caricamento modelli core", e);
                runOnUiThread(() -> tvVerdict.setText("Errore caricamento modelli."));
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
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(inferenceExecutor, imageProxy -> {
                    if (isProcessingFrame) {
                        imageProxy.close();
                        return;
                    }

                    isProcessingFrame = true;
                    try {
                        Bitmap bitmapFrame = imageProxy.toBitmap();

                        // Intercettiamo il frame se l'utente ha chiesto di imparare un nuovo oggetto
                        if (captureNextFrameForLearning) {
                            captureNextFrameForLearning = false;
                            runOnUiThread(() -> triggerOneShotLearningFlow(bitmapFrame));
                        } else {
                            // Altrimenti eseguiamo la classificazione live continua
                            processCameraFrame(bitmapFrame);
                        }

                    } catch (Exception e) {
                        Log.e(TAG, "Errore analisi stream video", e);
                    } finally {
                        imageProxy.close();
                        isProcessingFrame = false;
                    }
                });

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

                tvVerdict.setText("Inquadra una foglia...");
                tvConfidence.setText("Scansione attiva 100% offline");

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Errore configurazione CameraX", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Mostra la finestra di dialogo per l'inserimento del nome del nuovo oggetto
     */
    private void showNameInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("One-Shot Learning");
        builder.setMessage("Inserisci il nome del nuovo oggetto da imparare:");

        final EditText input = new EditText(this);
        input.setHint("Es. Vite_Nuova_Varieta");
        builder.setView(input);

        builder.setPositiveButton("Cattura e Impara", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (!name.isEmpty()) {
                // Imposta la trappola logica: il prossimo frame video utile verrà usato per l'apprendimento
                pendingLearningName = name;
                captureNextFrameForLearning = true;
            } else {
                Toast.makeText(MainActivity.this, "Il nome non può essere vuoto", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("Annulla", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private String pendingLearningName = "";

    private void triggerOneShotLearningFlow(Bitmap sample) {
        tvVerdict.setText("Apprendimento in corso...");
        tvConfidence.setText("Elaborazione firma iperdimensionale...");
        learnNewCustomObject(pendingLearningName, Collections.singletonList(sample));
    }

    public void processCameraFrame(Bitmap bitmap) {
        if (cnnExtractor == null || vsaBridge == null || statTracker == null || archetypes == null) return;

        try {
            long startInfer = System.currentTimeMillis();

            float[] cnnFeatures = cnnExtractor.extractFeatures(bitmap);
            vsaBridge.projectToContinuous(cnnFeatures, continuousBuffer);

            double[] meanArray = statTracker.getMean();
            double[] stdDevArray = statTracker.getStdDev();

            for (int i = 0; i < 10000; i++) {
                standardizedBuffer[i] = (continuousBuffer[i] - meanArray[i]) / stdDevArray[i];
            }

            vsaBridge.binarizeTo(standardizedBuffer, bscBlocksBuffer);
            VsaVector queryVector = new VsaVector(bscBlocksBuffer);

            String bestMatchClass = "Sconosciuta";
            double maxSimilarity = -1.0;

            for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
                double similarity = queryVector.getCosineSimilarity(entry.getValue());
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestMatchClass = entry.getKey();
                }
            }

            double theoreticalSigma = 0.01;
            double zScoreVerdict = maxSimilarity / theoreticalSigma;

            final String finalVerdict;
            final String finalConfidence = String.format("Sim. Coseno: %.4f (Z-Score: %.2fσ)", maxSimilarity, zScoreVerdict);

            if (zScoreVerdict < 3.0) {
                finalVerdict = "Oggetto Sconosciuto";
            } else {
                finalVerdict = bestMatchClass.replace("___", " - ").replace("_", " ");
            }

            long inferDuration = System.currentTimeMillis() - startInfer;
            Log.d(TAG, "Inferenza live completata in: " + inferDuration + " ms.");

            runOnUiThread(() -> {
                tvVerdict.setText(finalVerdict);
                tvConfidence.setText(finalConfidence);
            });

        } catch (Exception e) {
            Log.e(TAG, "Errore nel ciclo di inferenza", e);
        }
    }

    public void learnNewCustomObject(String className, List<Bitmap> sampleImages) {
        if (sampleImages == null || sampleImages.isEmpty()) return;

        inferenceExecutor.execute(() -> {
            try {
                long startLearning = System.currentTimeMillis();
                double[] continuousBundle = new double[10000];
                double[] tempContinuous = new double[10000];

                double[] meanArray = statTracker.getMean();
                double[] stdDevArray = statTracker.getStdDev();

                for (Bitmap bitmap : sampleImages) {
                    float[] cnnFeatures = cnnExtractor.extractFeatures(bitmap);
                    vsaBridge.projectToContinuous(cnnFeatures, tempContinuous);

                    for (int i = 0; i < 10000; i++) {
                        double standardizedValue = (tempContinuous[i] - meanArray[i]) / stdDevArray[i];
                        continuousBundle[i] += standardizedValue;
                    }
                }

                long[] newBscBlocks = new long[157];
                vsaBridge.binarizeTo(continuousBundle, newBscBlocks);
                VsaVector newArchetype = new VsaVector(newBscBlocks);

                archetypes.put(className, newArchetype);
                saveUserArchetypeToStorage(className, newArchetype);

                long duration = System.currentTimeMillis() - startLearning;
                Log.d(TAG, "Nuova classe '" + className + "' memorizzata in " + duration + " ms.");

                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "Oggetto '" + className + "' appreso!", Toast.LENGTH_LONG).show();
                    tvVerdict.setText(className.replace("_", " "));
                    tvConfidence.setText("Apprendimento completato con successo");
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore fatale durante l'apprendimento", e);
            }
        });
    }

    private synchronized void saveUserArchetypeToStorage(String className, VsaVector vector) {
        Map<String, VsaVector> userClasses = loadUserArchetypesFromStorage();
        userClasses.put(className, vector);

        try (java.io.FileOutputStream fos = openFileOutput(USER_ARCHETYPES_FILE, Context.MODE_PRIVATE);
             java.io.DataOutputStream dos = new java.io.DataOutputStream(fos)) {

            dos.writeInt(userClasses.size());
            dos.writeInt(10000);

            for (Map.Entry<String, VsaVector> entry : userClasses.entrySet()) {
                dos.writeUTF(entry.getKey());
                dos.writeInt(157);
                long[] blocks = entry.getValue().getBlocks();
                for (long b : blocks) {
                    dos.writeLong(b);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Errore scrittura storage", e);
        }
    }

    private Map<String, VsaVector> loadUserArchetypesFromStorage() {
        Map<String, VsaVector> loaded = new java.util.HashMap<>();
        java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
        if (!file.exists()) return loaded;

        try (java.io.FileInputStream fis = new java.io.FileInputStream(file);
             java.io.DataInputStream dis = new java.io.DataInputStream(fis)) {

            int numClasses = dis.readInt();
            int dimensions = dis.readInt();

            for (int i = 0; i < numClasses; i++) {
                String className = dis.readUTF();
                int numBlocks = dis.readInt();
                long[] blocks = new long[numBlocks];
                for (int j = 0; j < numBlocks; j++) {
                    blocks[j] = dis.readLong();
                }
                loaded.put(className, new VsaVector(blocks));
            }
        } catch (Exception e) {
            Log.e(TAG, "Errore lettura storage", e);
        }
        return loaded;
    }

    private Map<String, VsaVector> loadArchetypesFromAndroidAssets(Context context, String fileName) throws Exception {
        Map<String, VsaVector> loaded = new java.util.HashMap<>();
        try (java.io.DataInputStream dis = new java.io.DataInputStream(context.getAssets().open(fileName))) {
            int numClasses = dis.readInt();
            int dimensions = dis.readInt();
            for (int i = 0; i < numClasses; i++) {
                String className = dis.readUTF();
                int numBlocks = dis.readInt();
                long[] blocks = new long[numBlocks];
                for (int j = 0; j < numBlocks; j++) {
                    blocks[j] = dis.readLong();
                }
                loaded.put(className, new VsaVector(blocks));
            }
        }
        return loaded;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cnnExtractor != null) cnnExtractor.close();
        inferenceExecutor.shutdown();
    }
}