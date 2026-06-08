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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private VsaVector reusableQueryVector;
    private static final String TAG = "VSA_Pipeline";
    private static final int CAMERA_PERMISSION_CODE = 101;
    private static final String USER_ARCHETYPES_FILE = "user_custom_archetypes.bin";

    // Motore VSA e Rete
    private AndroidTFLiteExtractor cnnExtractor;
    private RandomProjectionBridge vsaBridge;
//    private DatasetStatTracker statTracker;
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


// Caricamento asincrono iniziale (MODALITÀ LIVE CAMERA SPAZIALE)
        inferenceExecutor.execute(() -> {
            try {
                long startTime = System.currentTimeMillis();

                // 1. Inizializziamo i Motori Core
                cnnExtractor = new AndroidTFLiteExtractor(this, "mobilenet_v2_spatial.tflite");
//                cnnExtractor = new AndroidTFLiteExtractor(this, "mobilenet_vsa.tflite");

                vsaBridge = new RandomProjectionBridge(1280, 42L);

                // 2. Inizializzazione Tabula Rasa (Catalogo Base Vuoto)
                archetypes = new java.util.HashMap<>();

                // 3. Buffer Ottimizzati
                continuousBuffer = new double[10000];
                standardizedBuffer = new double[10000]; // Mantenuto per sicurezza strutturale
                bscBlocksBuffer = new long[157];
                reusableQueryVector = new VsaVector(new long[157]); // Inizializzato una sola volta!

                // 4. Caricamento archetipi appresi dall'utente (Se esistono)
                Map<String, VsaVector> userCustomClasses = loadUserArchetypesFromStorage();
                if (!userCustomClasses.isEmpty()) {
                    archetypes.putAll(userCustomClasses);
                    Log.d(TAG, "Motore avviato con " + userCustomClasses.size() + " archetipi custom.");
                } else {
                    Log.d(TAG, "Motore avviato vuoto (Tabula Rasa).");
                }

                long duration = System.currentTimeMillis() - startTime;
                Log.d(TAG, "Motore VSA Spaziale pronto in: " + duration + " ms");

                // 5. ACCENDIAMO LA FOTOCAMERA E SBLOCCHIAMO L'INTERFACCIA
                runOnUiThread(() -> {
                    fabLearn.setEnabled(true);
                    checkCameraPermissionAndStart(); // Questa è la riga magica che fa partire il video!
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore caricamento modelli core", e);
                runOnUiThread(() -> tvVerdict.setText("Errore caricamento modelli. Guarda il Logcat."));
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

    /**
     * Rilevatore agnostico di forme tramite Proiezione XY dell'Energia dei Bordi.
     * ZERO Reti Neurali. Esecuzione stimata: < 10ms.
     */
    private List<android.graphics.Rect> detectBlobs(Bitmap bitmap) {
        List<android.graphics.Rect> boxes = new ArrayList<>();
        int origW = bitmap.getWidth();
        int origH = bitmap.getHeight();

        // 1. Riduciamo l'immagine a un francobollo 128x128 per volare
        int scale = 128;
        Bitmap small = Bitmap.createScaledBitmap(bitmap, scale, scale, false);
        int[] pixels = new int[scale * scale];
        small.getPixels(pixels, 0, scale, 0, 0, scale, scale);

        // 2. Array per la proiezione dell'energia sui due assi
        int[] projX = new int[scale];
        int[] projY = new int[scale];
        long totalEnergy = 0;

        // 3. Calcolo del gradiente (Bordi)
        for (int y = 0; y < scale - 1; y++) {
            for (int x = 0; x < scale - 1; x++) {
                int i = y * scale + x;
                int p1 = pixels[i];
                int p2 = pixels[i + 1];         // Destra
                int p3 = pixels[i + scale];     // Sotto

                // Somma approssimata per convertire in scala di grigi velocemente
                int g1 = ((p1 >> 16) & 0xff) + ((p1 >> 8) & 0xff) + (p1 & 0xff);
                int g2 = ((p2 >> 16) & 0xff) + ((p2 >> 8) & 0xff) + (p2 & 0xff);
                int g3 = ((p3 >> 16) & 0xff) + ((p3 >> 8) & 0xff) + (p3 & 0xff);

                // Calcolo della "forza" del bordo in questo pixel
                int edge = Math.abs(g1 - g2) + Math.abs(g1 - g3);

                projX[x] += edge;
                projY[y] += edge;
                totalEnergy += edge;
            }
        }

        // 4. Calcolo della soglia dinamica di rumore
        int avgX = (int) (totalEnergy / scale);
        int avgY = (int) (totalEnergy / scale);
        int thresholdX = (int) (avgX * 1.5); // Accettiamo solo picchi +50% sopra la media
        int thresholdY = (int) (avgY * 1.5);

        // 5. Estrazione dei segmenti attivi
        List<int[]> segmentsX = findSegments(projX, thresholdX);
        List<int[]> segmentsY = findSegments(projY, thresholdY);

        // 6. Costruzione dei Bounding Boxes mappati sulle dimensioni originali
        float ratioW = (float) origW / scale;
        float ratioH = (float) origH / scale;

        for (int[] sx : segmentsX) {
            for (int[] sy : segmentsY) {
                // Filtriamo i falsi positivi minuscoli (polvere o artefatti)
                if (sx[1] - sx[0] > 15 && sy[1] - sy[0] > 15) {
                    int left = (int) (sx[0] * ratioW);
                    int right = (int) (sx[1] * ratioW);
                    int top = (int) (sy[0] * ratioH);
                    int bottom = (int) (sy[1] * ratioH);
                    boxes.add(new android.graphics.Rect(left, top, right, bottom));
                }
            }
        }
        return boxes;
    }

    // Metodo di supporto per trovare linee contigue di energia
    private List<int[]> findSegments(int[] projection, int threshold) {
        List<int[]> segments = new ArrayList<>();
        boolean inSegment = false;
        int start = 0;
        for (int i = 0; i < projection.length; i++) {
            if (projection[i] > threshold && !inSegment) {
                start = i;
                inSegment = true;
            } else if (projection[i] <= threshold && inSegment) {
                segments.add(new int[]{start, i});
                inSegment = false;
            }
        }
        if (inSegment) segments.add(new int[]{start, projection.length - 1});
        return segments;
    }


    /**
    public void processCameraFrame(Bitmap bitmap) {
        if (cnnExtractor == null || vsaBridge == null || archetypes == null || archetypes.isEmpty()) return;

        try {
            long startTotal = System.currentTimeMillis();

            // 1. STESSO IDENTICO RITAGLIO DEL METODO DI APPRENDIMENTO
            int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight()) / 2;
            int startX = (bitmap.getWidth() - cropSize) / 2;
            int startY = (bitmap.getHeight() - cropSize) / 2;

            Bitmap cropped = Bitmap.createBitmap(bitmap, startX, startY, cropSize, cropSize);

            // 2. L'OCCHIO
            long startCNN = System.currentTimeMillis();
            float[] features = cnnExtractor.extractGlobalFeatures(cropped);
            long timeCNN = System.currentTimeMillis() - startCNN;

            // 3. IL CERVELLO
            long startVSA = System.currentTimeMillis();
            vsaBridge.projectFeatureVector(features, continuousBuffer);

            for (int i = 0; i < 157; i++) bscBlocksBuffer[i] = 0L;
            for (int i = 0; i < 10000; i++) {
                if (continuousBuffer[i] > 0) {
                    int blockIndex = i / 64;
                    int bitIndex = i % 64;
                    bscBlocksBuffer[blockIndex] |= (1L << bitIndex);
                }
            }
            reusableQueryVector.setBlocks(bscBlocksBuffer);

            // 4. RICONOSCIMENTO
            String bestMatch = "Nessun target";
            double bestSim = 0.35; // Soglia di sicurezza

            for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
                double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
                if (sim > bestSim) {
                    bestSim = sim;
                    bestMatch = entry.getKey().replace("_", " ");
                }
            }

            long timeVSA = System.currentTimeMillis() - startVSA;
            long totalTime = System.currentTimeMillis() - startTotal;

            Log.d(TAG, "TIMING -> CNN: " + timeCNN + " ms | VSA: " + timeVSA + " ms | TOTALE: " + totalTime + " ms");

            // 5. AGGIORNAMENTO SCHERMO
            final String finalVerdict = bestMatch.equals("Nessun target") ? bestMatch : bestMatch + " (" + String.format("%.2f", bestSim) + ")";
            final String finalConfidence = "Inferenza Centrale: " + totalTime + " ms";

            runOnUiThread(() -> {
                tvVerdict.setText(finalVerdict);
                tvConfidence.setText(finalConfidence);
            });

        } catch (Exception e) {
            Log.e(TAG, "Errore nell'inferenza centrale", e);
        }
    }
    */

    public void processCameraFrame(Bitmap bitmap) {
        if (cnnExtractor == null || vsaBridge == null || archetypes == null || archetypes.isEmpty()) return;

        try {
            // -- L'OCCHIO --
            long startCNN = System.currentTimeMillis();
            float[] spatialGrid = cnnExtractor.extractSpatialFeatures(bitmap);
            long timeCNN = System.currentTimeMillis() - startCNN;

            // -- IL CERVELLO --
            long startVSA = System.currentTimeMillis();

            // Taccuini per l'NMS (Non-Maximum Suppression)
            Map<String, Double> bestSims = new HashMap<>();
            Map<String, Integer> bestX = new HashMap<>();
            Map<String, Integer> bestY = new HashMap<>();

            for (int row = 0; row < 7; row++) {
                for (int col = 0; col < 7; col++) {
                    int offset = (row * 7 + col) * 1280;

                    // Proiezione e Zero-Allocation Binarization
                    vsaBridge.projectSpatialCellToContinuous(spatialGrid, offset, continuousBuffer);

                    for (int i = 0; i < 157; i++) bscBlocksBuffer[i] = 0L;
                    for (int i = 0; i < 10000; i++) {
                        if (continuousBuffer[i] > 0) {
                            int blockIndex = i / 64;
                            int bitIndex = i % 64;
                            bscBlocksBuffer[blockIndex] |= (1L << bitIndex);
                        }
                    }
                    reusableQueryVector.setBlocks(bscBlocksBuffer);

                    // Confronto Multiplo
                    for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
                        String name = entry.getKey();
                        double similarity = reusableQueryVector.getCosineSimilarity(entry.getValue());

                        if (similarity > 0.35) { // Soglia
                            double currentBest = bestSims.containsKey(name) ? bestSims.get(name) : 0.0;
                            if (similarity > currentBest) {
                                bestSims.put(name, similarity);
                                bestX.put(name, col);
                                bestY.put(name, row);
                            }
                        }
                    }
                }
            }

            // -- RENDERING --
            StringBuilder multiVerdict = new StringBuilder();
            int oggettiRilevatiCount = 0;

            for (String name : bestSims.keySet()) {
                multiVerdict.append(name.replace("_", " "))
                        .append(" [X:").append(bestX.get(name))
                        .append(", Y:").append(bestY.get(name)).append("] ")
                        .append(String.format("(%.2f)\n", bestSims.get(name)));
                oggettiRilevatiCount++;
            }

            long timeVSA = System.currentTimeMillis() - startVSA;
            long totalTime = timeCNN + timeVSA;

            Log.d(TAG, "TIMING -> CNN: " + timeCNN + " ms | VSA: " + timeVSA + " ms | TOTALE: " + totalTime + " ms");

            final String finalVerdict = (oggettiRilevatiCount > 0) ? multiVerdict.toString() : "Scansione 7x7 attiva... Nessun target.";
            final String finalConfidence = "Tempo Frame: " + totalTime + " ms";

            runOnUiThread(() -> {
                tvVerdict.setText(finalVerdict);
                tvConfidence.setText(finalConfidence);
            });

        } catch (Exception e) {
            Log.e(TAG, "Errore nella scansione spaziale 7x7", e);
        }
    }

//    public void learnNewCustomObject(String className, List<Bitmap> sampleImages) {
//        if (sampleImages == null || sampleImages.isEmpty()) return;
//
//        inferenceExecutor.execute(() -> {
//            try {
//                long startLearning = System.currentTimeMillis();
//                Bitmap bitmap = sampleImages.get(0);
//
//                // 1. IL RITAGLIO FISSO AL CENTRO (Focus Zone)
//                // Usiamo un quadrato pari alla metà del lato più corto dello schermo
//                int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight()) / 2;
//                int startX = (bitmap.getWidth() - cropSize) / 2;
//                int startY = (bitmap.getHeight() - cropSize) / 2;
//
//                Bitmap cropped = Bitmap.createBitmap(bitmap, startX, startY, cropSize, cropSize);
//
//                // 2. Estrazione e VSA
//                float[] features = cnnExtractor.extractGlobalFeatures(cropped);
//                double[] tempContinuous = new double[10000];
//                vsaBridge.projectFeatureVector(features, tempContinuous);
//
//                long[] newBscBlocks = new long[157];
//                for (int i = 0; i < 10000; i++) {
//                    if (tempContinuous[i] > 0) {
//                        int blockIndex = i / 64;
//                        int bitIndex = i % 64;
//                        newBscBlocks[blockIndex] |= (1L << bitIndex);
//                    }
//                }
//
//                VsaVector newArchetype = new VsaVector(newBscBlocks);
//                archetypes.put(className, newArchetype);
//                saveUserArchetypeToStorage(className, newArchetype);
//
//                long duration = System.currentTimeMillis() - startLearning;
//                Log.d(TAG, "Nuova classe '" + className + "' memorizzata in " + duration + " ms.");
//
//                runOnUiThread(() -> {
//                    Toast.makeText(MainActivity.this, "Appreso: " + className, Toast.LENGTH_SHORT).show();
//                    tvVerdict.setText(className.replace("_", " "));
//                });
//
//            } catch (Exception e) {
//                Log.e(TAG, "Errore durante l'apprendimento centrale", e);
//            }
//        });
//    }


    /**
     * Metodo apprendimento tramite identificazione di
     * Blob di Pixels
     * @param className
     * @param sampleImages
     */
//    public void learnNewCustomObject(String className, List<Bitmap> sampleImages) {
//        if (sampleImages == null || sampleImages.isEmpty()) return;
//
//        inferenceExecutor.execute(() -> {
//            try {
//                long startLearning = System.currentTimeMillis();
//                Bitmap bitmap = sampleImages.get(0);
//
//                // 1. Troviamo i blob (Esattamente come fa il radar)
//                List<android.graphics.Rect> boxes = detectBlobs(bitmap);
//
//                if (boxes.isEmpty()) {
//                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Nessun oggetto rilevato. Riprova con più contrasto!", Toast.LENGTH_SHORT).show());
//                    return;
//                }
//
//                // 2. Selezioniamo il Blob più grande
//                // (In fase di apprendimento, l'utente inquadra l'oggetto da vicino)
//                android.graphics.Rect bestBox = boxes.get(0);
//                int maxArea = 0;
//                for(android.graphics.Rect b : boxes) {
//                    int area = b.width() * b.height();
//                    if (area > maxArea) {
//                        maxArea = area;
//                        bestBox = b;
//                    }
//                }
//
//                // --- 3. LA LOGICA DEL QUADRATO (IDENTICA AL RADAR) ---
//                int centerX = bestBox.centerX();
//                int centerY = bestBox.centerY();
//
//                // Troviamo il lato più lungo e aggiungiamo il 20% di margine
//                int size = Math.max(bestBox.width(), bestBox.height());
//                size = (int) (size * 1.2);
//
//                // Calcoliamo i bordi del ritaglio
//                int startX = centerX - (size / 2);
//                int startY = centerY - (size / 2);
//
//                // Sicurezza: restiamo nei bordi dell'immagine
//                startX = Math.max(0, startX);
//                startY = Math.max(0, startY);
//                if (startX + size > bitmap.getWidth()) startX = bitmap.getWidth() - size;
//                if (startY + size > bitmap.getHeight()) startY = bitmap.getHeight() - size;
//                size = Math.min(size, Math.min(bitmap.getWidth(), bitmap.getHeight()));
//
//                // Il Taglierino Quadrato
//                Bitmap cropped = Bitmap.createBitmap(bitmap, startX, startY, size, size);
//                // -----------------------------------------------------
//
//                // 4. Estrazione Globale (L'Occhio)
//                float[] features = cnnExtractor.extractGlobalFeatures(cropped);
//
//                // 5. Proiezione VSA (Il Cervello)
//                double[] tempContinuous = new double[10000];
//                vsaBridge.projectFeatureVector(features, tempContinuous);
//
//                // Binarizzazione "Flat"
//                long[] newBscBlocks = new long[157];
//                for (int i = 0; i < 10000; i++) {
//                    if (tempContinuous[i] > 0) {
//                        int blockIndex = i / 64;
//                        int bitIndex = i % 64;
//                        newBscBlocks[blockIndex] |= (1L << bitIndex);
//                    }
//                }
//
//                VsaVector newArchetype = new VsaVector(newBscBlocks);
//
//                // 6. Salvataggio definitivo
//                archetypes.put(className, newArchetype);
//                saveUserArchetypeToStorage(className, newArchetype);
//
//                long duration = System.currentTimeMillis() - startLearning;
//                Log.d(TAG, "Nuova classe '" + className + "' memorizzata in " + duration + " ms.");
//
//                // Usiamo una variabile final per passarla alla UI
//                final int finalSize = size;
//                runOnUiThread(() -> {
//                    Toast.makeText(MainActivity.this, "Appreso: " + className + " (Ritaglio: " + finalSize + "x" + finalSize + ")", Toast.LENGTH_LONG).show();
//                    tvVerdict.setText(className.replace("_", " "));
//                });
//
//            } catch (Exception e) {
//                Log.e(TAG, "Errore fatale durante l'apprendimento col quadrato", e);
//            }
//        });
//    }


    /**
     * Apprendimento Griglia 7x7
     * @param className
     * @param sampleImages
     */


    public void learnNewCustomObject(String className, List<Bitmap> sampleImages) {
        if (sampleImages == null || sampleImages.isEmpty()) return;

        inferenceExecutor.execute(() -> {
            try {
                long startLearning = System.currentTimeMillis();
                Bitmap bitmap = sampleImages.get(0);

                // 1. Estrazione Mappa Spaziale 7x7
                float[] spatialGrid = cnnExtractor.extractSpatialFeatures(bitmap);

                // 2. Isoliamo SOLO la cella centrale [X:3, Y:3] -> Indice 24
                int centerOffset = (3 * 7 + 3) * 1280;

                double[] tempContinuous = new double[10000];
                vsaBridge.projectSpatialCellToContinuous(spatialGrid, centerOffset, tempContinuous);

                // 3. Binarizzazione
                long[] newBscBlocks = new long[157];
                for (int i = 0; i < 10000; i++) {
                    if (tempContinuous[i] > 0) {
                        int blockIndex = i / 64;
                        int bitIndex = i % 64;
                        newBscBlocks[blockIndex] |= (1L << bitIndex);
                    }
                }

                // 4. Salvataggio
                VsaVector newArchetype = new VsaVector(newBscBlocks);
                archetypes.put(className, newArchetype);
                saveUserArchetypeToStorage(className, newArchetype);

                long duration = System.currentTimeMillis() - startLearning;
                Log.d(TAG, "Nuova classe '" + className + "' memorizzata in " + duration + " ms.");

                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "Appreso (Cella 3,3): " + className, Toast.LENGTH_SHORT).show();
                    tvVerdict.setText(className.replace("_", " "));
                });

            } catch (Exception e) {
                Log.e(TAG, "Errore nell'apprendimento spaziale", e);
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