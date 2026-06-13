# VsaVisionSector — Analisi Completa del Progetto

**Data:** 12/06/2026  
**Obiettivo:** Refactoring per introdurre un modulo bottleneck a 256 dimensioni e trasformare la logica VSA per gestire 49 "Archetipi Spaziali" (uno per cella della griglia 7×7) anziché un singolo archetipo globale.

---

## 1. Gestione del Modello CNN (TensorFlow Lite)

### 1.1 Caricamento del modello e inferenza

**File:** `app/src/main/java/com/antonio/visionsector/AndroidTFLiteExtractor.java`

```java
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
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,
            fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }
```

### 1.2 Estrazione del tensore spaziale 7×7×1280

Questo è il **punto critico** dove il tensore viene letto e mappato in strutture Java.

```java
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
```

**Mappatura logica dell'array piatto (62.720 float):**

```
indice = (row * 7 + col) * 1280 + featureIndex

   row=0..6, col=0..6, featureIndex=0..1279

Esempio:
  cella [0,0] → offset 0
  cella [0,1] → offset 1280
  cella [3,3] → offset (3*7+3)*1280 = 24*1280 = 30720
```

La cella centrale `[3,3]` viene estratta in `MainActivity` così:

```java
int offset = (3 * 7 + 3) * 1280;
vsaBridge.projectSpatialCellToContinuous(spatialGrid, offset, continuousBuffer);
```

### 1.3 Estrazione globale (metodo legacy)

Questo metodo esiste ancora nel codice ma è al momento inutilizzato (commentato nella pipeline principale). Estrae un vettore piatto `[1][1280]` dal modello `mobilenet_vsa.tflite` (alternativo).

```java
    public float[] extractGlobalFeatures(Bitmap croppedBitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, 224, 224, true);
        java.nio.ByteBuffer inputBuffer = java.nio.ByteBuffer.allocateDirect(224 * 224 * 3 * 4);
        inputBuffer.order(java.nio.ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224);

        // Normalizzazione nell'intervallo [-1.0, 1.0] (Standard MobileNet)
        int pixel = 0;
        for (int i = 0; i < 224; i++) {
            for (int j = 0; j < 224; j++) {
                int val = intValues[pixel++];
                inputBuffer.putFloat((((val >> 16) & 0xFF) - 127.5f) / 127.5f);
                inputBuffer.putFloat((((val >> 8) & 0xFF) - 127.5f) / 127.5f);
                inputBuffer.putFloat(((val & 0xFF) - 127.5f) / 127.5f);
            }
        }

        float[][] outputArray = new float[1][1280];
        tflite.run(inputBuffer, outputArray);
        return outputArray[0];
    }
```

### 1.4 Preprocessing dell'immagine (normalizzazione MobileNetV2)

```java
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
```

### 1.5 Modelli TFLite in uso

| File | Descrizione |
|------|-------------|
| `app/src/main/assets/mobilenet_v2_spatial.tflite` | Modello principale attivo (output 7×7×1280) |
| `app/src/main/assets/mobilenet_vsa.tflite` | Modello alternativo (output [1][1280], attualmente inutilizzato) |

---

## 2. Pipeline della Fotocamera (CameraX)

**File:** `app/src/main/java/com/antonio/visionsector/MainActivity.java`

### 2.1 Configurazione CameraX con ImageAnalysis

```java
private void startCamera() {
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
        ProcessCameraProvider.getInstance(this);
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

                    if (captureNextFrameForLearning) {
                        captureNextFrameForLearning = false;
                        runOnUiThread(() -> triggerOneShotLearningFlow(bitmapFrame));
                    } else {
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
```

### 2.2 Conversione frame → square bitmap per la CNN

Il metodo `processCameraFrame()` ritaglia un quadrato centrale dall'immagine della fotocamera prima di passarlo alla CNN:

```java
public void processCameraFrame(Bitmap bitmap) {
    if (cnnExtractor == null || vsaBridge == null || archetypes == null || archetypes.isEmpty()) return;

    try {
        // 1. IL QUADRATO PERFETTO DAL FOTOGRAMMA
        int minDim = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int startX = (bitmap.getWidth() - minDim) / 2;
        int startY = (bitmap.getHeight() - minDim) / 2;
        Bitmap squareBitmap = Bitmap.createBitmap(bitmap, startX, startY, minDim, minDim);

        // 2. L'OCCHIO (CNN)
        long startCNN = System.currentTimeMillis();
        float[] spatialGrid = cnnExtractor.extractSpatialFeatures(squareBitmap);
        long timeCNN = System.currentTimeMillis() - startCNN;

        // 3. IL CERVELLO (VSA - solo cella centrale [3,3])
        // ... (vedi sezione VSA)
    }
}
```

**Flusso completo fotocamera → CNN:**

```
CameraX ImageProxy → .toBitmap() → Bitmap
    → crop quadrato centrale (min(w,h) × min(w,h))
    → Bitmap.createScaledBitmap(224, 224)
    → convertBitmapToByteBuffer() [normalizzazione [-1,1] per canale RGB]
    → tflite.run(imgData, outputBuffer) → ByteBuffer
    → outputBuffer.asFloatBuffer().get(spatialFeaturesArray) → float[62720]
```

---

## 3. Logica VSA Attuale su Android

**Linguaggio:** Java (non Kotlin)

### 3.1 Random Projection Bridge

**File:** `app/src/main/java/com/antonio/visionsector/RandomProjectionBridge.java`

Classe che proietta i vettori CNN (1280-dim) nello spazio VSA (10000-dim) usando una matrice di proiezione casuale booleana.

```java
package com.antonio.visionsector;

import java.util.Random;

public class RandomProjectionBridge {
    private final int cnnFeatureSize;   // 1280
    private final int vsaDimensions;    // 10000

    // Layout ottimizzato [10000][1280] per cache locality
    private final boolean[][] projectionMatrix;

    public RandomProjectionBridge(int cnnFeatureSize, long seed) {
        this.cnnFeatureSize = cnnFeatureSize;
        this.vsaDimensions = 10000;
        this.projectionMatrix = new boolean[vsaDimensions][cnnFeatureSize];

        Random random = new Random(seed);

        // Generazione matrice trasposta per accesso cache-friendly
        for (int i = 0; i < cnnFeatureSize; i++) {
            for (int d = 0; d < vsaDimensions; d++) {
                boolean casuale = random.nextBoolean();
                projectionMatrix[d][i] = casuale;
            }
        }
    }

    /**
     * Proiezione di una singola cella spaziale (offset nell'array piatto della CNN).
     * Ottimizzazione: filtraggio sparso (ReLU) — solo feature != 0 vengono proiettate.
     */
    public void projectSpatialCellToContinuous(float[] spatialGrid, int startOffset,
                                               double[] outContinuous) {
        // FASE 1: Filtraggio sparso (salta i valori zero della ReLU6)
        int[] activeIndices = new int[1280];
        float[] activeValues = new float[1280];
        int activeCount = 0;

        for (int i = 0; i < 1280; i++) {
            float val = spatialGrid[startOffset + i];
            if (val != 0.0f) {
                activeIndices[activeCount] = i;
                activeValues[activeCount] = val;
                activeCount++;
            }
        }

        // FASE 2: Proiezione iperdimensionale (solo sulle feature attive)
        for (int j = 0; j < 10000; j++) {
            boolean[] basis = projectionMatrix[j];
            double sum = 0.0;

            for (int k = 0; k < activeCount; k++) {
                int idx = activeIndices[k];
                if (basis[idx]) {
                    sum += activeValues[k];
                } else {
                    sum -= activeValues[k];
                }
            }
            outContinuous[j] = sum;
        }
    }

    /**
     * Proiezione di un vettore globale 1280-dim (alternativa legacy).
     */
    public void projectFeatureVector(float[] features, double[] outContinuous) {
        int[] activeIndices = new int[1280];
        float[] activeValues = new float[1280];
        int activeCount = 0;

        for (int i = 0; i < 1280; i++) {
            float val = features[i];
            if (val != 0.0f) {
                activeIndices[activeCount] = i;
                activeValues[activeCount] = val;
                activeCount++;
            }
        }

        for (int j = 0; j < 10000; j++) {
            boolean[] basis = projectionMatrix[j];
            double sum = 0.0;
            for (int k = 0; k < activeCount; k++) {
                int idx = activeIndices[k];
                if (basis[idx]) sum += activeValues[k];
                else sum -= activeValues[k];
            }
            outContinuous[j] = sum;
        }
    }

    /**
     * Binarizzazione: continuous → BSC blocks (long[])
     * Soglia zero: > 0 → bit=1, ≤ 0 → bit=0
     */
    public void binarizeTo(double[] continuousVector, long[] reuseBlocks) {
        java.util.Arrays.fill(reuseBlocks, 0L);

        for (int d = 0; d < vsaDimensions; d++) {
            if (continuousVector[d] > 0) {
                int blockIndex = d / 64;
                int bitPosition = d % 64;
                reuseBlocks[blockIndex] |= (1L << bitPosition);
            }
        }
    }
}
```

### 3.2 VSA Vector (BSC — Binary Spatter Code)

**File:** `app/src/main/java/com/antonio/visionsector/VsaVector.java`

Rappresentazione iperdimensionale binaria: 10000 bit compressi in 157 `long`.

```java
package com.antonio.visionsector;

public class VsaVector {
    public static final int DIMENSIONS = 10000;
    private static final int ARRAY_SIZE = (DIMENSIONS + 63) / 64;  // 157

    private long[] blocks;

    public VsaVector(long[] blocks) {
        if (blocks.length != ARRAY_SIZE) {
            throw new IllegalArgumentException("Dimensione array errata");
        }
        this.blocks = blocks;
    }

    /**
     * BINDING (Associazione) = XOR su ogni blocco a 64 bit.
     * Un singolo ciclo CPU elabora 64 dimensioni.
     */
    public VsaVector bind(VsaVector other) {
        long[] result = new long[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            result[i] = this.blocks[i] ^ other.blocks[i];
        }
        return new VsaVector(result);
    }

    /**
     * Distanza di Hamming: conteggio bit discordanti (POPCNT hardware).
     */
    public int calculateHammingDistance(VsaVector other) {
        int distance = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            long differences = this.blocks[i] ^ other.blocks[i];
            distance += Long.bitCount(differences);
        }
        return distance;
    }

    /**
     * Similarità del Coseno equivalente.
     * Formula: 1 - 2 * (Hamming / Dimensions)
     */
    public double getCosineSimilarity(VsaVector other) {
        int h = calculateHammingDistance(other);
        return 1.0 - 2.0 * ((double) h / DIMENSIONS);
    }

    public long[] getBlocks() {
        return blocks;
    }

    /**
     * Setter "Zero-Allocation": sovrascrive i 157 long in-place senza GC.
     */
    public void setBlocks(long[] newBlocks) {
        if (newBlocks != null && newBlocks.length == 157) {
            System.arraycopy(newBlocks, 0, this.blocks, 0, 157);
        }
    }
}
```

### 3.3 DatasetStatTracker (Welford / Z-Score)

**File:** `app/src/main/java/com/antonio/visionsector/DatasetStatTracker.java`

Classe per la standardizzazione Z-Score. Attualmente **non utilizzata nella pipeline live** (il campo `statTracker` è commentato in `MainActivity`). Carica media e deviazione standard da un file binario pre-calcolato.

```java
package com.antonio.visionsector;

import android.content.Context;
import java.io.DataInputStream;
import java.io.IOException;

public class DatasetStatTracker {
    private final int dimensions;
    private final double[] mean;
    private final double[] stdDev;

    public DatasetStatTracker(int dimensions, double[] mean, double[] stdDev) {
        this.dimensions = dimensions;
        this.mean = mean;
        this.stdDev = stdDev;
    }

    /**
     * Standardizzazione Z-Score: (x - μ) / σ
     */
    public double[] standardize(double[] continuousVector) {
        double[] standardized = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            standardized[i] = (continuousVector[i] - mean[i]) / stdDev[i];
        }
        return standardized;
    }

    /**
     * Caricamento statistiche da assets (vsa_stats.bin).
     */
    public static DatasetStatTracker importFromAndroidAssets(Context context, String fileName)
            throws IOException {
        try (DataInputStream dis = new DataInputStream(context.getAssets().open(fileName))) {
            int dims = dis.readInt();
            double[] loadedMean = new double[dims];
            double[] loadedStd = new double[dims];

            for (int i = 0; i < dims; i++) {
                loadedMean[i] = dis.readDouble();
            }
            for (int i = 0; i < dims; i++) {
                loadedStd[i] = dis.readDouble();
            }
            return new DatasetStatTracker(dims, loadedMean, loadedStd);
        }
    }
}
```

**Nota:** Non c'è implementazione dell'algoritmo di Welford online in Java in questo progetto. Il file `vsa_stats.bin` negli assets contiene statistiche pre-calcolate (presumibilmente da un training Python). L'algoritmo di Welford era stato probabilmente pianificato ma non implementato lato Android.

### 3.4 Flusso VSA completo (classificazione live)

Estratto da `MainActivity.processCameraFrame()` — il flusso attuale usa solo la **cella centrale [3,3]**:

```java
// Punto critico: estrazione cella centrale
int offset = (3 * 7 + 3) * 1280;

// Passo 1: Proiezione random da 1280 → 10000 (continuo)
vsaBridge.projectSpatialCellToContinuous(spatialGrid, offset, continuousBuffer);

// Passo 2: Binarizzazione manuale inline (senza chiamare binarizeTo)
for (int i = 0; i < 157; i++) bscBlocksBuffer[i] = 0L;
for (int i = 0; i < 10000; i++) {
    if (continuousBuffer[i] > 0) {
        int blockIndex = i / 64;
        int bitIndex = i % 64;
        bscBlocksBuffer[blockIndex] |= (1L << bitIndex);
    }
}

// Passo 3: Set in-place nel VsaVector riutilizzabile
reusableQueryVector.setBlocks(bscBlocksBuffer);

// Passo 4: Ricerca nearest-neighbor tra gli archetipi
String bestMatch = null;
double bestSim = 0.25; // Soglia di sicurezza
for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
    double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
    if (sim > bestSim) {
        bestSim = sim;
        bestMatch = entry.getKey();
    }
}
```

### 3.5 One-Shot Learning (apprendimento archetipo)

```java
public void learnNewCustomObject(String className, List<Bitmap> sampleImages) {
    inferenceExecutor.execute(() -> {
        Bitmap bitmap = sampleImages.get(0);

        // Stesso preprocessing della classificazione
        int minDim = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int startX = (bitmap.getWidth() - minDim) / 2;
        int startY = (bitmap.getHeight() - minDim) / 2;
        Bitmap squareBitmap = Bitmap.createBitmap(bitmap, startX, startY, minDim, minDim);

        float[] spatialGrid = cnnExtractor.extractSpatialFeatures(squareBitmap);

        // SOLO CELLA CENTRALE [3,3] per l'archetipo
        int centerOffset = (3 * 7 + 3) * 1280;
        double[] tempContinuous = new double[10000];
        vsaBridge.projectSpatialCellToContinuous(spatialGrid, centerOffset, tempContinuous);

        // Binarizzazione
        long[] newBscBlocks = new long[157];
        for (int i = 0; i < 10000; i++) {
            if (tempContinuous[i] > 0) {
                int blockIndex = i / 64;
                int bitIndex = i % 64;
                newBscBlocks[blockIndex] |= (1L << bitIndex);
            }
        }
        VsaVector newArchetype = new VsaVector(newBscBlocks);
        archetypes.put(className, newArchetype);
        saveUserArchetypeToStorage(className, newArchetype);
    });
}
```

### 3.6 Struttura degli archetipi

```java
// Map<String, VsaVector> — nome classe → vettore BSC (10000 bit)
private Map<String, VsaVector> archetypes;

// Serializzazione su disco (internal storage)
private static final String USER_ARCHETYPES_FILE = "user_custom_archetypes.bin";
```

---

## 4. Dipendenze e Configurazione

### 4.1 File `app/build.gradle.kts`

**File:** `app/build.gradle.kts`

```kotlin
plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.antonio.visionsector"
    compileSdk = 34

    androidResources {
        noCompress.add("tflite")     // <-- Critico: evita compressione APK dei modelli TFLite
    }

    defaultConfig {
        applicationId = "com.antonio.visionsector.mono"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    // AndroidX
    implementation("androidx.activity:activity-ktx:1.8.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Material Design
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.cardview:cardview:1.0.0")

    // CameraX
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")

    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    // Test
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}
```

### 4.2 Version Catalog

**File:** `gradle/libs.versions.toml`

```toml
[versions]
agp = "9.2.1"
junit = "4.13.2"
junitVersion = "1.1.5"
espressoCore = "3.5.1"
appcompat = "1.6.1"
material = "1.10.0"
activityKtx = "1.8.0"
constraintlayout = "2.1.4"
```

### 4.3 Root `build.gradle.kts`

**File:** `build.gradle.kts` (root)

```kotlin
plugins {
    alias(libs.plugins.android.application) apply false
}
```

---

## Riepilogo Tecnico per il Refactoring

### Stato attuale
- **CNN:** MobileNetV2 (`.tflite`) → output `float[62720]` = griglia 7×7×1280
- **Proiezione:** Random projection booleana 1280 → 10000 (matrice `boolean[10000][1280]`, seed=42)
- **Binarizzazione:** Soglia zero (valore > 0 → bit=1)
- **VSA:** BSC a 10000 bit, binding=XOR, similarità=coseno da Hamming
- **Classificazione:** Solo cella centrale [3,3] proiettata e confrontata con archetipi
- **Archetipi:** Un unico `VsaVector` per classe, memorizzato in `Map<String, VsaVector>`
- **Normalizzazione:** Z-Score pianificato (`DatasetStatTracker` / `vsa_stats.bin`) ma non attivo
- **Linguaggio:** 100% Java, nessun Kotlin

### Punti di intervento per il refactoring

| Componente | File | Modifica richiesta |
|-----------|------|-------------------|
| Bottleneck 256-dim | `AndroidTFLiteExtractor.java` / nuovo modello `.tflite` | Nuovo tensore output: da `7×7×1280` a `7×7×256` |
| Proiezione VSA | `RandomProjectionBridge.java` | Adattare `cnnFeatureSize` da 1280 a 256; o generare nuova matrice `boolean[10000][256]` |
| 49 Archetipi spaziali | `MainActivity.java` | `Map<String, VsaVector>` → `Map<String, VsaVector[]>` (49 vettori per classe); bundling/binding sulle 49 celle |
| Classificazione | `MainActivity.processCameraFrame()` | Iterare su tutte le 49 celle, proiettare ciascuna, combinare via bundling/XOR |
| One-shot learning | `MainActivity.learnNewCustomObject()` | Salvare 49 vettori invece di 1 |
| Statistiche Z-Score | `MainActivity` | Riattivare `DatasetStatTracker` con `vsa_stats.bin` aggiornato per 256-dim |
