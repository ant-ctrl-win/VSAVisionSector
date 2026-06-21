# Extract_Android.md — Pipeline VSA su Android (Two‑Stage Cascade + Multi‑Shot)

**App**: VsaVisionSector — Object Detection + Multi‑Shot Learning basato su VSA/BSC  
**Piattaforma**: Android (TFLite, GPU Delegate, CameraX, MediaPipe)  
**Linguaggio**: Java 11  
**Data analisi**: 2026‑06‑21

📂 **Struttura dei package attuale**:
```
com.antonio.visionsector
├── ui/
│   ├── MainActivity.java           — Orchestratore app
│   ├── DetectionStabilizer.java    — Stabilizzazione temporale
│   └── TrackingOverlayView.java    — Overlay bounding box
├── detector/
│   └── SsdExtractor.java           — Stadio 1: SSD Object Detection
├── encoder/
│   └── FeatureEncoder.java         — Stadio 2: CNN MobileNetV2
├── engine/
│   ├── vsa/
│   │   ├── VsaVector.java          — Ipervettore BSC
│   │   ├── RandomProjectionBridge.java — Proiezione 1280→10000
│   │   └── DatasetStatTracker.java — Z‑Score (non cablato)
│   └── learning/
│       └── LearningSessionManager.java — Multi‑shot bundling
└── utils/
    └── AntiGlareFilter.java        — Filtro LUT gamma (non cablato)
```

---

## 1. Caricamento e pre‑processamento dell'immagine

### 1.1 Origine dell'immagine

L'immagine proviene **esclusivamente dalla fotocamera posteriore**, tramite CameraX. Non è implementata la selezione da galleria. **Non è più usato `ImageCapture`** — l'apprendimento avviene in streaming dai frame dell'`ImageAnalysis`, senza scattare foto.

```java
// MainActivity.java:219-221
ImageAnalysis analysis = new ImageAnalysis.Builder()
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build();
```

L'analyzer viene eseguito su un `ExecutorService` single‑thread con flag `isProcessingFrame` per saltare frame quando la pipeline è occupata:

```java
// MainActivity.java:68,76
private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
private boolean isProcessingFrame = false;

// MainActivity.java:223-240
analysis.setAnalyzer(inferenceExecutor, imageProxy -> {
    if (isProcessingFrame) { imageProxy.close(); return; }
    isProcessingFrame = true;
    try {
        Bitmap frame = imageProxy.toBitmap();
        if (frame != null) {
            processCameraFrame(frame, imageProxy.getImageInfo().getRotationDegrees());
        }
    } catch (Exception e) { /* ... */ }
    finally { imageProxy.close(); isProcessingFrame = false; }
});
```

CameraX ora lega **solo** `Preview` e `ImageAnalysis` (nessun `ImageCapture`):

```java
// MainActivity.java:241-244
CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
provider.unbindAll();
provider.bindToLifecycle(this, selector, preview, analysis);
```

### 1.2 Object Detection (Primo Stadio — con GPU fallback)

Lo stadio 1 è un **SSD MobileNetV2 class‑agnostic** (addestrato con MediaPipe Model Maker) che produce bounding box normalizzati [0.0, 1.0]. Modello: `mobilenet_ssd.tflite`.

```
Input:  float32[1, 256, 256, 3]
Output: boxes  float32[1, 12276, 4]   (raw anchor offsets)
        scores float32[1, 12276, 2]   (index 1 = object confidence)
```

Anchor decoding, NMS e ridimensionamento sono gestiti internamente da **MediaPipe Tasks Vision** (`ObjectDetector`).

**Costruttore con GPU/CPU fallback**:

```java
// SsdExtractor.java:47-91
public SsdExtractor(Context context, String modelPath) throws Exception {
    boolean gpuOk = false;
    ObjectDetector det;

    // ── Tentativo GPU ─────────────────────────────────────────────
    try {
        BaseOptions gpuOpts = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.GPU)
                .build();
        ObjectDetector.ObjectDetectorOptions opts =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(gpuOpts)
                        .setMaxResults(1)
                        .setScoreThreshold(0.3f)
                        .setRunningMode(RunningMode.IMAGE)
                        .build();
        det = ObjectDetector.createFromOptions(context, opts);
        gpuOk = true;
        Log.i(TAG, "SSD avviato con GPU — model: " + modelPath);
    } catch (Exception gpuError) {
        Log.w(TAG, "GPU SSD non disponibile — ricado su CPU");
        // ── Fallback CPU ──────────────────────────────────────────
        BaseOptions cpuOpts = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                .build();
        // ... stesse opzioni ...
        det = ObjectDetector.createFromOptions(context, opts);
    }
    this.detector = det;
    this.isUsingGpu = gpuOk;
}
```

**Nota sul name‑clash**: `com.google.mediapipe.tasks.core.Delegate` (enum: CPU/GPU) e `org.tensorflow.lite.Delegate` (interfaccia) coesistono come tipi distinti. In questo file si usa il fully‑qualified name per evitare ambiguità.

Il metodo `detectObject()` converte il bitmap in `MPImage` e restituisce il box normalizzato:

```java
// SsdExtractor.java:100-120
public RectF detectObject(Bitmap bitmap) {
    int imgW = bitmap.getWidth();
    int imgH = bitmap.getHeight();
    MPImage mpImage = new BitmapImageBuilder(bitmap).build();
    ObjectDetectionResult result = detector.detect(mpImage);
    List<Detection> detections = result.detections();
    if (detections == null || detections.isEmpty()) return null;
    RectF bbox = detections.get(0).boundingBox();
    return new RectF(
        bbox.left   / imgW,
        bbox.top    / imgH,
        bbox.right  / imgW,
        bbox.bottom / imgH
    );
}
```

### 1.3 Espansione del box (padding 20%)

```java
// MainActivity.java:495-501
private RectF expandBox(RectF box, float ratio) {
    float w = box.width() * ratio;
    float h = box.height() * ratio;
    return new RectF(
        Math.max(0f, box.left - w), Math.max(0f, box.top - h),
        Math.min(1f, box.right + w), Math.min(1f, box.bottom + h)
    );
}
```

### 1.4 Sanity check del box

```java
// MainActivity.java:504-508
private boolean isBoxSane(RectF box) {
    if (box == null) return false;
    float area = box.width() * box.height();
    if (area < 0) area = -area;
    return area >= 0.01f && area <= 0.80f;
}
```

### 1.5 Ritaglio e Letterbox

Il box in pixel viene ritagliato dal frame originale e centrato su un **quadrato nero** il cui lato è `max(crop.width, crop.height)`:

```java
// MainActivity.java:511-534
private Bitmap letterboxCrop(Bitmap source, RectF box) {
    // ... ritaglia RectF → Bitmap crop ...
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
```

### 1.6 Ridimensionamento a 224×224

Il `FeatureEncoder` ridimensiona il quadrato letterbox a 224×224 con filtro bilineare:

```java
// FeatureEncoder.java:80
Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);
```

### 1.7 Normalizzazione per TFLite

I pixel ARGB_8888 sono normalizzati nell'intervallo **[-1.0, 1.0]**:

$$
\text{value} = \frac{\text{channel} - 127.5}{127.5}
$$

```java
// FeatureEncoder.java:91-95
for (int pixel : pixels) {
    inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f); // R
    inputBuffer.putFloat((((pixel >> 8)  & 0xFF) - 127.5f) / 127.5f); // G
    inputBuffer.putFloat((( pixel        & 0xFF) - 127.5f) / 127.5f); // B
}
```

### 1.8 Conversione in input TFLite

`ByteBuffer.allocateDirect()` in ordine nativo, layout **NHWC** (224×224×3 canali interleaved):

```java
// FeatureEncoder.java:83-84
ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
inputBuffer.order(ByteOrder.nativeOrder());
```

Dimensione: 224 × 224 × 3 × 4 = **602 112 byte** (~588 KB).

---

## 2. Estrazione delle feature CNN

### 2.1 Modello utilizzato

**MobileNetV2** esportato in `.tflite`. File: `mobilenet_vsa.tflite` (~2.5 MB).

| Proprietà | Valore |
|-----------|--------|
| File | `app/src/main/assets/mobilenet_vsa.tflite` |
| Formato | TensorFlow Lite FlatBuffer |
| Input | `float32[1, 224, 224, 3]` (NHWC) |
| Output | `float32[1, 1280]` |
| Normalizzazione | [-1.0, 1.0] |

### 2.2 Caricamento dell'interprete TFLite con GPU/CPU fallback

Il modello è caricato via `MappedByteBuffer` da asset. Il costruttore usa `CompatibilityList` per verificare il supporto GPU, ottiene le migliori opzioni per il dispositivo con `getBestOptionsForThisDevice()` e imposta `INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER`. Se la GPU non è compatibile, ricade su CPU 4‑thread. Il flag `isUsingGpu` **non è `final`** per consentire l'assegnazione condizionale:

```java
// FeatureEncoder.java:30-64
private boolean isUsingGpu = false;                      // non‑final, assegnato dinamicamente

public FeatureEncoder(Context context, String modelPath) throws Exception {
    Interpreter.Options options = new Interpreter.Options();

    try {
        CompatibilityList compatList = new CompatibilityList();     // org.tensorflow.lite.gpu.CompatibilityList
        if (compatList.isDelegateSupportedOnThisDevice()) {
            GpuDelegate.Options gpuOptions = compatList.getBestOptionsForThisDevice();
            gpuOptions.setInferencePreference(
                GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
            gpuDelegate = new GpuDelegate(gpuOptions);
            options.addDelegate(gpuDelegate);
            isUsingGpu = true;
            Log.i(TAG, "TFLite FeatureEncoder: GPU Delegate inizializzato con successo.");
        } else {
            throw new UnsupportedOperationException("GPU non supportata dalla CompatibilityList.");
        }
    } catch (Exception e) {
        Log.w(TAG, "Fallback su CPU (4 thread). Motivo: " + e.getMessage());
        options.setNumThreads(4);
        isUsingGpu = false;
    }

    MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
    tflite = new Interpreter(modelBuffer, options);
}
```

Il metodo `loadModelFile()` usa `AssetFileDescriptor` + `FileChannel.map()` per zero‑copy:

```java
// FeatureEncoder.java:66-71
private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
    AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
    FileInputStream is = new FileInputStream(fd.getFileDescriptor());
    FileChannel ch = is.getChannel();
    return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
}
```

### 2.3 Output tensor e inferenza

```java
// FeatureEncoder.java:76-102
public float[] extract(Bitmap crop) {
    if (crop == null) return new float[FEATURE_DIM];

    Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);
    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
    inputBuffer.order(ByteOrder.nativeOrder());

    int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
    resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
    resized.recycle();

    for (int pixel : pixels) {
        inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f);
        inputBuffer.putFloat((((pixel >> 8)  & 0xFF) - 127.5f) / 127.5f);
        inputBuffer.putFloat((( pixel        & 0xFF) - 127.5f) / 127.5f);
    }

    float[][] output = new float[1][FEATURE_DIM];
    tflite.run(inputBuffer, output);
    return output[0];  // float[1280]
}
```

**Chiusura risorse GPU**:

```java
// FeatureEncoder.java:107-119
public void close() {
    try {
        if (tflite != null) {
            tflite.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close(); // rilascia contesto OpenCL/GLES
        }
        Log.d(TAG, "Risorse dell'interprete TFLite rilasciate correttamente.");
    } catch (Exception e) { /* ... */ }
}
```

### 2.4 GPU config label

All'avvio, viene determinato lo stato GPU di entrambi i modelli e mostrato nell'UI:

```java
// MainActivity.java:138-139
gpuConfigLabel = "SSD:" + (ssdExtractor.isUsingGpu() ? "GPU" : "CPU")
        + " Enc:" + (encoder.isUsingGpu() ? "GPU" : "CPU");
```

---

## 3. Proiezione casuale 1280 → 10 000

### 3.1 Generazione della matrice di proiezione

Matrice `boolean[][]` con layout **trasposto**: `[10000][1280]` per cache locality. Seed deterministico `42L`.

```java
// RandomProjectionBridge.java:10-15
private final boolean[][] projectionMatrix;  // [10000][1280]

public RandomProjectionBridge(int cnnFeatureSize, long seed) {
    this.cnnFeatureSize = cnnFeatureSize;          // 1280
    this.vsaDimensions = 10000;
    this.projectionMatrix = new boolean[vsaDimensions][cnnFeatureSize];

    Random random = new Random(seed);
    for (int i = 0; i < cnnFeatureSize; i++) {
        for (int d = 0; d < vsaDimensions; d++) {
            boolean casuale = random.nextBoolean();
            projectionMatrix[d][i] = casuale;      // memorizzato trasposto
        }
    }
}
```

L'ordine di estrazione dal PRNG è **identico** alla versione Java desktop.

**Dimensione memoria**: 10000 × 1280 boolean ≈ **12.8 MB** in heap.

### 3.2 Algoritmo del prodotto scalare ±1 (con zero‑skipping)

$$
v_d = \sum_{i=0}^{1279} s_i \cdot f_i \quad \text{con } s_i = \begin{cases} +1 & \text{se } \text{projectionMatrix}[d][i] = \texttt{true} \\ -1 & \text{se } \text{projectionMatrix}[d][i] = \texttt{false} \end{cases}
$$

```java
// RandomProjectionBridge.java:103-128
public void projectFeatureVector(float[] features, double[] outContinuous) {
    int[] activeIndices = new int[cnnFeatureSize];
    float[] activeValues = new float[cnnFeatureSize];
    int activeCount = 0;

    // Zero‑skipping: raccoglie solo feature non‑nulle
    for (int i = 0; i < cnnFeatureSize; i++) {
        float val = features[i];
        if (val != 0.0f) {
            activeIndices[activeCount] = i;
            activeValues[activeCount] = val;
            activeCount++;
        }
    }

    // Proiezione: itera solo sui filtri attivi
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
```

### 3.3 Variante completa senza zero‑skipping (`projectToContinuous`)

Metodo alternativo che itera su **tutte** le 1280 feature (nessuno zero‑skipping) e include la normalizzazione L2 **inline** nel risultato. Accetta un buffer `reuseVector` per azzerare la garbage collection:

```java
// RandomProjectionBridge.java:33-61
public double[] projectToContinuous(float[] cnnFeatures, double[] reuseVector) {
    double sumOfSquares = 0.0;

    for (int d = 0; d < vsaDimensions; d++) {
        double sum = 0.0;
        boolean[] row = projectionMatrix[d];      // riga caricata in cache L1
        for (int i = 0; i < cnnFeatureSize; i++) {
            if (row[i]) sum += cnnFeatures[i];
            else sum -= cnnFeatures[i];
        }
        reuseVector[d] = sum;
        sumOfSquares += sum * sum;
    }

    // Normalizzazione L2 integrata
    double magnitude = Math.sqrt(sumOfSquares);
    if (magnitude > 1e-9) {
        for (int d = 0; d < vsaDimensions; d++) reuseVector[d] /= magnitude;
    }
    return reuseVector;
}
```

**Nota**: questo metodo non è attualmente chiamato dalla pipeline attiva (che usa `projectFeatureVector` + `normalizeL2` separati), ma è disponibile come alternativa con L2 embedded.

### 3.4 Normalizzazione L2 (`normalizeL2`)

```java
// RandomProjectionBridge.java:131-138
public void normalizeL2(double[] vector) {
    double sumSq = 0;
    for (int i = 0; i < vsaDimensions; i++) sumSq += vector[i] * vector[i];
    double norm = Math.sqrt(sumSq);
    if (norm > 1e-9) {
        for (int i = 0; i < vsaDimensions; i++) vector[i] /= norm;
    }
}
```

**Quando si applica**:
- **Inferenza**: MAI (la binarizzazione è diretta, senza L2)
- **Apprendimento**: SÌ — ogni frame viene normalizzato L2 prima di essere aggiunto all'accumulatore; il bundle finale viene normalizzato L2 prima del salvataggio

### 3.5 Binarizzazione diretta sul bridge (`binarizeTo`)

Il `RandomProjectionBridge` espone anche un metodo di binarizzazione diretta che scrive in un buffer `long[]` pre‑allocato, alternativa al `binarizeToBlocks` di `MainActivity`:

```java
// RandomProjectionBridge.java:144-155
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
```

### 3.6 Buffer pre‑allocati (anti‑GC)

```java
// MainActivity.java:64-66
private double[] continuousBuffer;       // double[10000] — riusato ogni frame
private long[] bscBlocksBuffer;          // long[157]    — riusato ogni frame
private VsaVector reusableQueryVector;   // VsaVector    — setBlocks() zero‑allocation
```

---

## 4. Standardizzazione Z‑Score (DatasetStatTracker)

La classe `DatasetStatTracker` esiste nel package `engine.vsa` ma **non è attualmente integrata nella pipeline attiva**.

### 4.1 Formato del file `vsa_stats.bin`

| Offset | Tipo | Contenuto |
|--------|------|-----------|
| 0 | `int` (4 byte) | Numero dimensioni `dims` |
| 4 | `double[dims]` | Vettore `mean` |
| 4+8×dims | `double[dims]` | Vettore `stdDev` |

Il file viene caricato tramite il factory method statico `DatasetStatTracker.importFromAndroidAssets(Context, String)`:

```java
// DatasetStatTracker.java:39-56
public static DatasetStatTracker importFromAndroidAssets(Context context, String fileName) throws IOException {
    try (DataInputStream dis = new DataInputStream(context.getAssets().open(fileName))) {
        int dims = dis.readInt();
        double[] loadedMean = new double[dims];
        double[] loadedStd = new double[dims];
        for (int i = 0; i < dims; i++) loadedMean[i] = dis.readDouble();
        for (int i = 0; i < dims; i++) loadedStd[i] = dis.readDouble();
        return new DatasetStatTracker(dims, loadedMean, loadedStd);
    }
}
```

### 4.2 Applicazione (non in uso)

```java
// DatasetStatTracker.java:18-24
public double[] standardize(double[] continuousVector) {
    double[] standardized = new double[dimensions];
    for (int i = 0; i < dimensions; i++) {
        standardized[i] = (continuousVector[i] - mean[i]) / stdDev[i];
    }
    return standardized;
}
```

Nella pipeline attuale la binarizzazione avviene direttamente sul vettore proiettato con soglia 0, senza standardizzazione.

---

## 5. Binarizzazione e bit‑packing

### 5.1 Funzione segno (soglia 0)

$$
b_d = \begin{cases} 1 & \text{se } v_d > 0 \\ 0 & \text{se } v_d \leq 0 \end{cases}
$$

dove $b_d = 1$ rappresenta $+1$, $b_d = 0$ rappresenta $-1$.

La binarizzazione è disponibile in due forme equivalenti: in `MainActivity` e in `RandomProjectionBridge`:

```java
// MainActivity.java:550-557 (versione usata in pipeline)
private void binarizeToBlocks(double[] continuous, long[] blocks) {
    java.util.Arrays.fill(blocks, 0L);
    for (int d = 0; d < VSA_DIM; d++) {
        if (continuous[d] > 0) {
            blocks[d / 64] |= (1L << (d % 64));
        }
    }
}
```

Il bridge offre un equivalente `binarizeTo(double[], long[])` (vedi §3.5) con identica logica, utile per test standalone.

### 5.2 Packing in `long[]`

10 000 dimensioni → $\lceil 10000 / 64 \rceil = 157$ elementi `long`:

```java
// VsaVector.java:9-10
public static final int DIMENSIONS = 10000;
private static final int ARRAY_SIZE = (DIMENSIONS + 63) / 64;  // = 157
```

L'ultimo long (`blocks[156]`) ha solo 16 bit significativi; i restanti 48 sono 0 e non influenzano la distanza di Hamming.

### 5.3 Wrapper per apprendimento

```java
// MainActivity.java:559-563
private VsaVector binarize(double[] continuous) {
    long[] blocks = new long[157];
    binarizeToBlocks(continuous, blocks);
    return new VsaVector(blocks);
}
```

### 5.4 VsaVector.setBlocks() zero‑allocation

```java
// VsaVector.java:72-76
public void setBlocks(long[] newBlocks) {
    if (newBlocks != null && newBlocks.length == 157) {
        System.arraycopy(newBlocks, 0, this.blocks, 0, 157);
    }
}
```

---

## 6. Distanza di Hamming e confronto con gli archetipi

### 6.1 XOR + Long.bitCount() (POPCNT hardware)

```java
// VsaVector.java:41-52
public int calculateHammingDistance(VsaVector other) {
    int distance = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        long differences = this.blocks[i] ^ other.blocks[i];
        distance += Long.bitCount(differences);   // POPCNT: CNT+ADDV su ARMv8
    }
    return distance;
}
```

### 6.2 Conversione in similarità coseno

$$
\cos\theta = 1 - \frac{2h}{D}, \quad D = 10000
$$

```java
// VsaVector.java:58-61
public double getCosineSimilarity(VsaVector other) {
    int h = calculateHammingDistance(other);
    return 1.0 - 2.0 * ((double) h / DIMENSIONS);
}
```

### 6.3 Caricamento e persistenza archetipi

Due mappe in memoria:

| Mappa | Chiave | Valore | Uso |
|-------|--------|--------|-----|
| `archetypesMAP` | `String` (nome classe) | `double[10000]` | Persistenza + learning |
| `archetypesBSC` | `String` (nome classe) | `VsaVector` | Matching veloce |

**File**: `archetype_map.bin` in `getFilesDir()` (internal storage).

**Formato binario**:

| Offset | Tipo | Contenuto |
|--------|------|-----------|
| 0 | `int` | `numClasses` |
| 4 | `int` | `VSA_DIM` (10000) |
| 8 | variabile | Per ogni classe: `UTF` nome + `double[10000]` |

**Caricamento con late binarization**:

```java
// MainActivity.java:162-166
loadAllArchetypesMAP();
if (!archetypesMAP.isEmpty()) {
    for (Map.Entry<String, double[]> e : archetypesMAP.entrySet()) {
        archetypesBSC.put(e.getKey(), binarize(e.getValue()));
    }
}
```

All'avvio il file viene **sempre cancellato** per un test pulito:

```java
// MainActivity.java:156-160
java.io.File oldFile = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
if (oldFile.exists()) {
    oldFile.delete();
    Log.d(TAG, "Archetipi precedenti cancellati — test pulito");
}
```

### 6.4 Soglia di rifiuto

Hard‑coded: **0.35** cosine similarity. Se nessun archetipo la supera, la UI mostra "Ricerca in corso...".

### 6.5 Matching

```java
// MainActivity.java:325-334 (dentro processCameraFrame, ramo inferenza)
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
}
```

### 6.6 File asset `vsa_archetypes.bin`

Esiste negli assets ma **non caricato** dalla pipeline attuale. Residuo di versioni precedenti.

---

## 7. Pipeline completa — Riepilogo a diagramma

```
                        ┌─────────────────────────────────────────────────┐
                        │              CAMERA FRAME (CameraX)              │
                        │        ImageProxy → Bitmap (es. 640×480)        │
                        └────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                   ┌─────────────────────────────────────────────────┐
                   │           STAGE 1 — OBJECT DETECTION             │
                   │   SsdExtractor.detectObject(bitmap)             │
                   │   • MediaPipe ObjectDetector                    │
                   │   • Tenta GPU (Delegate.GPU), fallback CPU      │
                   │   • Modello: mobilenet_ssd.tflite               │
                   │   • Input: [256, 256, 3] float32                │
                   │   • Score threshold: 0.3, max 1 result          │
                   └────────────────────┬────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │ box == null?              │
                          └─────────────┬─────────────┘
                                 Sì     │     No
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
           UI: "Ricerca..."              ┌──────────────────────────────────┐
                                        │      EXPAND (+20%) + LETTERBOX    │
                                        │      • Sanity check (1%-80% area) │
                                        │      • Ritaglia + centro su nero  │
                                        └────────────────┬─────────────────┘
                                                         │
                                                         ▼
                   ┌─────────────────────────────────────────────────┐
                   │         STAGE 2 — CNN FEATURE EXTRACTION         │
                   │   FeatureEncoder.extract(square)                │
                   │   • resize 224×224 bilineare                    │
                   │   • Normalizzazione [-1.0, 1.0] per canale RGB  │
                   │   • TFLite Interpreter (GPU o CPU 4‑thread)     │
                   │   • Modello: mobilenet_vsa.tflite               │
                   │   • Output: float[1280]                         │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │            DEBINDING (black padding)             │
                   │   applyDebinding(features, boxPx)               │
                   │   • Sottrae firma padding ∝ paddingRatio        │
                   │   • Clamp a max(0, value)                       │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │         STAGE 3 — VSA RANDOM PROJECTION         │
                   │   RandomProjectionBridge.projectFeatureVector() │
                   │   • Matrice boolean[10000][1280], seed=42       │
                   │   • Zero‑skipping su feature ReLU               │
                   │   • Prodotto scalare ±1                         │
                   │   • Output: double[10000]                       │
                   └────────────────────┬────────────────────────────┘
                                        │
                          ┌─────────────┴──────────────┐
                          │  Stato LearningSession?    │
                          └─────────────┬──────────────┘
                      COLLECTING         │          IDLE / PROCESSING
                    ┌────────────────────┴──────────────────────┐
                    ▼                                           ▼
    ┌──────────────────────────────┐       ┌──────────────────────────────┐
    │   MULTI‑SHOT LEARNING        │       │   INFERENZA STANDARD          │
    │   • normalizeL2(buffer)      │       │   • binarizeToBlocks → BSC   │
    │   • addFrame() → accumula    │       │   • Matching vs archetypesBSC │
    │   • Update ProgressBar       │       │   • soglia cos > 0.35         │
    │   • Se target raggiunto:     │       │   • Stabilizer temporale      │
    │     → getFinalBundleAndReset │       └──────────────┬───────────────┘
    │     → normalizeL2(bundle)    │                      │
    │     → save MAP + BSC          │                      ▼
    └──────────────────────────────┘       ┌──────────────────────────────┐
                                           │         UI RENDERING          │
                                           │  • tvVerdict + tvConfidence   │
                                           │  • TrackingOverlayView        │
                                           │  • Rotazione box per display  │
                                           └──────────────────────────────┘
```

---

## 8. Addestramento / creazione archetipi

### 8.1 Flusso Multi‑Shot Learning (on‑device)

**Non più one‑shot tramite ImageCapture!** Ora il learning è multi‑shot in streaming dai frame di ImageAnalysis.

#### Passo 1: Input del nome e avvio sessione

L'utente preme il FAB "+" (in stato IDLE). Un `AlertDialog` chiede il nome. Al click "Avvia Acquisizione":

```java
// MainActivity.java:397-425
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
            learningSessionManager.startLearning(name);   // → State: COLLECTING
            runOnUiThread(() -> {
                progressLearning.setVisibility(View.VISIBLE);
                progressLearning.setProgress(0);
                fabLearn.setImageResource(android.R.drawable.ic_media_pause);
                Toast.makeText(MainActivity.this,
                        "Inquadra l'oggetto. Premi + per terminare.", Toast.LENGTH_LONG).show();
            });
        }
    });
    builder.setNegativeButton("Annulla", (dialog, which) -> dialog.cancel());
    builder.show();
}
```

Lo stato del FAB cambia in base al contesto:

| Stato LearningSession | Tap FAB | Long‑press FAB |
|----------------------|---------|----------------|
| `IDLE` | Mostra dialog nome | Pulisci archetipi |
| `COLLECTING` | `finishLearningEarly()` | `cancelLearning()` |
| `PROCESSING` / `CANCELLED` | Mostra dialog nome | Pulisci archetipi |

#### Passo 2: Accumulazione frame (BIVIO nella pipeline)

Durante l'elaborazione di ogni frame, `processCameraFrame()` verifica se lo stato è `COLLECTING`. Se sì, **invece di binarizzare e confrontare**, normalizza L2 e passa il vettore al `LearningSessionManager`:

```java
// MainActivity.java:294-321 (bivio in processCameraFrame)
if (learningSessionManager.getCurrentState() == State.COLLECTING) {

    vsaBridge.normalizeL2(continuousBuffer);              // (1) L2 normalization

    boolean isDone = learningSessionManager.addFrame(continuousBuffer); // (2) accumula
    int collected = learningSessionManager.getFramesCollected();
    int target = learningSessionManager.getTargetFrames();
    String pName = learningSessionManager.getPendingClassName();
    float prog = learningSessionManager.getProgress();

    runOnUiThread(() -> {
        progressLearning.setProgress((int)(prog * 100));
        tvVerdict.setText("Learning: " + pName);
        tvConfidence.setText("Shot " + collected + "/" + target + " — muovi leggermente");
    });

    if (isDone) {
        double[] finalBundle = learningSessionManager.getFinalBundleAndReset();
        saveBundleAsArchetype(pName, finalBundle);        // (3) salva
    }
    frame.recycle();
    return;  // ← esce senza eseguire inferenza/stabilizzazione
}
```

#### Passo 3: Bundling additivo

`LearningSessionManager.addFrame()` somma il vettore L2‑normalizzato in un accumulatore `double[10000]`:

```java
// LearningSessionManager.java:67-87
public synchronized boolean addFrame(double[] normalizedContinuousVSA) {
    if (currentState != State.COLLECTING) return false;
    for (int i = 0; i < vsaDimensions; i++) {
        bundleVector[i] += normalizedContinuousVSA[i];
    }
    framesCollected++;
    if (framesCollected >= targetFrames) {
        this.currentState = State.PROCESSING;
        return true;     // target raggiunto
    }
    return false;
}
```

**Equazione del bundling** (somma additiva di vettori L2‑normalizzati):

$$
\mathbf{b}_{\text{bundle}} = \sum_{k=1}^{N} \frac{\mathbf{v}_k}{\|\mathbf{v}_k\|_2}, \quad N = \text{targetFrames} = 8 \text{ (default)}
$$

#### Passo 4: Salvataggio archetipo

```java
// MainActivity.java:453-476
private void saveBundleAsArchetype(String className, double[] bundle) {
    vsaBridge.normalizeL2(bundle);                     // (1) normalizza il bundle

    archetypesMAP.put(className, bundle);              // (2) salva continuo (MAP)
    VsaVector archetypeBSC = binarize(bundle);         // (3) binarizza in BSC
    archetypesBSC.put(className, archetypeBSC);
    saveAllArchetypesMAP();                            // (4) persisti su file

    int activeBits = 0;
    for (int i = 0; i < 157; i++) {
        activeBits += Long.bitCount(archetypeBSC.getBlocks()[i]);
    }
    Log.i(TAG, "ARCHETIPO CREATO: " + className + " | Bit attivi: " + activeBits + "/10000");

    runOnUiThread(() -> {
        progressLearning.setVisibility(View.GONE);
        fabLearn.setImageResource(android.R.drawable.ic_input_add);
        Toast.makeText(MainActivity.this, "Appreso: " + className, Toast.LENGTH_SHORT).show();
        tvVerdict.setText(className.replace("_", " "));
        tvConfidence.setText(gpuConfigLabel);
    });
}
```

### 8.2 Completamento anticipato

L'utente può premere il FAB durante la raccolta per terminare con i sample già accumulati:

```java
// MainActivity.java:427-440
private void finishLearningEarly() {
    double[] finalBundle = learningSessionManager.finish();  // forza PROCESSING
    if (finalBundle != null) {
        saveBundleAsArchetype(learningSessionManager.getPendingClassName(), finalBundle);
    } else {
        // nessun sample raccolto → mostra errore
    }
}
```

### 8.3 Cancellazione sessione

```java
// MainActivity.java:442-451
private void cancelLearning() {
    learningSessionManager.cancel();
    runOnUiThread(() -> {
        progressLearning.setVisibility(View.GONE);
        fabLearn.setImageResource(android.R.drawable.ic_input_add);
        Toast.makeText(MainActivity.this, "Apprendimento annullato", Toast.LENGTH_SHORT).show();
        tvVerdict.setText("Inquadra un oggetto...");
        tvConfidence.setText(gpuConfigLabel);
    });
}
```

### 8.4 Cancellazione archetipi

Long‑press in stato IDLE:

```java
// MainActivity.java:113-128
// Pulisce entrambe le mappe e cancella archetype_map.bin
// UI: "Tabula Rasa. Premi + per imparare."
```

### 8.5 Debinding del padding nero

La firma del padding nero viene calcolata all'avvio e sottratta proporzionalmente al `paddingRatio` sia in inferenza che in apprendimento:

```java
// MainActivity.java:142-145 (calcolo firma all'avvio)
Bitmap black = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
new Canvas(black).drawColor(Color.BLACK);
blackPaddingSignature = encoder.extract(black);    // float[1280]
black.recycle();

// MainActivity.java:536-548 (applicazione)
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
```

### 8.6 Binding (XOR) — disponibile ma non usato

```java
// VsaVector.java:28-35
public VsaVector bind(VsaVector other) {
    long[] result = new long[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        result[i] = this.blocks[i] ^ other.blocks[i];
    }
    return new VsaVector(result);
}
```

Il binding XOR esiste come operatore VSA ma non è attualmente chiamato da nessuna parte della pipeline.

### 8.7 Formato persistenza

L'unico formato è `archetype_map.bin` (vedi §6.3). Nessuna esportazione CSV o matrice di proiezione.

---

## 9. Validazione

### 9.1 Accuratezza sul dispositivo

**Non esiste un modulo di validazione formale.** I file di test sono placeholder:

```
app/src/test/java/com/antonio/vsatest/ExampleUnitTest.java          → vuoto
app/src/androidTest/java/com/antonio/vsatest/ExampleInstrumentedTest.java → vuoto
```

### 9.2 Logging diagnostico

La validazione empirica avviene tramite log di debug (`VSA_TwoStage`). Per ogni archetipo creato, viene loggato il numero di bit attivi:

```java
// MainActivity.java:463-467
int activeBits = 0;
for (int i = 0; i < 157; i++) {
    activeBits += Long.bitCount(archetypeBSC.getBlocks()[i]);
}
Log.i(TAG, "ARCHETIPO CREATO: " + className + " | Bit attivi: " + activeBits + "/10000");
```

In modalità inferenza, il verdetto e la similarità sono mostrati direttamente nella UI e nel log:

```
Verdict: "Vite M5 (0.82)"
Confidence: "SSD:GPU Enc:GPU | SSD+45ms VSA:12ms TOT:67ms"
```

### 9.3 UI confidence e ProgressBar

La `ProgressBar` orizzontale (verde, `progressTint="#2E7D32"`) mostra il progresso durante l'apprendimento:

```java
// activity_main.xml:63-69
<ProgressBar
    android:id="@+id/progressLearning"
    style="?android:attr/progressBarStyleHorizontal"
    android:layout_width="match_parent"
    android:layout_height="6dp"
    android:visibility="gone"
    android:progressTint="#2E7D32" />
```

---

## 10. Esportazione della matrice di proiezione

**Non implementata.** La matrice `boolean[10000][1280]` è generata al volo con seed `42L` e mai esportata. La riproducibilità è garantita dal seed deterministico.

---

## 11. Riepilogo dei file chiave e delle classi

| File / Componente | Ruolo | Package | Percorso |
|-------------------|-------|---------|----------|
| **MainActivity.java** | Orchestratore: camera, pipeline two‑stage, multi‑shot learning, bivio learning/inference, persistenza, debinding, binarizzazione | `ui` | `app/src/main/java/.../visionsector/ui/MainActivity.java` |
| **LearningSessionManager.java** | Stato macchina multi‑shot: IDLE→COLLECTING→PROCESSING; bundling additivo, progresso, cancel/finish | `engine.learning` | `app/src/main/java/.../visionsector/engine/learning/LearningSessionManager.java` |
| **SsdExtractor.java** | Stadio 1: SSD Object Detection via MediaPipe Tasks, GPU/CPU fallback | `detector` | `app/src/main/java/.../visionsector/detector/SsdExtractor.java` |
| **FeatureEncoder.java** | Stadio 2: MobileNetV2 encoder TFLite, GPU/CPU fallback (GpuDelegate) | `encoder` | `app/src/main/java/.../visionsector/encoder/FeatureEncoder.java` |
| **RandomProjectionBridge.java** | Proiezione 1280→10000 (±1), zero‑skipping, L2 normalization, projectToContinuous, binarizeTo(), projectSpatialCellToContinuous | `engine.vsa` | `app/src/main/java/.../visionsector/engine/vsa/RandomProjectionBridge.java` |
| **VsaVector.java** | Ipervettore BSC: XOR binding, Hamming (POPCNT), cosine similarity, setBlocks zero‑alloc | `engine.vsa` | `app/src/main/java/.../visionsector/engine/vsa/VsaVector.java` |
| **DatasetStatTracker.java** | Z‑Score standardization (pronto, non cablato) | `engine.vsa` | `app/src/main/java/.../visionsector/engine/vsa/DatasetStatTracker.java` |
| **DetectionStabilizer.java** | EMA smoothing spaziale + majority voting semantico + coasting | `ui` | `app/src/main/java/.../visionsector/ui/DetectionStabilizer.java` |
| **TrackingOverlayView.java** | View bounding box con aspect‑ratio fit, dim mask, etichette | `ui` | `app/src/main/java/.../visionsector/ui/TrackingOverlayView.java` |
| **AntiGlareFilter.java** | Filtro LUT gamma 1.4 (pronto, non cablato) | `utils` | `app/src/main/java/.../visionsector/utils/AntiGlareFilter.java` |
| **mobilenet_ssd.tflite** | Modello SSD Object Detection | assets | `app/src/main/assets/mobilenet_ssd.tflite` |
| **mobilenet_vsa.tflite** | Modello MobileNetV2 encoder feature | assets | `app/src/main/assets/mobilenet_vsa.tflite` |
| **vsa_archetypes.bin** | Archetipi pre‑addestrati (asset, non caricati) | assets | `app/src/main/assets/vsa_archetypes.bin` |
| **vsa_stats.bin** | Statistiche Z‑Score (asset, non usate) | assets | `app/src/main/assets/vsa_stats.bin` |
| **archetype_map.bin** | Persistenza runtime archetipi | internal storage | `getFilesDir()/archetype_map.bin` |
| **activity_main.xml** | Layout: PreviewView, FAB, CardView, ProgressBar, TrackingOverlayView | res | `app/src/main/res/layout/activity_main.xml` |
| **build.gradle.kts** | Dipendenze: TFLite 2.16.1 + GPU, MediaPipe 0.10.14 (escluso LiteRT), CameraX 1.3.4 | | `app/build.gradle.kts` |

---

## 12. Differenze rispetto alla versione Java desktop

| Aspetto | Java Desktop | Android |
|---------|-------------|---------|
| **Runtime CNN** | TensorFlow SavedModel (Java API) | TensorFlow Lite (`Interpreter`, `MappedByteBuffer`) |
| **Accelerazione hardware** | CPU only / CUDA | GPU Delegate (OpenCL/GLES) con fallback CPU; MediaPipe Delegate.GPU per SSD |
| **Formato modello** | SavedModel directory | `.tflite` FlatBuffer (~2.5 MB encoder, ~4 MB SSD) |
| **Object Detection** | Non presente o separata | Stadio 1 integrato: SSD MobileNetV2 via MediaPipe Tasks |
| **Origine immagine** | File su disco | CameraX in tempo reale (`ImageProxy → Bitmap`), nessun ImageCapture |
| **Pre‑processing** | `BufferedImage`, `resize()` | `Bitmap.createScaledBitmap`, `getPixels()`, `ByteBuffer.allocateDirect` |
| **Memory budget** | Heap abbondante (GB) | Mobile‑constrained. Buffer pre‑allocati, `arraycopy` zero‑alloc, bitmap recycling aggressivo |
| **Matrice proiezione** | `boolean[10000][1280]` su heap | Identica (~12.8 MB) |
| **Zero‑skipping** | Probabilmente assente | Implementato: itera solo su feature ≠ 0 |
| **Normalizzazione L2** | Fissa in pipeline | Solo durante apprendimento; l'inferenza binarizza senza L2 |
| **Standardizzazione Z‑Score** | Probabilmente integrata | Classe `DatasetStatTracker` pronta ma **non cablata** |
| **Binarizzazione** | Soglia 0, packing `long[]` | Identico, buffer pre‑allocato riutilizzato |
| **POPCNT hardware** | `Long.bitCount()` → `POPCNT` x86 | `Long.bitCount()` → `CNT`+`ADDV` su ARMv8; software su ARMv7 |
| **Threading** | Main thread o pool | `SingleThreadExecutor` dedicato per inferenza, UI thread separato |
| **Persistenza archetipi** | File binario su disco | `DataOutputStream` su `FileOutputStream` in `getFilesDir()` |
| **Apprendimento** | Batch multi‑sample (cache feature, bundling, late binarization) | **Multi‑shot streaming**: accumula frame da ImageAnalysis in tempo reale, bundling additivo di vettori L2‑normalizzati, target configurabile (default 8) |
| **Bundling** | Somma vettori → late binarization | Somma additiva L2‑normalizzati, poi L2 sul bundle finale → binarize |
| **Binding (XOR)** | Usato per associare ruoli | Disponibile in `VsaVector.bind()` ma **non chiamato** |
| **Debinding** | Non presente | Implementato: sottrae firma del padding nero ∝ `paddingRatio` |
| **Stabilizzazione temporale** | Non necessaria | `DetectionStabilizer`: EMA box + majority voting + coasting |
| **Anti‑glare** | Non presente | `AntiGlareFilter` (LUT gamma 1.4) pronto ma non cablato |
| **Validazione** | Misura accuratezza su dataset | Nessun modulo di test; logging diagnostico e bit attivi |
| **ProgressBar apprendimento** | Non applicabile | UI con ProgressBar orizzontale durante la raccolta frame |
| **Esportazione matrice** | Probabilmente supportata (CSV) | Non implementata |
| **Rotazione display** | Non applicabile | Gestione 0°/90°/180°/270° del sensore fotocamera |
| **Dependency hell** | Non applicabile | Risolto: escluso `litert-api` da MediaPipe, escluso `tflite-api` da GPU |
| **Struttura package** | Package piatto o simile | 6 sub‑package: `ui`, `detector`, `encoder`, `engine.vsa`, `engine.learning`, `utils` |
| **Soglia matching** | Probabilmente basata su 3σ | Hard‑coded `0.35` cosine |
| **Numero thread TFLite** | Variabile | GPU: nessun thread CPU; CPU fallback: 4 thread |
| **Spatial grid (7×7)** | Probabilmente usata | Supportata da `projectSpatialCellToContinuous()` ma non attiva |

---

*Documento generato il 2026‑06‑20 dall'analisi completa del codice sorgente Android.*
