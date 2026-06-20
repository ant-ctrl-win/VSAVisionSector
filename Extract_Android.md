# Extract_Android.md — Pipeline VSA su Android (Two‑Stage Cascade)

**App**: VsaVisionSector — Object Detection + 1‑Shot Learning basato su VSA/BSC
**Piattaforma**: Android (TFLite, CameraX, MediaPipe)  
**Linguaggio**: Java 11  
**Data analisi**: 2026‑06‑20

---

## 1. Caricamento e pre‑processamento dell'immagine

### 1.1 Origine dell'immagine

L'immagine proviene **esclusivamente dalla fotocamera**, tramite CameraX. Non è implementata la selezione da galleria.

```java
// MainActivity.java:189-191
ImageAnalysis analysis = new ImageAnalysis.Builder()
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build();
```

L'`ImageAnalysis` è configurato con `STRATEGY_KEEP_ONLY_LATEST` per evitare accumulo di frame. L'analyzer viene eseguito su un `ExecutorService` single‑thread:

```java
// MainActivity.java:63
private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();

// MainActivity.java:193-211
analysis.setAnalyzer(inferenceExecutor, imageProxy -> {
    if (isProcessingFrame) { imageProxy.close(); return; }
    isProcessingFrame = true;
    try {
        Bitmap frame = imageProxy.toBitmap();
        // ...
    } finally {
        imageProxy.close();
        isProcessingFrame = false;
    }
});
```

Il flag `isProcessingFrame` evita accodamento di frame durante l'elaborazione.

La fotocamera usata è quella posteriore:

```java
// MainActivity.java:214
CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
```

### 1.2 Object Detection (Primo Stadio)

Lo stadio 1 è un **SSD MobileNetV2 class‑agnostic** (addestrato con MediaPipe Model Maker) che produce **bounding box normalizzati [0.0, 1.0]**. Il modello è `mobilenet_ssd.tflite` (assets):

```
Input:  float32[1, 256, 256, 3]
Output: boxes  float32[1, 12276, 4]   (raw anchor offsets)
        scores float32[1, 12276, 2]   (index 1 = object confidence)
```

Non si usa l'Interpreter TFLite grezzo: il decoder, la NMS e il ridimensionamento sono gestiti internamente da **MediaPipe Tasks Vision** (`ObjectDetector`).

```java
// SsdExtractor.java:41-56
public SsdExtractor(Context context, String modelPath) throws Exception {
    BaseOptions baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelPath)         // "mobilenet_ssd.tflite"
            .build();
    ObjectDetector.ObjectDetectorOptions options =
            ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)             // solo la detection migliore
                    .setScoreThreshold(0.3f)      // soglia confidenza
                    .setRunningMode(RunningMode.IMAGE)
                    .build();
    detector = ObjectDetector.createFromOptions(context, options);
}
```

Il metodo `detectObject()` converte il bitmap in `MPImage` e restituisce il box normalizzato:

```java
// SsdExtractor.java:65-85
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

Il box SSD viene espanso del 20% per garantire contesto attorno all'oggetto:

```java
// MainActivity.java:340-348
private RectF expandBox(RectF box, float ratio) {
    float w = box.width() * ratio;
    float h = box.height() * ratio;
    return new RectF(
        Math.max(0f, box.left - w),
        Math.max(0f, box.top - h),
        Math.min(1f, box.right + w),
        Math.min(1f, box.bottom + h)
    );
}
```

### 1.4 Sanity check del box

Si scartano box troppo piccoli (<1% area) o troppo grandi (>80% area):

```java
// MainActivity.java:351-356
private boolean isBoxSane(RectF box) {
    if (box == null) return false;
    float area = box.width() * box.height();
    if (area < 0) area = -area;
    return area >= 0.01f && area <= 0.80f;
}
```

### 1.5 Ritaglio e Letterbox

Il box in pixel viene ritagliato dal frame originale e incollato al centro di un **quadrato nero** (`letterboxCrop`), il cui lato è `max(crop.width, crop.height)`. Questo preserva l'aspect ratio dell'oggetto ed evita distorsioni.

```java
// MainActivity.java:358-381
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
    canvas.drawColor(Color.BLACK);                      // sfondo nero
    int offsetX = (squareSize - cropW) / 2;
    int offsetY = (squareSize - cropH) / 2;
    canvas.drawBitmap(crop, offsetX, offsetY, null);    // crop centrato
    crop.recycle();
    return square;
}
```

### 1.6 Ridimensionamento a 224×224

Il `FeatureEncoder` ridimensiona il quadrato letterbox a 224×224 usando `Bitmap.createScaledBitmap()` con filtro bilineare (`true`).

```java
// FeatureEncoder.java:43
Bitmap resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true);
```

### 1.7 Normalizzazione per TFLite

I pixel ARGB_8888 vengono letti con `getPixels()` e normalizzati nell'intervallo **[-1.0, 1.0]** secondo la formula standard MobileNet:

$$
\text{value} = \frac{\text{channel} - 127.5}{127.5}
$$

```java
// FeatureEncoder.java:52-56
for (int pixel : pixels) {
    inputBuffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f); // R
    inputBuffer.putFloat((((pixel >> 8)  & 0xFF) - 127.5f) / 127.5f); // G
    inputBuffer.putFloat((( pixel        & 0xFF) - 127.5f) / 127.5f); // B
}
```

### 1.8 Conversione in input TFLite

L'input buffer è un `ByteBuffer` diretto (`allocateDirect`) in ordine nativo, layout **NHWC** (224×224×3 canali interleaved):

```java
// FeatureEncoder.java:45-46
ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
inputBuffer.order(ByteOrder.nativeOrder());
```

Dimensioni buffer: 224 × 224 × 3 × 4 byte/float = **602 112 byte** (~588 KB).

---

## 2. Estrazione delle feature CNN

### 2.1 Modello utilizzato

**MobileNetV2** (probabilmente con α=1.0, global average pooling), esportato in formato `.tflite`. Il file asset è `mobilenet_vsa.tflite` (~2.5 MB).

| Proprietà | Valore |
|-----------|--------|
| File | `app/src/main/assets/mobilenet_vsa.tflite` |
| Formato | TensorFlow Lite FlatBuffer |
| Input | `float32[1, 224, 224, 3]` (NHWC) |
| Output | `float32[1, 1280]` |
| Normalizzazione | [-1.0, 1.0] |

### 2.2 Caricamento dell'interprete TFLite

Il modello viene caricato dagli asset Android usando `MappedByteBuffer` (memory‑mapped file descriptor), che evita di copiare l'intero modello in heap:

```java
// FeatureEncoder.java:26-30
public FeatureEncoder(Context context, String modelPath) throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    options.setNumThreads(4);               // sfrutta CPU multi‑core
    tflite = new Interpreter(loadModelFile(context, modelPath), options);
}

// FeatureEncoder.java:32-37
private MappedByteBuffer loadModelFile(Context context, String modelPath) throws Exception {
    AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
    FileInputStream is = new FileInputStream(fd.getFileDescriptor());
    FileChannel ch = is.getChannel();
    return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
}
```

**Numero thread**: 4.  
**Modelli esclusi dalla compressione APK**: `noCompress.add("tflite")` nel `build.gradle.kts`.

### 2.3 Output tensor

Il modello ha un singolo tensore di output (non ne viene specificato il nome, l'interprete usa il primo/default). Forma: `float[1][1280]`.

```java
// FeatureEncoder.java:58-61
float[][] output = new float[1][FEATURE_DIM];  // float[1][1280]
tflite.run(inputBuffer, output);
return output[0];                              // float[1280]
```

L'output è un vettore di 1280 float, frutto del global average pooling del backbone MobileNetV2 (penultimo strato prima della classificazione).

### 2.4 Gestione buffer nativi

- **ByteBuffer input**: allocato a ogni chiamata `extract()` (`allocateDirect`). Dimensione: 602 KB. Non riciclato tra chiamate.
- **ByteBuffer output**: gestito internamente da TFLite (l'API `tflite.run(input, output)` alloca internamente il buffer di output corrispondente al tensore).
- **Bitmap**: il `Bitmap.createScaledBitmap(224×224)` viene riciclato dopo `getPixels()` con `resized.recycle()`.

---

## 3. Proiezione casuale 1280 → 10 000

### 3.1 Generazione della matrice di proiezione

La matrice è di tipo `boolean[][]` con layout **trasposto**: `[10000][1280]` invece di `[1280][10000]`. Ciò garantisce cache locality: ogni riga di 1280 boolean è contigua in memoria e viene attraversata sequenzialmente.

```java
// RandomProjectionBridge.java:9-10,12-28
private final boolean[][] projectionMatrix;  // [10000][1280]

public RandomProjectionBridge(int cnnFeatureSize, long seed) {
    this.cnnFeatureSize = cnnFeatureSize;          // 1280
    this.vsaDimensions = 10000;                    // 10000
    this.projectionMatrix = new boolean[vsaDimensions][cnnFeatureSize];

    Random random = new Random(seed);

    for (int i = 0; i < cnnFeatureSize; i++) {
        for (int d = 0; d < vsaDimensions; d++) {
            boolean casuale = random.nextBoolean();
            projectionMatrix[d][i] = casuale;      // memorizzato in modo trasposto
        }
    }
}
```

**Seed deterministico**: `42L`, garantisce riproducibilità.  
**Dimensione memoria matrice**: `10000 × 1280` boolean ≈ 12.8 MB in heap (ogni `boolean` in Java occupa 1 byte nell'array).

Il loop di generazione estrae i bit nell'ordine **originale** (stesso della versione Java desktop): per ogni feature i, per ogni dimensione d, estrae `nextBoolean()`. Il salvataggio trasposto `projectionMatrix[d][i]` è un'ottimizzazione di layout; l'ordine di estrazione dal PRNG rimane identico.

### 3.2 Algoritmo del prodotto scalare `±1`

Ad ogni dimensione VSA $d \in [0, 9999]$ corrisponde un vettore base $\mathbf{b}_d \in \{-1, +1\}^{1280}$ (riga della matrice booleana). Il valore proiettato è:

$$
v_d = \sum_{i=0}^{1279} s_i \cdot f_i \quad \text{con } s_i = \begin{cases} +1 & \text{se } \text{projectionMatrix}[d][i] = \texttt{true} \\ -1 & \text{se } \text{projectionMatrix}[d][i] = \texttt{false} \end{cases}
$$

```java
// RandomProjectionBridge.java:103-128
public void projectFeatureVector(float[] features, double[] outContinuous) {
    int[] activeIndices = new int[cnnFeatureSize];
    float[] activeValues = new float[cnnFeatureSize];
    int activeCount = 0;

    // 1. Raccoglie solo le feature non‑nulle (zero‑skipping)
    for (int i = 0; i < cnnFeatureSize; i++) {
        float val = features[i];
        if (val != 0.0f) {
            activeIndices[activeCount] = i;
            activeValues[activeCount] = val;
            activeCount++;
        }
    }

    // 2. Proiezione: solo sui filtri attivi
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

**Ottimizzazione zero‑skipping**: poiché dopo ReLU molte feature sono esattamente 0.0f, il loop interno viene eseguito solo sulle feature attive, riducendo i cicli CPU fino all'80%.

**Versione alternativa con L2 normalization integrata** (`projectToContinuous`):

```java
// RandomProjectionBridge.java:33-61
public double[] projectToContinuous(float[] cnnFeatures, double[] reuseVector) {
    double sumOfSquares = 0.0;
    for (int d = 0; d < vsaDimensions; d++) {
        double sum = 0.0;
        boolean[] row = projectionMatrix[d];
        for (int i = 0; i < cnnFeatureSize; i++) {
            if (row[i]) sum += cnnFeatures[i];
            else sum -= cnnFeatures[i];
        }
        reuseVector[d] = sum;
        sumOfSquares += sum * sum;
    }
    double magnitude = Math.sqrt(sumOfSquares);
    if (magnitude > 1e-9) {
        for (int d = 0; d < vsaDimensions; d++) reuseVector[d] /= magnitude;
    }
    return reuseVector;
}
```

Questa versione **non usa zero‑skipping** ma applica la normalizzazione L2 in‑line. È usata implicitamente nell'apprendimento? No — l'apprendimento chiama `projectFeatureVector` + `normalizeL2` separatamente.

### 3.3 Normalizzazione L2

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

**Quando si applica**: solo durante l'apprendimento (salvataggio MAP). Durante l'inferenza la pipeline **non normalizza** prima della binarizzazione.

### 3.4 Gestione memoria su Android

| Buffer | Tipo | Dimensione | Note |
|--------|------|-----------|------|
| `projectionMatrix` | `boolean[10000][1280]` | ~12.8 MB | Allocato una volta, persistente |
| `continuousBuffer` | `double[10000]` | ~80 KB | Pre‑allocato in `MainActivity`, riusato |
| `activeIndices/value` | `int[1280]` + `float[1280]` | ~10 KB | Allocati a ogni chiamata (temporanei) |

La matrice da 12.8 MB è il costo maggiore in RAM, ma accettabile su dispositivi moderni.

---

## 4. Standardizzazione Z‑Score (DatasetStatTracker)

### 4.1 Classe e caricamento

La classe `DatasetStatTracker` esiste nel progetto ma **non è attualmente integrata nella pipeline attiva**. Carica un file binario `vsa_stats.bin` dagli asset contenente media e deviazione standard pre‑calcolate.

```java
// DatasetStatTracker.java:39-52
public static DatasetStatTracker importFromAndroidAssets(Context context, String fileName)
        throws IOException {
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

### 4.2 Formato del file `vsa_stats.bin`

| Offset | Tipo | Contenuto |
|--------|------|-----------|
| 0 | `int` (4 byte) | Numero dimensioni `dims` |
| 4 | `double[dims]` (8×dims byte) | Vettore `mean` |
| 4+8×dims | `double[dims]` (8×dims byte) | Vettore `stdDev` |

### 4.3 Applicazione dello Z‑Score (non in uso)

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

**Nota**: nella pipeline attuale la standardizzazione è **bypassata**. La binarizzazione avviene direttamente sul vettore proiettato non standardizzato con soglia 0.

### 4.4 Provenienza dei parametri

I parametri media/stdDev sono calcolati **offline** (presumibilmente dalla versione Java desktop) ed esportati nel file binario incluso negli asset. La classe è pronta per un futuro wiring.

---

## 5. Binarizzazione e bit‑packing

### 5.1 Funzione segno (soglia 0)

La binarizzazione BSC (Bipolar Scrambled Code) applica la funzione segno con soglia 0:

$$
b_d = \begin{cases} 1 & \text{se } v_d > 0 \\ 0 & \text{se } v_d \leq 0 \end{cases}
$$

dove $b_d = 1$ rappresenta $+1$ e $b_d = 0$ rappresenta $-1$ nella codifica BSC.

```java
// MainActivity.java:408-415
private void binarizeToBlocks(double[] continuous, long[] blocks) {
    java.util.Arrays.fill(blocks, 0L);
    for (int d = 0; d < VSA_DIM; d++) {
        if (continuous[d] > 0) {
            blocks[d / 64] |= (1L << (d % 64));
        }
    }
}
```

Esiste anche una versione equivalente in `RandomProjectionBridge`:

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

### 5.2 Packing in `long[]`

Le 10 000 dimensioni binarie sono compattate in $\lceil 10000 / 64 \rceil = 157$ elementi `long` (64 bit ciascuno). La dimensione è calcolata staticamente:

```java
// VsaVector.java:9-10
public static final int DIMENSIONS = 10000;
private static final int ARRAY_SIZE = (DIMENSIONS + 63) / 64;  // = 157
```

L'elemento `long[156]` (l'ultimo) contiene solo $10000 - 156 \times 64 = 10000 - 9984 = 16$ bit significativi. I restanti 48 bit sono **don't care** (rimangono a 0).

### 5.3 Gestione bit in eccesso

I bit eccedenti nell'ultimo long (posizioni 16–63) sono sempre 0 e non influenzano la distanza di Hamming perché lo XOR con 0 restituisce il bit dell'altro operando, che è anch'esso 0 (entrambi i vettori hanno i bit eccedenti a 0). Pertanto la `Long.bitCount()` su questi bit restituisce sempre 0.

### 5.4 Wrapper per l'apprendimento

```java
// MainActivity.java:417-421
private VsaVector binarize(double[] continuous) {
    long[] blocks = new long[157];
    binarizeToBlocks(continuous, blocks);
    return new VsaVector(blocks);
}
```

### 5.5 Buffer pre‑allocati (anti‑GC)

Per l'inferenza frame‑by‑frame, i buffer sono pre‑allocati come membri di `MainActivity`:

```java
// MainActivity.java:59-61
private double[] continuousBuffer;       // double[10000] — riusato ogni frame
private long[] bscBlocksBuffer;          // long[157]    — riusato ogni frame
private VsaVector reusableQueryVector;   // VsaVector    — setBlocks() zero‑allocation
```

Il metodo `VsaVector.setBlocks()` usa `System.arraycopy` per sovrascrivere i blocchi senza allocare nuovi oggetti:

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

### 6.1 Calcolo XOR e `Long.bitCount()` (POPCNT)

La distanza di Hamming tra due ipervettori BSC è il numero di bit discordanti. Si calcola con XOR bit‑a‑bit sui 157 blocchi `long`, seguito da `Long.bitCount()` che su architetture ARMv8 viene mappato sull'istruzione hardware `CNT` + `ADDV` (POPCNT).

```java
// VsaVector.java:41-52
public int calculateHammingDistance(VsaVector other) {
    int distance = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        long differences = this.blocks[i] ^ other.blocks[i];  // XOR
        distance += Long.bitCount(differences);               // POPCNT hardware
    }
    return distance;
}
```

### 6.2 Conversione in similarità coseno

La relazione tra distanza di Hamming $h$ e similarità coseno $\cos\theta$ per vettori bipolari ±1 di dimensione $D = 10000$ è:

$$
\cos\theta = 1 - \frac{2h}{D}
$$

```java
// VsaVector.java:58-61
public double getCosineSimilarity(VsaVector other) {
    int h = calculateHammingDistance(other);
    return 1.0 - 2.0 * ((double) h / DIMENSIONS);
}
```

- Due vettori identici: $h = 0$, $\cos\theta = 1.0$
- Due vettori ortogonali (attesi per vettori casuali): $h \approx 5000$, $\cos\theta \approx 0.0$
- Due vettori opposti: $h = 10000$, $\cos\theta = -1.0$

### 6.3 Caricamento archetipi

Gli archetipi sono mantenuti in due mappe:

| Mappa | Chiave | Valore | Uso |
|-------|--------|--------|-----|
| `archetypesMAP` | `String` (nome classe) | `double[10000]` | Persistenza, apprendimento |
| `archetypesBSC` | `String` (nome classe) | `VsaVector` | Matching veloce (BSC pre‑calcolato) |

**File di persistenza**: `archetype_map.bin` nella directory `getFilesDir()` (memoria interna privata dell'app).

#### Formato `archetype_map.bin`

| Offset | Tipo | Contenuto |
|--------|------|-----------|
| 0 | `int` (4 byte) | `numClasses` |
| 4 | `int` (4 byte) | `VSA_DIM` (10000) |
| 8 | variabile | Per ogni classe: |
| | `UTF` (2 + N byte) | Nome classe |
| | `double[10000]` | Vettore MAP continuo |

#### Salvataggio

```java
// MainActivity.java:558-572
private synchronized void saveAllArchetypesMAP() {
    try (FileOutputStream fos = openFileOutput(USER_ARCHETYPES_FILE, Context.MODE_PRIVATE);
         DataOutputStream dos = new DataOutputStream(fos)) {
        dos.writeInt(archetypesMAP.size());
        dos.writeInt(VSA_DIM);
        for (Map.Entry<String, double[]> entry : archetypesMAP.entrySet()) {
            dos.writeUTF(entry.getKey());
            for (double v : entry.getValue()) dos.writeDouble(v);
        }
    }
}
```

#### Caricamento e late binarization

```java
// MainActivity.java:574-599
private void loadAllArchetypesMAP() {
    java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
    if (!file.exists()) return;
    try (FileInputStream fis = new FileInputStream(file);
         DataInputStream dis = new DataInputStream(fis)) {
        int numClasses = dis.readInt();
        int dims = dis.readInt();
        if (dims != VSA_DIM) { file.delete(); return; }  // incompatibile
        for (int i = 0; i < numClasses; i++) {
            String name = dis.readUTF();
            double[] vec = new double[VSA_DIM];
            for (int d = 0; d < VSA_DIM; d++) vec[d] = dis.readDouble();
            archetypesMAP.put(name, vec);
        }
    }
}
```

Dopo il caricamento della MAP, ogni vettore viene binarizzato in BSC per il matching veloce:

```java
// MainActivity.java:134-137
loadAllArchetypesMAP();
if (!archetypesMAP.isEmpty()) {
    for (Map.Entry<String, double[]> e : archetypesMAP.entrySet()) {
        archetypesBSC.put(e.getKey(), binarize(e.getValue()));
    }
}
```

**Nota**: all'avvio, il file `archetype_map.bin` viene sempre **cancellato** per garantire un test pulito:

```java
// MainActivity.java:127-131
java.io.File oldFile = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
if (oldFile.exists()) {
    oldFile.delete();
    Log.d(TAG, "Archetipi precedenti cancellati — test pulito");
}
```

### 6.4 Soglia di rifiuto

La soglia di similarità minima è **0.35** (cosine similarity), hard‑coded:

```java
// MainActivity.java:236
double bestSim = 0.35;
```

Non viene usata una soglia statistica (3σ). Se nessun archetipo supera 0.35, il risultato è `null` e la UI mostra "Ricerca in corso...".

### 6.5 Algoritmo di matching completo

```java
// MainActivity.java:268-276
StringBuilder diag = new StringBuilder("VSA sim: ");
for (Map.Entry<String, VsaVector> entry : archetypesBSC.entrySet()) {
    double sim = reusableQueryVector.getCosineSimilarity(entry.getValue());
    diag.append(entry.getKey()).append("=").append(String.format("%.3f", sim)).append(" ");
    if (sim > bestSim) {
        bestSim = sim;
        bestMatch = entry.getKey();
    }
}
```

### 6.6 File asset `vsa_archetypes.bin`

Il file `vsa_archetypes.bin` esiste negli assets ma **non viene caricato dalla pipeline attiva**. È un residuo di una versione precedente o è destinato a contenere archetipi pre‑addestrati da importare. La pipeline corrente usa solo `archetype_map.bin` (in internal storage).

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
                   │   • Modello: mobilenet_ssd.tflite               │
                   │   • Input: [256, 256, 3] float32                │
                   │   • Output: RectF norm [0.0–1.0]                │
                   │   (anchor decoding + NMS interna a MediaPipe)   │
                   └────────────────────┬────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │ box == null?              │
                          └─────────────┬─────────────┘
                                 Sì     │     No
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
           UI: "Ricerca..."              ┌──────────────────────────────────┐
                                        │      EXPAND BOX (+20% padding)    │
                                        │      LETTERBOX CROP               │
                                        │      • Ritaglia RectF da frame    │
                                        │      • Centro su quadrato nero    │
                                        └────────────────┬─────────────────┘
                                                         │
                                                         ▼
                   ┌─────────────────────────────────────────────────┐
                   │         STAGE 2 — CNN FEATURE EXTRACTION         │
                   │   FeatureEncoder.extract(square)                │
                   │   • resize a 224×224 (bilinear)                 │
                   │   • Normalizzazione [-1.0, 1.0] per canale RGB  │
                   │   • TFLite Interpreter (4 thread)               │
                   │   • Modello: mobilenet_vsa.tflite               │
                   │   • Output: float[1280]                         │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │            DEBINDING (black padding)             │
                   │   applyDebinding(features, box)                 │
                   │   • Sottrae firma del padding nero              │
                   │   • Proporzionale al paddingRatio               │
                   │   • Clamp a max(0, value)                       │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │         STAGE 3 — VSA RANDOM PROJECTION         │
                   │   RandomProjectionBridge.projectFeatureVector() │
                   │   • Matrice boolean[10000][1280], seed=42       │
                   │   • Zero‑skipping sulle feature ReLU            │
                   │   • Prodotto scalare ±1                         │
                   │   • Output: double[10000] continuo              │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │         BINARIZZAZIONE BSC (bit‑packing)        │
                   │   binarizeToBlocks(continuousBuffer, blocks)    │
                   │   • Soglia 0 (funzione segno)                   │
                   │   • Packing in long[157]                        │
                   │   • VsaVector.setBlocks() (zero‑alloc)          │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │       MATCHING CONTRO ARCHETIPI BSC             │
                   │   Per ogni archetipo in archetypesBSC:          │
                   │   • XOR bit‑a‑bit (long[157])                   │
                   │   • Long.bitCount() → Hamming distance          │
                   │   • cos = 1 − 2·h/10000                         │
                   │   • bestMatch se sim > 0.35                     │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │      STABILIZZAZIONE TEMPORALE                  │
                   │   DetectionStabilizer.update(box, className)    │
                   │   • EMA smoothing spaziale (α=0.3)              │
                   │   • Majority voting su 5 frame                  │
                   │   • Coasting 6 frame dopo scomparsa             │
                   └────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────────────┐
                   │                 UI RENDERING                     │
                   │   tvVerdict, tvConfidence, TrackingOverlayView  │
                   │   • Rotazione box per display (0°/90°/180°/270°)│
                   │   • Aspect‑ratio fit nella PreviewView          │
                   └─────────────────────────────────────────────────┘
```

---

## 8. Addestramento / creazione archetipi

### 8.1 Flusso di apprendimento (on‑device)

L'apprendimento **one‑shot** è eseguito interamente sul dispositivo. Il flusso ricalca la pipeline di inferenza fino alla proiezione VSA, poi salva l'archetipo invece di confrontarlo.

#### Passo 1: Input del nome

L'utente preme il FAB "+", inserisce il nome e l'app scatta una foto:

```java
// MainActivity.java:423-444
private void showNameInputDialog() {
    // Mostra AlertDialog con EditText
    // Al click "Cattura e Impara" → pendingLearningName = name; capturePhotoForLearning();
}
```

#### Passo 2: Cattura foto

```java
// MainActivity.java:446-480
private void capturePhotoForLearning() {
    File photoFile = new File(getCacheDir(), "learning_photo.jpg");
    ImageCapture.OutputFileOptions outputOptions =
            new ImageCapture.OutputFileOptions.Builder(photoFile).build();
    imageCapture.takePicture(outputOptions, inferenceExecutor, callback);
    // callback.onImageSaved → BitmapFactory.decodeFile → learnNewCustomObject()
}
```

#### Passo 3: Pipeline completa + salvataggio

```java
// MainActivity.java:482-556 (learnNewCustomObject, estratti chiave)
inferenceExecutor.execute(() -> {
    // 1. SSD → boxNorm
    RectF boxNorm = ssdExtractor.detectObject(sampleFrame);
    if (boxNorm == null) return; // errore: nessun oggetto

    // 2. Expand + letterbox crop
    RectF cropBoxNorm = expandBox(boxNorm, 0.20f);
    RectF boxPx = /* ... */;
    Bitmap square = letterboxCrop(sampleFrame, boxPx);

    // 3. Encoder CNN
    float[] features = encoder.extract(square);
    square.recycle();

    // 4. Debinding (sottrazione firma padding nero)
    applyDebinding(features, boxPx);

    // 5. Proiezione VSA + L2 normalization
    double[] mapVec = new double[VSA_DIM];
    vsaBridge.projectFeatureVector(features, mapVec);
    vsaBridge.normalizeL2(mapVec);               // ← L2 normalization SOLO in learning

    // 6. Salvataggio MAP + BSC
    archetypesMAP.put(className, mapVec);         // double[10000] normalizzato L2
    archetypesBSC.put(className, binarize(mapVec)); // VsaVector binarizzato
    saveAllArchetypesMAP();                       // persistenza su file
});
```

**Differenza chiave learning vs inference**:
- **Learning**: proiezione → **L2 normalization** → salva MAP continuo + binarizza per BSC
- **Inference**: proiezione → binarizza direttamente (senza L2 normalization)

### 8.2 Debinding del padding nero

Il letterbox introduce pixel neri che inquinano il feature vector. Per mitigarlo, si calcola una **firma del padding nero** all'avvio e la si sottrae proporzionalmente al `paddingRatio`:

```java
// MainActivity.java:113-116 (all'avvio)
Bitmap black = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
new Canvas(black).drawColor(Color.BLACK);
blackPaddingSignature = encoder.extract(black);   // float[1280] della firma nera
black.recycle();
```

Durante l'elaborazione, se il `paddingRatio` supera il 10%, si applica la sottrazione:

```java
// MainActivity.java:383-406
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

Il risultato viene clippato a `max(0, value)` per rispettare la semantica ReLU.

### 8.3 Bundling (associazione VSA)

L'operazione di bundling classica (somma e binarizzazione) **non è usata** per l'apprendimento one‑shot. Ogni oggetto viene appreso con un singolo scatto (one‑shot), quindi l'archetipo è direttamente il vettore proiettato di quel singolo sample.

Il `bind()` (XOR) esiste nella classe `VsaVector` ma non è utilizzato nella pipeline attuale:

```java
// VsaVector.java:28-35 (binding XOR — disponibile ma non chiamato)
public VsaVector bind(VsaVector other) {
    long[] result = new long[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        result[i] = this.blocks[i] ^ other.blocks[i];
    }
    return new VsaVector(result);
}
```

### 8.4 Esportazione e persistenza

L'unico formato di esportazione è `archetype_map.bin` (vedi §6.3 per il formato). Non esiste esportazione CSV o in altri formati dall'app.

Il file `vsa_archetypes.bin` negli assets è un file pre‑esistente che non viene mai caricato dalla pipeline attuale.

### 8.5 Cancellazione archetipi

Il long‑click sul FAB "+" cancella tutti gli archetipi:

```java
// MainActivity.java:87-104
fabLearn.setOnLongClickListener(v -> {
    new AlertDialog.Builder(this)
        .setTitle("Pulisci Archetipi")
        .setMessage("Cancellare tutti gli oggetti appresi?")
        .setPositiveButton("Si, cancella tutto", (d, w) -> {
            inferenceExecutor.execute(() -> {
                clearArchetypes();
                // UI: "Tabula Rasa. Premi + per imparare."
            });
        })
        // ...
    return true;
});
```

`clearArchetypes()` svuota entrambe le mappe e cancella il file:

```java
// MainActivity.java:601-607
private synchronized void clearArchetypes() {
    archetypesMAP.clear();
    archetypesBSC.clear();
    java.io.File file = new java.io.File(getFilesDir(), USER_ARCHETYPES_FILE);
    if (file.exists()) file.delete();
}
```

---

## 9. Validazione

### 9.1 Accuratezza sul dispositivo

**Non esiste un modulo di validazione formale**. Non ci sono test che misurano l'accuratezza del classificatore VSA. I file di test esistenti sono placeholder vuoti:

```
app/src/test/java/com/antonio/vsatest/ExampleUnitTest.java        → placeholder
app/src/androidTest/java/com/antonio/vsatest/ExampleInstrumentedTest.java → placeholder
```

### 9.2 Logging diagnostico

La validazione empirica avviene tramite log di debug (TAG `VSA_TwoStage`):

```java
// Ad ogni frame, logga tutte le similarità:
Log.d(TAG, "VSA sim: " + diag.toString());
// Esempio output: "VSA sim: Vite_M5=0.823 Dado_M8=0.214 Rondella=0.105"

// Tempi di esecuzione:
Log.d(TAG, "Box=[x,y,w,h] match=className sim=0.823");
// UI mostra: "SSD+45ms VSA:12ms TOT:67ms"
```

### 9.3 UI di confidence

La UI mostra in tempo reale il verdetto e la similarità coseno:

```java
// MainActivity.java:296
fVerdict = result.className.replace("_", " ") + " (" + String.format("%.2f", bestSim) + ")";
```

Il `TrackingOverlayView` disegna il bounding box con etichetta e confidence.

---

## 10. Esportazione della matrice di proiezione

**Non implementata nell'app Android.** La matrice di proiezione è generata al volo all'avvio con seed deterministico (`42L`) e non viene mai esportata in CSV o altri formati. A differenza della versione Java desktop, non esiste un metodo per salvare la matrice su file.

La riproducibilità è garantita dal seed fisso: ogni esecuzione genera la stessa identica matrice `boolean[10000][1280]`.

---

## 11. Riepilogo dei file chiave e delle classi

| File / Componente | Ruolo | Percorso |
|-------------------|-------|----------|
| **MainActivity.java** | Orchestratore: camera, pipeline two‑stage, one‑shot learning, persistenza, UI, debinding, binarizzazione | `app/src/main/java/com/antonio/visionsector/MainActivity.java` |
| **SsdExtractor.java** | Stadio 1: Object Detection SSD via MediaPipe Tasks | `app/src/main/java/com/antonio/visionsector/SsdExtractor.java` |
| **FeatureEncoder.java** | Stadio 2: Encoder CNN MobileNetV2 via TFLite Interpreter | `app/src/main/java/com/antonio/visionsector/FeatureEncoder.java` |
| **RandomProjectionBridge.java** | Proiezione casuale 1280→10000 (±1), zero‑skipping, L2 normalization, binarizeTo() | `app/src/main/java/com/antonio/visionsector/RandomProjectionBridge.java` |
| **VsaVector.java** | Ipervettore BSC: XOR binding, Hamming distance (POPCNT), cosine similarity, setBlocks zero‑alloc | `app/src/main/java/com/antonio/visionsector/VsaVector.java` |
| **DetectionStabilizer.java** | Stabilizzazione temporale: EMA smoothing spaziale + majority voting semantico | `app/src/main/java/com/antonio/visionsector/DetectionStabilizer.java` |
| **TrackingOverlayView.java** | View personalizzata: bounding box con aspect‑ratio fit, dim mask, etichette | `app/src/main/java/com/antonio/visionsector/TrackingOverlayView.java` |
| **AndroidTFLiteExtractor.java** | Legacy: estrazione spaziale 7×7 (non più usato) | `app/src/main/java/com/antonio/visionsector/AndroidTFLiteExtractor.java` |
| **DatasetStatTracker.java** | Standardizzazione Z‑Score (classe pronta, non attualmente integrata) | `app/src/main/java/com/antonio/visionsector/DatasetStatTracker.java` |
| **AntiGlareFilter.java** | Filtro anti‑riflesso LUT gamma 1.4 (non attualmente integrato) | `app/src/main/java/com/antonio/visionsector/AntiGlareFilter.java` |
| **mobilenet_ssd.tflite** | Modello SSD Object Detection (MediaPipe) | `app/src/main/assets/mobilenet_ssd.tflite` |
| **mobilenet_ssd_Generic.tflite** | Variante SSD non utilizzata | `app/src/main/assets/mobilenet_ssd_Generic.tflite` |
| **mobilenet_vsa.tflite** | Modello MobileNetV2 encoder feature (TFLite) | `app/src/main/assets/mobilenet_vsa.tflite` |
| **mobilenet_v2_spatial.tflite** | Modello spatial grid legacy (non usato) | `app/src/main/assets/mobilenet_v2_spatial.tflite` |
| **vsa_archetypes.bin** | Archetipi pre‑addestrati (asset, non caricati dalla pipeline attuale) | `app/src/main/assets/vsa_archetypes.bin` |
| **vsa_stats.bin** | Statistiche Z‑Score pre‑calcolate (asset, non usate) | `app/src/main/assets/vsa_stats.bin` |
| **test_foglia.jpg** | Immagine di test "foglia" (foglia) | `app/src/main/assets/test_foglia.jpg` |
| **archetype_map.bin** | Persistenza runtime archetipi (internal storage) | `getFilesDir()/archetype_map.bin` |
| **learning_photo.jpg** | Foto temporanea per apprendimento | `getCacheDir()/learning_photo.jpg` |
| **activity_main.xml** | Layout UI: PreviewView, FAB, CardView risultati, TrackingOverlayView | `app/src/main/res/layout/activity_main.xml` |
| **AndroidManifest.xml** | Permessi fotocamera, dichiarazione Activity | `app/src/main/AndroidManifest.xml` |
| **build.gradle.kts** (app) | Dipendenze: TFLite 2.16.1, CameraX 1.3.4, MediaPipe Tasks Vision 0.10.14 | `app/build.gradle.kts` |

---

## 12. Differenze rispetto alla versione Java desktop

| Aspetto | Java Desktop | Android |
|---------|-------------|---------|
| **Runtime CNN** | TensorFlow SavedModel (Java API) | TensorFlow Lite (`Interpreter`, `MappedByteBuffer`) |
| **Formato modello** | SavedModel directory | `.tflite` (FlatBuffer, ≈2.5 MB) |
| **Object Detection** | Non presente / separata | Stadio 1 integrato: SSD MobileNetV2 via MediaPipe Tasks |
| **Origine immagine** | File su disco | CameraX in tempo reale (`ImageProxy → Bitmap`) |
| **Pre‑processing** | `BufferedImage`, `resize()` | `Bitmap.createScaledBitmap`, `getPixels()`, `ByteBuffer.allocateDirect` |
| **Memory budget** | Heap abbondante (GB) | Limitato (dispositivo mobile). Buffer pre‑allocati, `arraycopy` zero‑alloc, bitmap recycling aggressivo |
| **Matrice proiezione** | `boolean[10000][1280]` su heap | Identica, stessa dimensione (~12.8 MB) |
| **Zero‑skipping** | Probabilmente assente | Implementato: `projectFeatureVector()` itera solo su feature ≠ 0 |
| **Normalizzazione L2** | Applicata in pipeline fissa | Solo durante apprendimento; l'inferenza binarizza senza L2 |
| **Standardizzazione Z‑Score** | Probabilmente integrata | Classe `DatasetStatTracker` pronta ma **non cablata** nella pipeline |
| **Binarizzazione** | Soglia 0, packing `long[]` | Identico algoritmo, buffer pre‑allocato riutilizzato |
| **POPCNT hardware** | `Long.bitCount()` → istruzione `POPCNT` x86 | `Long.bitCount()` → istruzione `CNT`+`ADDV` su ARMv8; su ARMv7 è software (più lento) |
| **Threading** | Main thread o thread pool | `SingleThreadExecutor` dedicato per inferenza, UI thread separato |
| **Persistenza archetipi** | File binario su disco | `DataOutputStream` su `FileOutputStream` in internal storage (`getFilesDir()`) |
| **Apprendimento** | Batch multi‑sample (cache feature, bundling, late binarization) | **One‑shot**: singolo scatto, nessun bundling multi‑sample |
| **Bundling** | Somma di più vettori → binarizzazione | Non implementato (one‑shot = 1 sample per classe) |
| **Binding (XOR)** | Usato per associare ruoli (colore, posizione) | Disponibile in `VsaVector.bind()` ma **non chiamato** nella pipeline |
| **Debinding** | Non presente | Implementato: sottrae firma del padding nero proporzionalmente al `paddingRatio` |
| **Stabilizzazione temporale** | Non necessaria (immagini statiche) | `DetectionStabilizer`: EMA box + majority voting classe + coasting |
| **Anti‑glare** | Non presente | `AntiGlareFilter` (LUT gamma 1.4) pronto ma non cablato |
| **Validazione** | Misura accuratezza su dataset | Nessun modulo di test; solo logging diagnostico |
| **Esportazione matrice** | Probabilmente supportata (CSV) | Non implementata |
| **Rotazione display** | Non applicabile | Gestione rotazione 0°/90°/180°/270° del sensore fotocamera |
| **Input SSD** | Non applicabile | 256×256 float32 gestito internamente da MediaPipe |
| **Input encoder** | 224×224 float [-1,1] | 224×224 float [-1,1] — identico |
| **Numero thread TFLite** | Variabile | Hard‑coded a 4 (`options.setNumThreads(4)`) |
| **Caricamento modello** | `SavedModelBundle.load()` | `MappedByteBuffer` da `AssetFileDescriptor` (zero‑copy dal filesystem) |
| **Garbage Collection** | Meno critica | Ottimizzata: buffer riutilizzati, `setBlocks()` con `System.arraycopy`, zero‑allocation ove possibile |
| **Soglia matching** | Probabilmente basata su 3σ statistico | Hard‑coded `0.35` cosine similarity |
| **Spatial grid (7×7)** | Probabilmente usata nella versione legacy | Supportata da `AndroidTFLiteExtractor` e `projectSpatialCellToContinuous()` ma **non più attiva** |

---

*Documento generato il 2026‑06‑20 dall'analisi completa del codice sorgente Android.*
