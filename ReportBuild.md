# ReportBuild.md — VsaVisionSector Two-Stage Cascade

## Stato: BUILD SUCCESSFUL

---

## 1. Architettura Two-Stage Cascade

### Diagramma del flusso

```
Camera Frame (ImageProxy → Bitmap, es. 640×480)
│
├─► [STAGE 1 — SSD Localizzatore]                  SsdExtractor.java
│   • Crea copia 300×300 del frame                  (uint8, 0-255)
│   • Inferenza con mobilenet_ssd.tflite
│   • Seleziona detection con max confidence > 0.5
│   • Restituisce RectF normalizzato [0.0–1.0]
│   • .recycle() del 300×300
│   • Se nessuna detection → UI "Ricerca in corso..."
│
├─► [RIPROIEZIONE]  coord × (origW, origH) → RectF pixel
│
├─► [STAGE 2 — Crop + Letterbox + Encoder]
│   • Ritaglia il RectF dal frame originale ad alta risoluzione
│   • Letterbox: incolla il crop al centro di quadrato nero
│     (lato = max(crop.width, crop.height))
│   • Ridimensiona a 224×224
│   • Inferenza con mobilenet_vsa.tflite → float[1280]
│
├─► [STAGE 3 — VSA Projection]                     RandomProjectionBridge.java
│   • Proiezione casuale 1280 → 10000 dimensioni
│   • Binarizzazione BSC (157 long blocks)
│   • Matching con tutti gli archetipi (cosine similarity)
│
└─► UI Update (verdict + bounding box sull'overlay)
```

### Stessa pipeline per Detection e Learning

- **Detection** (`processCameraFrame`): SSD → letterbox → encoder → VSA match → UI
- **Learning** (`learnNewCustomObject`): SSD → letterbox → encoder → VSA projection → salva MAP + BSC

---

## 2. File del Progetto

### 2.1 File Modificati/Nuovi

| File | Righe | Ruolo |
|------|-------|-------|
| `SsdExtractor.java` | 113 | **NUOVO** — Localizzatore SSD TFLite. Input uint8 300×300, output `RectF` [0,1]. Ignora classi, solo coordinate. |
| `FeatureEncoder.java` | 69 | **NUOVO** — Encoder MobileNetV2. Input 224×224 float [-1,1], output `float[1280]`. |
| `MainActivity.java` | 450 | **RISCRITTO** — Pipeline two-stage completa, one-shot learning, persistenza MAP. |
| `TrackingOverlayView.java` | 112 | **MODIFICATO** — `DetectedItem` usa coordinate pixel reali; `onDraw` scala su view con aspect ratio. |
| `RandomProjectionBridge.java` | 156 | **INVARIATO** (cambiato solo valore passato al costruttore: `1280` invece di `256`). |
| `VsaVector.java` | 77 | **INVARIATO** — Vettore BSC 10000-dim con XOR binding e cosine similarity via Hamming. |
| `AndroidTFLiteExtractor.java` | 132 | **INVARIATO** — Mantenuto per riferimento, non più usato nella pipeline attiva. |
| `DatasetStatTracker.java` | 57 | **INVARIATO** — Standardizzatore statistico, non attualmente usato. |

### 2.2 Modelli TFLite in assets

| Modello | Dimensione | Input | Output | Usato da |
|---------|-----------|-------|--------|----------|
| `mobilenet_ssd.tflite` | 4 MB | 300×300, uint8 [0-255] | `TFLite_Detection_PostProcess` (4 tensor: boxes, classes, scores, num) | `SsdExtractor` |
| `mobilenet_vsa.tflite` | 2.5 MB | 224×224, float [-1,1] | `output_0` [1,1280] | `FeatureEncoder` |
| `mobilenet_v2_spatial.tflite` | — | — | — | Non usato |
| `mobilenet_vsa_7x7x256.tflite` | — | — | — | Non più presente |

---

## 3. Problema di Compilazione Riscontrato e Risoluzione

### Errore originale

```
SsdExtractor.java:67: error: incompatible types: Object[] cannot be converted to Map<Integer,Object>
    tflite.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
                                                                  ^
```

### Causa

L'API TensorFlow Lite 2.16.1 ha cambiato la firma di `runForMultipleInputsOutputs`. La versione usata richiede un `Map<Integer, Object>` per gli output invece di un `Object[]`.

**API corretta:**
```java
void runForMultipleInputsOutputs(Object[] inputs, Map<Integer, Object> outputs)
```

### Fix applicato

**Prima (errato):**
```java
Object[] outputs = {detectionBoxes, detectionClasses, detectionScores, numDetections};
tflite.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
```

**Dopo (corretto):**
```java
Map<Integer, Object> outputs = new HashMap<>();
outputs.put(0, detectionBoxes);
outputs.put(1, detectionClasses);
outputs.put(2, detectionScores);
outputs.put(3, numDetections);
tflite.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
```

Le chiavi (0, 1, 2, 3) corrispondono all'ordine dei tensori di output del modello SSD:
- `0` → Boxes `[1, maxDet, 4]`
- `1` → Classes `[1, maxDet]`
- `2` → Scores `[1, maxDet]`
- `3` → NumDetections `[1]`

### Import aggiunto

```java
import java.util.HashMap;
import java.util.Map;
```

---

## 4. Dettaglio delle Classi

### 4.1 SsdExtractor (nuovo)

```
Costruttore:
  SsdExtractor(Context, "mobilenet_ssd.tflite")
    → Carica modello TFLite, 4 thread
    → Alloca input buffer uint8 (300×300×3 × 1 byte)
    → Legge output shape dal modello (maxDet dal tensor 0)

Metodo principale:
  RectF detectObject(Bitmap bitmap)
    → Bitmap.createScaledBitmap(300×300) + converti in ByteBuffer uint8
    → tflite.runForMultipleInputsOutputs con Map<Integer,Object>
    → Scansiona detectionScores[0][0..num-1] per max > 0.5
    → Restituisce RectF(xmin, ymin, xmax, ymax) normalizzato [0,1]
    → null se nessuna detection supera la soglia

Gestione memoria:
    → resized.recycle() dopo l'inferenza
    → inputBuffer riutilizzato (rewind)
```

### 4.2 FeatureEncoder (nuovo)

```
Costruttore:
  FeatureEncoder(Context, "mobilenet_vsa.tflite")
    → Carica modello TFLite, 4 thread

Metodo principale:
  float[] extract(Bitmap crop)
    → Bitmap.createScaledBitmap(224×224)
    → getPixels + normalizzazione [-1,1] per canale RGB
    → tflite.run(inputBuffer, float[1][1280])
    → Ritorna output[0]
    → resized.recycle()

Gestione memoria:
    → ByteBuffer allocato a ogni chiamata (1280×4 byte, trascurabile)
    → resized bitmap riciclata
```

### 4.3 MainActivity (riscritto)

```
Costanti:
  FEATURE_DIM = 1280  (output encoder)
  VSA_DIM = 10000     (dimensione ipervettore)
  USER_ARCHETYPES_FILE = "archetype_map.bin"  (persistenza MAP)

Membri motore:
  ssdExtractor   : SsdExtractor
  encoder        : FeatureEncoder
  vsaBridge      : RandomProjectionBridge(1280, 42L)
  archetypesMAP  : Map<String, double[]>
  archetypesBSC  : Map<String, VsaVector>

Membri buffer (anti-GC):
  continuousBuffer : double[10000]
  bscBlocksBuffer  : long[157]
  reusableQueryVector : VsaVector

Threading:
  inferenceExecutor : Executors.newSingleThreadExecutor()
    → CameraX ImageAnalysis analyzer
    → Caricamento modelli iniziale
    → processCameraFrame e learnNewCustomObject

Metodi principali:
  processCameraFrame(Bitmap frame)
    → SSD → riproiezione → letterboxCrop → encoder → VSA → matching → UI

  learnNewCustomObject(String, Bitmap)
    → SSD → riproiezione → letterboxCrop → encoder → VSA → salva MAP+BSC

  letterboxCrop(Bitmap, RectF)
    → Ritaglia il box dal frame originale
    → Crea quadrato nero (lato = max(crop.w, crop.h))
    → Incolla crop al centro
    → recycle del crop intermedio
    → Restituisce quadrato

  binarizeToBlocks / binarize
    → Converte double[10000] in long[157] BSC

Persistenza:
  saveAllArchetypesMAP → DataOutputStream: numClasses, vsaDim, key+values
  loadAllArchetypesMAP → DataInputStream: verificato compatibilità dim, late binarize
```

### 4.4 TrackingOverlayView (modificato)

```
Modifiche rispetto alla versione precedente:

DetectedItem: coordinate pixel reali (pixelLeft/Right/Top/Bottom)
  invece di celle griglia (minCol/minRow/maxCol/maxRow)

updateDetections(List<DetectedItem>, int frameW, int frameH)
  invece di (List<DetectedItem>, int gridSize)

onDraw:
  → Rimossa griglia 7×7
  → Aggiunto fit aspect-ratio del frame nella view
  → Box disegnati con scala: pixelCoord * (viewSize / frameSize)
  → Mantenuto dim mask e viewfinder dorato per estetica
```

### 4.5 RandomProjectionBridge (invariato)

```
Unica modifica: inizializzazione in MainActivity
  Prima: new RandomProjectionBridge(256, 42L)     ← 256 feature della griglia 7×7
  Dopo:  new RandomProjectionBridge(1280, 42L)    ← 1280 feature dell'encoder

La classe stessa non è stata toccata. La matrice di proiezione ora è
boolean[10000][1280] (~12.8 MB) invece di boolean[10000][256] (~2.5 MB).

Metodo usato nella nuova pipeline:
  projectFeatureVector(float[1280], double[10000])  ← zero-skipping, cache-local
```

### 4.6 VsaVector (invariato)

```
Nessuna modifica. Usa:
  - setBlocks(long[157])     per zero-allocation update del query vector
  - getCosineSimilarity()    per matching contro archetipi
  - calculateHammingDistance() con hardware POPCNT
```

---

## 5. Gestione Memoria (Bitmap Recycling)

| Oggetto | Dove | Quando riciclato |
|---------|------|-----------------|
| `ImageProxy` | `startCamera` analyzer | `.close()` nel `finally` |
| Frame originale (`Bitmap`) | `processCameraFrame` | `.recycle()` a fine metodo |
| 300×300 SSD input | `SsdExtractor.detectObject` | `.recycle()` dopo inferenza |
| Crop dal frame | `letterboxCrop` | `.recycle()` dopo `drawBitmap` sul quadrato |
| Quadrato letterbox | `processCameraFrame` | `.recycle()` dopo `encoder.extract()` |
| 224×224 encoder input | `FeatureEncoder.extract` | `.recycle()` dopo `getPixels` |

---

## 6. Build Info

```
Gradle:       9.4.1
AGP:          9.2.1
compileSdk:   34
minSdk:       24
targetSdk:    34
Java:         11
TFLite:       2.16.1
CameraX:      1.3.4
Material:     1.12.0
```

---

## 7. Riepilogo Modifiche

| # | File | Operazione | Righe |
|---|------|-----------|-------|
| 1 | `SsdExtractor.java` | Creato | 113 |
| 2 | `FeatureEncoder.java` | Creato | 69 |
| 3 | `MainActivity.java` | Riscritto (da 783 → 450 righe) | 450 |
| 4 | `TrackingOverlayView.java` | Modificato (da 122 → 112 righe) | 112 |
| 5 | `RandomProjectionBridge.java` | Invariato (cambio solo parametro costruttore) | 156 |
| 6 | `VsaVector.java` | Invariato | 77 |
| 7 | `AndroidTFLiteExtractor.java` | Invariato (non più usato, tenuto per ref) | 132 |
| 8 | `DatasetStatTracker.java` | Invariato | 57 |
