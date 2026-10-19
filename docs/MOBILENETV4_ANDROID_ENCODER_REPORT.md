# MobileNetV4 Android Encoder — Report

Branch: `feat/latent-v4`

Stato: wrapper TFLite implementato e compilato. Nessuna inferenza runtime verificata su dispositivo.

## 1. File implementato

`app/src/main/java/com/antonio/visionsector/encoder/MobileNetV4Encoder.java`

Classe: `public final class MobileNetV4Encoder implements AutoCloseable`.

## 2. Modello usato

`app/src/main/assets/mobilenetv4_conv_small_encoder.tflite`

Contratto atteso e verificato dal wrapper (`validateContract`):

- input `[1,224,224,3]`
- input `FLOAT32`
- output `[1,960]`
- output `FLOAT32`
- un solo input
- un solo output

Costanti usate dal wrapper:

- `INPUT_SIZE = 224`
- `INPUT_CHANNELS = 3`
- `INPUT_ELEMENT_COUNT = 224 * 224 * 3` (150528)
- `EMBEDDING_DIM = 960`

## 3. API del wrapper

```java
public float[] encode(float[] nhwcInput)
```

Requisiti e comportamento:

- `nhwcInput` non nullo;
- `nhwcInput.length == 224 * 224 * 3` (150528);
- input in formato NHWC, float già pronti (nessun preprocessing applicato dal wrapper);
- output: `float[960]`.

Vincoli violati producono `IllegalArgumentException` (input nullo o lunghezza errata).

## 4. Gestione memoria

Il wrapper, per ogni chiamata a `encode`:

- crea internamente un `ByteBuffer` diretto di `INPUT_ELEMENT_COUNT * Float.BYTES` byte;
- imposta l'ordine con `ByteOrder.nativeOrder()`;
- copia i float dell'input nel buffer senza modificarli (`putFloat` in ciclo);
- esegue `rewind()` prima dell'inferenza;
- crea un output `float[1][960]`;
- restituisce `output[0]`.

Ottimizzazione futura possibile, **non implementata**: riuso dei buffer (input diretto e output) tra inferenze per evitare allocazioni per chiamata.

## 5. Caricamento modello

`loadModel(Context)` usa:

- `AssetFileDescriptor` da `context.getAssets().openFd(MODEL_ASSET)`;
- `FileInputStream` sul file descriptor;
- `FileChannel` ottenuto dallo stream;
- `MappedByteBuffer` tramite `channel.map(READ_ONLY, descriptor.getStartOffset(), descriptor.getDeclaredLength())`;
- `Interpreter` costruito su tale `MappedByteBuffer`.

Le risorse di caricamento sono gestite in try-with-resources; in caso di errore durante `allocateTensors()`/validazione, l'`Interpreter` viene chiuso e l'eccezione rilanciata.

## 6. Validazione runtime

Alla costruzione, dopo `allocateTensors()`, `validateContract(Interpreter)` verifica:

- tensor count: esattamente 1 input e 1 output;
- shape input: `[1,224,224,3]`;
- shape output: `[1,960]`;
- dtype input: `FLOAT32`;
- dtype output: `FLOAT32`.

In `encode`:

- input null: `IllegalArgumentException("nhwcInput must not be null")`;
- lunghezza input diversa da `224*224*3`: `IllegalArgumentException` con messaggio esplicativo.

Le violazioni di contratto del modello producono `IllegalStateException`.

## 7. Lifecycle

La classe implementa `AutoCloseable`; `close()` chiude l'`Interpreter`. Non è presente alcuna gestione di stato aggiuntiva oltre al riferimento `interpreter`.

## 8. Cosa NON è stato implementato

Confermato: NON sono presenti

- Bitmap
- resize
- crop
- square padding
- normalizzazione ImageNet
- SSD
- background compensation
- L2 normalization
- Fisher
- Mahalanobis
- Welford
- VSA
- classificazione
- enrollment
- GPU/NNAPI

Il wrapper si limita a: caricare il modello, validare il contratto, copiare float NHWC in un `ByteBuffer` diretto ed eseguire l'inferenza restituendo il vettore `float[960]`.

## 9. Verifica eseguita

Comando eseguito:

```text
.\gradlew.bat :app:compileDebugJavaWithJavac --console=plain
```

Risultato:

```text
BUILD SUCCESSFUL in 12s
8 actionable tasks: 1 executed, 7 up-to-date
```

Questa verifica riguarda esclusivamente la **compilazione Java**. NON verifica l'inferenza reale su dispositivo Android né la correttezza numerica dell'output.

## 10. Prossimo passo minimo

Realizzare un smoke test runtime Android del wrapper `MobileNetV4Encoder` su input float NHWC già preparato, prima di implementare il preprocessing Bitmap/crop -> tensor.
