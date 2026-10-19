# Report sessione — Export MobileNetV4 encoder (PyTorch → LiteRT)

Data: 2026-10-19  
Branch: `feat/latent-v4`  
Tipo di sessione: recupero da task MobileNetV4 interrotto + export encoder e verifica di parità PyTorch → LiteRT.

> Nota: questo report descrive esclusivamente ciò che è stato realmente fatto. La preparazione è avvenuta localmente; la prima esecuzione reale, la conversione LiteRT e il test di parità sono stati eseguiti in Google Colab. Il risultato definitivo è documentato nella sezione 13.

---

## 1. Stato Git

### Branch corrente

```text
feat/latent-v4
```

### `git status --short`

Snapshot rilevato **prima** della creazione di questo report:

```text
 M .idea/misc.xml
 M app/build.gradle.kts
 M app/src/main/AndroidManifest.xml
 M app/src/main/java/com/antonio/visionsector/encoder/FeatureEncoder.java
 M app/src/main/java/com/antonio/visionsector/engine/vsa/DatasetStatTracker.java
 M app/src/main/java/com/antonio/visionsector/network/RetrofitClient.java
 M app/src/main/java/com/antonio/visionsector/network/VSHubApi.java
 M app/src/main/java/com/antonio/visionsector/network/dto/ClassEntryDTO.java
 M app/src/main/java/com/antonio/visionsector/network/dto/ModuleBundleDTO.java
 M app/src/main/java/com/antonio/visionsector/ui/MainActivity.java
 M app/src/main/java/com/antonio/visionsector/utils/LocalStorageManager.java
 M app/src/main/res/layout/activity_main.xml
 M app/src/main/res/values/strings.xml
?? Android_060726.md
?? Android_070726.md
?? Android_VSA_Description.md
?? DettagliAndroid.md
?? Pulizia_10072026.md
?? ResocontoAndroid_11072026.md
?? app/release/app-release.apk
?? app/release/baselineProfiles/0/app-release.dm
?? app/release/baselineProfiles/1/app-release.dm
?? app/release/output-metadata.json
?? app/src/main/java/com/antonio/visionsector/encoder/latent/BackgroundCompensator.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/Base64Codec.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/BicubicResize.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureCapture.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureStats.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentBoundingBox.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizer.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoder.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoderContract.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentExportHook.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentPreprocessing.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecord.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecordCodec.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentSmokeBenchmark.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LatentTensorValidator.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/LegacyBlackPaddingCompensator.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/ModelFingerprint.java
?? app/src/main/java/com/antonio/visionsector/encoder/latent/NoBackgroundCompensator.java
?? app/src/main/session-ses_0b43.md
?? app/src/test/java/com/antonio/visionsector/encoder/latent/BackgroundCompensatorTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/Base64CodecTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/BlackSignatureStatsTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizerTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/LatentEncoderContractTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/LatentPreprocessingTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/LatentRecordCodecTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/LatentTensorValidatorTest.java
?? app/src/test/java/com/antonio/visionsector/encoder/latent/ModelFingerprintTest.java
?? app/src/test/java/com/antonio/vsatest/RetrofitClientTest.java
?? docs/CURRENT_ANDROID_PIPELINE_AUDIT.md
?? session-ses_0ce3.md
?? session-ses_0e75.md
?? session-ses_0fbf.md
?? session-ses_eb08.md
?? tools/__pycache__/export_mobilenetv4_latent.cpython-313.pyc
?? tools/export_mobilenetv4_latent.py
?? tools/mobilenetv4_export/export_encoder.py
?? tools/mobilenetv4_export/requirements.txt
?? tools/requirements-latent-export.txt
```

Dopo la creazione di questo report, `git status` mostrava in più:

```text
?? docs/MOBILENETV4_EXPORT_SESSION_REPORT.md
```

Dopo la successiva esecuzione Colab si sono aggiunti anche gli artefatti in `tools/mobilenetv4_export/out/` e la copia in `app/src/main/assets/`.

### `git diff --stat`

```text
 .idea/misc.xml                                     |   2 +-
 app/build.gradle.kts                               |  20 +-
 app/src/main/AndroidManifest.xml                   |   4 +-
 .../visionsector/encoder/FeatureEncoder.java       |  26 +-
 .../engine/vsa/DatasetStatTracker.java             |  24 +-
 .../visionsector/network/RetrofitClient.java       | 143 ++-
 .../com/antonio/visionsector/network/VSHubApi.java |  19 +
 .../visionsector/network/dto/ClassEntryDTO.java    |  28 +-
 .../visionsector/network/dto/ModuleBundleDTO.java  |  36 +-
 .../com/antonio/visionsector/ui/MainActivity.java   | 997 +++++++++++++++++++--
 .../visionsector/utils/LocalStorageManager.java    | 280 +++++-
 app/src/main/res/layout/activity_main.xml          |  19 +
 app/src/main/res/values/strings.xml                |   4 +-
 13 files changed, 1493 insertions(+), 109 deletions(-)
```

### File untracked rilevanti

Sono presenti numerosi file untracked preesistenti, tra cui:

- documentazione root: `Android_060726.md`, `Android_070726.md`, `Android_VSA_Description.md`, `DettagliAndroid.md`, `Pulizia_10072026.md`, `ResocontoAndroid_11072026.md`;
- build/release: `app/release/app-release.apk`, `app/release/baselineProfiles/0/app-release.dm`, `app/release/baselineProfiles/1/app-release.dm`, `app/release/output-metadata.json`;
- session log: `session-ses_0ce3.md`, `session-ses_0e75.md`, `session-ses_0fbf.md`, `session-ses_eb08.md`, `app/src/main/session-ses_0b43.md`;
- audit: `docs/CURRENT_ANDROID_PIPELINE_AUDIT.md`;
- test preesistente: `app/src/test/java/com/antonio/vsatest/RetrofitClientTest.java`.

### A. File/modifiche già presenti prima di questa sessione

I 13 file tracked modificati elencati sopra erano già modificati prima di questa sessione. In particolare:

- 12 file hanno timestamp di luglio 2026;
- `.idea/misc.xml` ha timestamp 2026-10-19 08:16:08, precedente al primo file del tentativo MobileNetV4 interrotto (08:34);
- nessuno di questi file è stato modificato durante questa sessione.

I file untracked del precedente tentativo MobileNetV4 erano già presenti all'inizio di questa sessione:

- 18 file in `app/src/main/java/com/antonio/visionsector/encoder/latent/`;
- 9 test in `app/src/test/java/com/antonio/visionsector/encoder/latent/`;
- `tools/export_mobilenetv4_latent.py`;
- `tools/requirements-latent-export.txt`;
- `tools/__pycache__/export_mobilenetv4_latent.cpython-313.pyc`.

Questi file **non sono stati creati, modificati o cancellati** in questa sessione.

### B. File creati o modificati durante questa sessione

Creati:

- `tools/mobilenetv4_export/export_encoder.py`;
- `tools/mobilenetv4_export/requirements.txt`;
- `docs/MOBILENETV4_EXPORT_SESSION_REPORT.md`.

Modificato durante la sessione:

- `tools/mobilenetv4_export/export_encoder.py` — modificato durante la sessione: prima correzione del controllo classifier, poi sostituzione del percorso pre-logits con backbone + global average pooling dopo la prima esecuzione Colab.
- `tools/mobilenetv4_export/requirements.txt` — creato e successivamente aggiornato a `torch==2.11.0+cpu` e `torchvision==0.26.0+cpu`.

Non sono stati modificati codice Android, file Gradle, file della pipeline VSA legacy o file del precedente tentativo. La successiva esecuzione Colab ha generato gli artefatti di export; il solo `.tflite` è stato inoltre copiato in `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite`; nessuna integrazione nel codice è stata implementata.

---

## 2. Obiettivo della sessione

Il task assegnato era esclusivamente preparare e validare il percorso:

```text
MobileNetV4 PyTorch pretrained
→ backbone feature map [N, 960, H, W]
→ global average pooling
→ embedding [N, 960]
→ LiteRT/TFLite
```

L'obiettivo era predisporre un export minimale e un test di parità tra encoder PyTorch ed encoder TFLite, senza modificare l'app Android.

Non rientravano nello scope, e non sono stati implementati:

- Fisher;
- Mahalanobis;
- Welford/ClassModel;
- VSA;
- background compensation;
- preprocessing Android;
- multi-object.

La sessione ha preparato script, requirements e comandi. La prima esecuzione Colab si è fermata al controllo dell'output PyTorch a causa del percorso pre-logits; dopo la correzione backbone + GAP, una seconda esecuzione Colab ha completato conversione LiteRT, verifica del contratto e test di parità.

---

## 3. MobileNetV4 scelto

| Campo | Valore | Stato |
|---|---|---|
| Model ID | `timm/mobilenetv4_conv_small.e2400_r224_in1k` | Fornito nel task |
| Repository/source | `timm/mobilenetv4_conv_small.e2400_r224_in1k` (Hugging Face) | Fornito nel task |
| Revisione weights | `331fb803779522b685cf942e15f914fb6741c1eb` | Verificata in Colab: modello caricato con questa revisione senza errori |
| Numero di parametri | Non riportato | Non verificato |
| `num_features` | `960` | Verificato: check `num_features == 960` superato in Colab |
| Dimensione embedding | `960` | Verificata: output GAP osservato `[4, 960]` e `[1, 960]` |
| Punto di output | `forward_features(x)` → `adaptive_avg_pool2d(features, 1)` → `flatten(1)` | Verificato in Colab: feature map `[4, 960, 7, 7]` → GAP → `[4, 960]` |
| Logits ImageNet 1000 | Esclusi per costruzione | Verificato: output definitivo a 960 dimensioni |

### Fatti verificati in questa sessione

- Lo script carica il modello pretrained con `num_classes=0`.
- Lo script verifica che `model.num_features` sia `960`.
- Lo script non usa `forward_head(..., pre_logits=True)` come output.
- L'output è ottenuto con `forward_features` seguito da `adaptive_avg_pool2d(features, 1).flatten(1)`.
- Prima della conversione, lo script verifica esplicitamente:
  - che la feature map di `forward_features` abbia rank 4;
  - che la channel dimension sia `960`;
  - che l'output dopo GAP sia `[N, 960]`.
- La sintassi del model ID pinnato `hf-hub:<repo_id>@<revision>` è compatibile con il codice di timm 1.0.30, per ispezione del sorgente.
- La prima esecuzione Colab ha caricato il modello e superato il check `num_features == 960`.
- La prima esecuzione reale su Colab ha osservato `(1, 1280)` dal percorso `forward_head(..., pre_logits=True)`; questo ha confermato che il pre-logits head non produce il 960-D desiderato.
- Dopo la correzione, la seconda esecuzione Colab ha osservato:
  - feature map backbone `(4, 960, 7, 7)`;
  - output GAP `(4, 960)`;
  - output PyTorch single-sample `(1, 960)`.
- La conversione LiteRT è completata e il contratto TFLite osservato è `[1, 224, 224, 3] float32 → [1, 960] float32`.

### Verifiche completate in Colab

- La revisione Hugging Face è stata caricata con successo; non è stato eseguito un checksum esterno dei weights.
- `forward_features` ha restituito una feature map con 960 canali a 224×224: `[4, 960, 7, 7]`.
- L'output dopo GAP è `[4, 960]` sul batch e `[1, 960]` sul singolo campione.
- L'export LiteRT è riuscito e produce lo stesso output a 960 dimensioni.
- Il modello esportato non usa i 1000 logits ImageNet.
- Il numero di parametri del modello resta non verificato.

Nessun peso è stato scaricato e nessun modello è stato caricato nell'ambiente Windows locale. Il caricamento e l'export sono avvenuti in Google Colab.

---

## 4. Conversione PyTorch → LiteRT/TFLite

### File preparati

#### `tools/mobilenetv4_export/export_encoder.py`

- **Scopo:** caricare il modello pinnato, rimuovere/bypassare il classifier, creare input deterministici, eseguire PyTorch, convertire in LiteRT, verificare il contratto tensoriale, confrontare gli output e scrivere `.tflite` + manifest.
- **Input:**
  - modello pretrained da Hugging Face, revisione pinnata;
  - argomenti CLI `--output-dir`, `--samples`, `--seed`;
  - tensori float32 deterministici generati con NumPy.
- **Output:**
  - `out/mobilenetv4_conv_small_encoder.tflite`;
  - `out/mobilenetv4_conv_small_encoder.manifest.json`;
  - riepilogo JSON su stdout.
- **Dipendenze:** `numpy`, `torch`, `timm`, `litert-torch`, `ai-edge-litert`.
- **Stato:** creato, modificato durante la sessione (controllo classifier, poi backbone + GAP), syntax check superata; **eseguito con successo in Google Colab**; mai eseguito localmente su Windows.

#### `tools/mobilenetv4_export/requirements.txt`

- **Scopo:** pinning del stack di conversione per Linux/Colab.
- **Input:** letto da `pip install -r`.
- **Output:** installazione dei pacchetti nel virtual environment.
- **Dipendenze:** nessuna a runtime; è un file di requisiti.
- **Stato:** creato e aggiornato a `torch==2.11.0+cpu` / `torchvision==0.26.0+cpu`; installazione eseguita con successo nell'ambiente Google Colab, non eseguita localmente su Windows.

### Flusso implementato ed eseguito in Google Colab

1. **Caricamento modello**
   - `timm.create_model(f"hf-hub:{MODEL_ID}@{HF_REVISION}", pretrained=True, num_classes=0)`.
   - `model.eval()`.
   - Controllo su `num_features == 960`.

2. **Rimozione/bypass classifier**
   - Il classifier ImageNet non viene usato.
   - La parte hidden/pre-logits della classification head non viene usata.
   - L'output viene preso con `forward_features` + global average pooling.
   - `forward_head(..., pre_logits=True)` non è più il percorso di output.

3. **Creazione input deterministico**
   - `rng = np.random.default_rng(seed)`.
   - `standard_normal((samples, 224, 224, 3)) * 2.0`.
   - Cast a `float32`.
   - Versione NHWC contigua per LiteRT.
   - Versione NCHW contigua per PyTorch, ottenuta con `transpose(0, 3, 1, 2)`.

4. **Inferenza PyTorch**
   - Verifica esplicita della feature map: rank 4, channel dimension `960`.
   - GAP con `adaptive_avg_pool2d(features, 1).flatten(1)`.
   - Verifica esplicita dell'output GAP `[N, 960]`.
   - `torch.inference_mode()`.
   - Input `torch.from_numpy(nchw)`.
   - Output convertito in NumPy `float32`.
   - Verifica della shape `[1, 960]` sul primo campione e `[samples, 960]` sul batch.

5. **Conversione LiteRT**
   - `litert_torch.to_channel_last_io(encoder, args=[0])`.
   - Input di esempio NHWC `[1, 224, 224, 3]`.
   - `torch.no_grad()`.
   - `litert_torch.convert(...)`.
   - `edge_model.export(...)`.

6. **Inferenza sul modello esportato**
   - `Interpreter(model_path=..., num_threads=1)`.
   - `allocate_tensors()`.
   - Verifica di input/output tramite `get_input_details()` e `get_output_details()`.
   - Invocazione campione per campione con tensor NHWC.

7. **Confronto output**
   - `max_abs_error`.
   - `mean_abs_error`.
   - `cosine_similarity_mean`.
   - `cosine_similarity_min`.
   - Controllo che le metriche siano finite.

8. **Scrittura `.tflite`**
   - Il file viene scritto solo al termine della conversione, in `out/mobilenetv4_conv_small_encoder.tflite`.

9. **Manifest JSON**
   - Modello e revisione.
   - Dimensione embedding.
   - Contratto input/output.
   - Metriche di parità.
   - Seed, numero campioni e descrizione della distribuzione degli input.
   - Dimensione e SHA-256 del `.tflite`.
   - Versioni dei pacchetti rilevanti.

**Importante:** l'intero flusso è stato eseguito con successo in Google Colab. Il `.tflite` e il manifest sono stati generati in `tools/mobilenetv4_export/out/`; il `.tflite` è stato copiato anche in `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite`.

---

## 5. Contratto tensoriale

### Atteso

```text
PyTorch input:  [1, 3, 224, 224] float32   (NCHW)
PyTorch output: [1, 960] float32           (forward_features → GAP)

LiteRT input:   [1, 224, 224, 3] float32   (NHWC)
LiteRT output:  [1, 960] float32
```

Lo script codifica queste attese come asserzioni:

- input LiteRT esattamente `[1, 224, 224, 3]`;
- output LiteRT esattamente `[1, 960]`;
- dtype `float32` per entrambi;
- un solo input e un solo output.

### Verificato in Google Colab

Contratto osservato:

```text
LiteRT input:   [1, 224, 224, 3] float32   (NHWC)
LiteRT output:  [1, 960] float32
```

- un solo input e un solo output;
- dtype `float32` per entrambi;
- shape esatte verificate dallo script;
- output PyTorch osservato: `[4, 960]` sul batch e `[1, 960]` sul singolo campione;
- feature map backbone osservata: `[4, 960, 7, 7]`;
- il precedente output `[1, 1280]` proveniva dal percorso pre-logits ed è stato escluso.

---

## 6. Test di parità

**ESEGUITO in Google Colab su 4 campioni deterministici.**

| Voce | Valore |
|---|---|
| Numero di input eseguiti | `4` |
| Max absolute error | `2.1979212760925293e-06` |
| Mean absolute error | `2.2774430874505926e-08` |
| Cosine similarity media | `0.9999999999732869` |
| Cosine similarity minima | `0.9999999999602538` |
| Threshold automatico | Nessuno |
| Valutazione | Parity sostanzialmente perfetta |

Lo script non implementa una soglia di accettazione PASS/FAIL automatica; verifica che le shape coincidano e che le metriche siano finite. I valori misurati sono compatibili con un export corretto.

---

## 7. Ambiente di export

### Ambiente richiesto

- **OS:** Linux, inclusa Google Colab.
- **Python:** `litert-torch` richiede Python >= 3.10; la documentazione ufficiale consiglia Python 3.11.
- **Ambiente locale Windows:** Python 3.13.5, con solo `numpy` installato.
- **Stack locale assente:** `torch`, `timm`, `litert_torch`, `ai_edge_litert`.

### Ambiente Google Colab verificato

- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`
- `torchao 0.18.0+cpu`
- `timm 1.0.30`
- import di `litert_torch` OK
- warning relativo a Triton ignorato: l'export è stato eseguito su CPU.

### Perché non è stata fatta l'installazione locale

- Il task richiedeva esplicitamente di non installare lo stack di conversione nel Python Windows locale.
- La conversione completa non doveva essere eseguita su Windows.
- L'ambiente Windows non contiene i pacchetti necessari.

### Requirements usati in Colab

File: `tools/mobilenetv4_export/requirements.txt`

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0+cpu
torchvision==0.26.0+cpu
timm==1.0.30
litert-torch==0.9.4
ai-edge-litert==2.2.0
numpy==2.3.5
pillow==12.0.0
```

### Comandi Google Colab

Da eseguire nella root del repository:

```bash
cd /content/VsaVisionSector
python3 -m venv /content/mnv4-export-venv
source /content/mnv4-export-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tools/mobilenetv4_export/requirements.txt
python tools/mobilenetv4_export/export_encoder.py --samples 4 --seed 0
```

Download opzionale dei risultati:

```python
from google.colab import files
files.download("tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.tflite")
files.download("tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.manifest.json")
```

### Comandi Linux

Da eseguire nella root del repository:

```bash
cd /path/to/VsaVisionSector
python3 -m venv .venv-mobilenetv4-export
source .venv-mobilenetv4-export/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tools/mobilenetv4_export/requirements.txt
python tools/mobilenetv4_export/export_encoder.py \
  --samples 4 \
  --seed 0 \
  --output-dir tools/mobilenetv4_export/out
```

I comandi Colab sono stati eseguiti con successo. I comandi Linux restano la variante locale equivalente; l'ambiente Windows non è stato usato per installazione o conversione.

---

## 8. File finali della sessione

| File | Creato/modificato | Scopo | Stato |
|---|---|---|---|
| `tools/mobilenetv4_export/export_encoder.py` | Creato; modificato durante la sessione (controllo classifier, poi backbone + GAP) | Export encoder PyTorch → LiteRT, verifica contratto, parity e manifest | `SYNTAX CHECKED`; `EXECUTED ON COLAB` |
| `tools/mobilenetv4_export/requirements.txt` | Creato; aggiornato a torch 2.11.0+cpu / torchvision 0.26.0+cpu | Pinning dipendenze per Linux/Colab | `INSTALLED ON COLAB` |
| `docs/MOBILENETV4_EXPORT_SESSION_REPORT.md` | Creato; aggiornato con correzione post-Colab e risultato definitivo | Report di questa sessione | `UPDATED` |
| `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.tflite` | Generato in Colab | Artefatto encoder LiteRT | `READY`; SHA-256 `7F5D8D7796C5A48987748BB1CAD6C9E20788E6B92FD7D4D5F9C011D8CAA90F15` |
| `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.manifest.json` | Generato in Colab | Manifest con contratto e metriche di parity | `READY` |
| `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite` | Copia dell'artefatto export | Copia Android dell'artefatto | `READY`; SHA-256 identico alla copia export |

Nessun codice Android, file Gradle o file VSA è stato modificato; l'unico file Android aggiunto è la copia dell'artefatto in `app/src/main/assets/`.

---

## 9. Verifiche realmente eseguite

| Comando/test | Esito | Note |
|---|---|---|
| `git branch --show-current` | PASS | Output: `feat/latent-v4` |
| `git status --short --untracked-files=all` | PASS | Inventario del worktree |
| `git diff --stat` | PASS | 13 file, 1493 inserimenti, 109 cancellazioni |
| `Get-ChildItem` su directory latent e `tools` per timestamp | PASS | Usato per distinguere i file del tentativo precedente |
| `Get-Content -LiteralPath 'session-ses_eb08.md' -TotalCount 40` | PASS | Identificato come transcript di audit precedente |
| `Select-String` su `session-ses_eb08.md` | PASS | Confermata la natura di session log, non codice sorgente |
| `Test-Path -LiteralPath 'tools'` | PASS | `True` |
| `Test-Path -LiteralPath 'tools/mobilenetv4_export'` | PASS | `False` prima della creazione |
| `python -c "from pathlib import Path; p=Path('tools/mobilenetv4_export/export_encoder.py'); compile(p.read_text(encoding='utf-8'), str(p), 'exec'); print('syntax OK')"` | PASS | Output: `syntax OK` (prima della correzione GAP) |
| Stesso syntax check dopo la correzione backbone + GAP | PASS | Output: `syntax OK` |
| Prima esecuzione reale su Colab | FAIL/STOPPED | Output PyTorch `(1, 1280)`, poi `RuntimeError: unexpected PyTorch output shape (1, 1280), expected (1, 960)`; nessuna conversione TFLite |
| Seconda esecuzione reale su Colab dopo correzione GAP | PASS | Feature map `(4, 960, 7, 7)`, GAP `(4, 960)`, PyTorch single-sample `(1, 960)`; conversione LiteRT completata |
| `Get-ChildItem -Recurse -File 'tools\mobilenetv4_export'` | PASS | Mostrati i file creati |
| `git status --short --untracked-files=all \| Select-String -Pattern 'mobilenetv4_export\|requirements.txt'` | PASS | Confermati i file untracked nuovi |
| `Invoke-WebRequest` verso il sorgente timm `classifier.py` | FAIL/INCONCLUSIVE | Nessun output; non usato come validazione |
| Ispezione sorgente/wheel di `litert-torch==0.9.4` | PASS (solo ispezione) | Confermate API `convert`, `export`, `to_channel_last_io` |
| Ispezione sorgente di `timm==1.0.30` per `hf-hub:<repo>@<revision>` | PASS (solo ispezione) | Confermata la sintassi |
| `python -m py_compile ...` | NOT RUN | È stato usato `compile()` direttamente |
| Verifica versioni pacchetti locali | NOT RUN localmente | `torch`, `timm`, `litert-torch`, `ai-edge-litert` non sono installati localmente; ambiente Colab verificato: torch 2.11.0+cpu, torchvision 0.26.0+cpu, torchao 0.18.0+cpu, timm 1.0.30, `litert_torch` import OK |
| Download modello / verifica revisione Hugging Face | PASS (Colab) | Modello caricato con revisione pinnata; nessun checksum esterno dei weights |
| Inferenza PyTorch | PASS (Colab) | Feature map `(4, 960, 7, 7)`, GAP `(4, 960)`, single-sample `(1, 960)` |
| `litert_torch.convert` / `.export` | PASS (Colab) | Conversione completata; `.tflite` generato |
| Apertura `.tflite` con `ai_edge_litert.Interpreter` | PASS (Colab) | Contratto osservato `[1, 224, 224, 3] float32 → [1, 960] float32` |
| Test di parità | PASS (Colab) | 4 campioni; max abs `2.1979212760925293e-06`, mean abs `2.2774430874505926e-08`, cosine mean `0.9999999999732869`, cosine min `0.9999999999602538` |
| Verifica SHA-256 copia export vs copia Android | PASS | Entrambe `7F5D8D7796C5A48987748BB1CAD6C9E20788E6B92FD7D4D5F9C011D8CAA90F15` |
| Warning Triton | IGNORED | Export eseguito su CPU; nessun impatto sul risultato |
| Gradle | NOT RUN | Intenzionalmente escluso |

Le ispezioni di sorgente non sono state considerate come validazione runtime della conversione.

---

## 10. Cosa NON è stato fatto

Intenzionalmente fuori scope e non realizzato:

- nessuna modifica alla pipeline Android;
- nessun nuovo classificatore applicativo;
- nessun Fisher;
- nessuna Mahalanobis;
- nessun Welford/ClassModel;
- nessuna VSA;
- nessun preprocessing Android definitivo;
- nessun debinding MobileNetV4;
- nessun multi-object;
- nessuna modifica a Gradle;
- nessuna esecuzione Gradle;
- nessun commit, stash, reset, checkout o cancellazione;
- nessuna modifica ai file del precedente tentativo MobileNetV4;
- nessuna generazione locale di `.tflite` o manifest; la generazione è avvenuta in Google Colab;
- nessuna integrazione Android dell'encoder; il `.tflite` è stato solo copiato in `app/src/main/assets/`, senza collegarlo al codice.

Lo script di export rimuove il classifier **solo nel modello Python usato per l'export**; non è stato aggiunto alcun classificatore all'app Android.

Eccezioni: nessuna modifica accidentale a codice Android, Gradle o VSA legacy è stata rilevata. L'unica aggiunta nel progetto Android è la copia dell'artefatto in `app/src/main/assets/`.

---

## 11. Problemi o dubbi emersi

1. **Ambiente Windows inadeguato:** `torch`, `timm`, `litert-torch` e `ai-edge-litert` non sono installati localmente; la conversione non può essere eseguita su Windows. Risolto usando Google Colab.
2. **Metadata del modello:** revisione e `num_features == 960` verificati in Colab; il numero di parametri non è stato verificato.
3. **Output `[1,960]` verificato:** il contratto TFLite osservato è `[1, 224, 224, 3] float32 → [1, 960] float32`.
4. **Layout NCHW/NHWC verificato:** l'input LiteRT osservato è NHWC `[1, 224, 224, 3]`.
5. **API LiteRT validate a runtime:** `convert`, `export` e `to_channel_last_io` sono stati eseguiti con successo in Colab.
6. **Nessuna soglia di parità:** non esiste un criterio PASS/FAIL automatico; le metriche vengono solo riportate. I valori misurati indicano parity sostanzialmente perfetta.
7. **Compatibilità Python/Colab verificata in Colab:** `torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `torchao 0.18.0+cpu`, `timm 1.0.30`, import di `litert_torch` OK. La versione Python usata non è stata riportata.
8. **Supporto operatori MobileNetV4:** la conversione LiteRT è completata; la compatibilità con il runtime Android non è ancora stata testata.
9. **Tentativo di fetch del sorgente timm senza output:** non è stato possibile usare quel controllo come verifica.
10. **Prima esecuzione Colab fermata a `(1, 1280)`:** risolta correggendo il percorso con backbone + GAP; la seconda esecuzione è completata con successo.
11. **Warning Triton:** ignorato perché l'export è stato eseguito su CPU; nessun impatto sul risultato.

---

## Correzione dopo prima esecuzione Colab

- La prima esecuzione reale su Colab ha prodotto `PyTorch single-sample output shape: (1, 1280)` e si è poi fermata con `RuntimeError: unexpected PyTorch output shape (1, 1280), expected (1, 960)`.
- Causa: `forward_head(..., pre_logits=True)` restituisce le hidden features della classification head (1280), non il backbone a 960 canali.
- Decisione: l'encoder usa `model.forward_features(x)` → feature map `[N, 960, H, W]` → `adaptive_avg_pool2d(features, 1).flatten(1)` → `[N, 960]`.
- `forward_head(..., pre_logits=True)` è stato rimosso dal percorso di output; il classifier ImageNet e la parte hidden/pre-logits della head sono bypassati.
- Aggiunte verifiche esplicite prima della conversione: rank 4 della feature map, channel dimension `960`, output GAP `[N, 960]`.
- Dopo questa correzione, la seconda esecuzione Colab ha completato la conversione TFLite.
- Ambiente Colab verificato (non verificato localmente su Windows):
  - `torch 2.11.0+cpu`
  - `torchvision 0.26.0+cpu`
  - `torchao 0.18.0+cpu`
  - `timm 1.0.30`
  - import di `litert_torch` OK
- Requirements aggiornati di conseguenza a `torch==2.11.0+cpu` e `torchvision==0.26.0+cpu`.

---

## 12. Prossimo passo minimo

L'export è completato. Il prossimo passo minimo, **non implementato**, è l'integrazione Android dell'encoder 960-D e la progettazione separata del preprocessing. Nessuna integrazione Android è stata ancora realizzata.

---

## 13. Prima esecuzione reale in Google Colab e risultato definitivo

### Ambiente Colab funzionante

- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`
- `torchao 0.18.0+cpu`
- `timm 1.0.30`
- import di `litert_torch` OK
- warning relativo a Triton ignorato perché l'export è stato eseguito su CPU.

### Modello e correzione verificata

- Model ID: `timm/mobilenetv4_conv_small.e2400_r224_in1k`
- Revisione Hugging Face: `331fb803779522b685cf942e15f914fb6741c1eb`
- `forward_features` ha prodotto `[N, 960, 7, 7]`.
- Global Average Pooling ha prodotto `[N, 960]`.
- Il precedente output `[N, 1280]` proveniva dalla parte pre-logits della head ed è stato escluso.

Output osservato:

```text
Backbone feature map shape: (4, 960, 7, 7)
Global-average-pooled embedding shape: (4, 960)
PyTorch single-sample output shape: (1, 960)
```

### Conversione e contratto TFLite

La conversione LiteRT è completata con successo.

```text
Input:  [1, 224, 224, 3] float32
Output: [1, 960] float32
```

### Parity su 4 campioni deterministici

| Metrica | Valore |
|---|---|
| `max_abs_error` | `2.1979212760925293e-06` |
| `mean_abs_error` | `2.2774430874505926e-08` |
| `cosine_similarity_mean` | `0.9999999999732869` |
| `cosine_similarity_min` | `0.9999999999602538` |

### Artefatti generati

- `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.tflite`
- `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.manifest.json`
- Copia Android: `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite`

Verifica SHA-256 tra copia export e copia Android:

```text
7F5D8D7796C5A48987748BB1CAD6C9E20788E6B92FD7D4D5F9C011D8CAA90F15
```

Entrambe le copie hanno lo stesso hash.

### Conclusione

- Export PyTorch → LiteRT completato.
- Contratto tensoriale verificato.
- Embedding backbone definitivo: **960-D**.
- Parity numerica sostanzialmente perfetta.
- Artefatto pronto per l'integrazione Android.
- Nessuna integrazione Android è stata ancora implementata.

L'aggiornamento di questo report non ha modificato codice Android e non ha eseguito Gradle.
