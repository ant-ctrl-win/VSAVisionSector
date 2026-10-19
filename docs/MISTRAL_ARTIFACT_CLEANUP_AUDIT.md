# Audit artefatti Mistral / MobileNetV4

Audit read-only eseguito il 2026-10-19 sulla worktree corrente. Nessuna rimozione, nessun comando Gradle, nessuna modifica ad Android, Gradle, VSA o file tracked. L'unica modifica prodotta è la creazione di questo report.

Nei file Java non compare la stringa `Mistral`. L'attribuzione al tentativo Mistral deriva dal transcript `session-ses_eb08.md`, dove l'assistant è identificato come `Build · Mistral Large`, e dalla corrispondenza esatta di path, contenuti e timestamp. Il tema tecnico del tentativo è MobileNetV4.

## 1. Stato Git

- Branch corrente: `feat/latent-v4`
- HEAD: `28609cd Add validated MobileNetV4 latent encoder` (`origin/feat/latent-v4`)
- Nessun commit, stash, reset, checkout o clean eseguito.

File tracked modificati (13, preesistenti e non candidati alla rimozione):

```text
.idea/misc.xml
app/build.gradle.kts
app/src/main/AndroidManifest.xml
app/src/main/java/com/antonio/visionsector/encoder/FeatureEncoder.java
app/src/main/java/com/antonio/visionsector/engine/vsa/DatasetStatTracker.java
app/src/main/java/com/antonio/visionsector/network/RetrofitClient.java
app/src/main/java/com/antonio/visionsector/network/VSHubApi.java
app/src/main/java/com/antonio/visionsector/network/dto/ClassEntryDTO.java
app/src/main/java/com/antonio/visionsector/network/dto/ModuleBundleDTO.java
app/src/main/java/com/antonio/visionsector/ui/MainActivity.java
app/src/main/java/com/antonio/visionsector/utils/LocalStorageManager.java
app/src/main/res/layout/activity_main.xml
app/src/main/res/values/strings.xml
```

Riepilogo untracked:

- 18 file Java nel package `app/src/main/java/com/antonio/visionsector/encoder/latent/`
- 9 file di test in `app/src/test/java/com/antonio/visionsector/encoder/latent/`
- 3 artefatti del vecchio tooling: `tools/export_mobilenetv4_latent.py`, `tools/requirements-latent-export.txt`, `tools/__pycache__/export_mobilenetv4_latent.cpython-313.pyc`
- `session-ses_eb08.md`, transcript della sessione Mistral
- documentazione storica Android, artefatti `app/release/*`, session log generici, `docs/CURRENT_ANDROID_PIPELINE_AUDIT.md`, `app/src/test/java/com/antonio/vsatest/RetrofitClientTest.java`
- `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.tflite` (artefatto validato, da conservare)

Il commit `28609cd` contiene esclusivamente i 5 file validati dell'export nuovo (asset Android, script, requirements, manifest, report). Non contiene il package latent né il vecchio tooling: `git ls-files` e `git log --all` non restituiscono nulla per quei path.

## 2. Artefatti chiaramente attribuibili al tentativo Mistral

I 30 artefatti tecnici sotto sono untracked e non sono referenziati da codice tracked. I file del package latent si referenziano solo tra loro e dai test untracked; le colonne “Referenziato?” si riferiscono a riferimenti da codice tracked corrente. Il transcript Mistral è riportato in coda alla tabella con decisione `UNCERTAIN`.

| File | Tracked? | Scopo | Evidenza | Referenziato? | Decisione |
|---|---|---|---|---|---|
| `app/src/main/java/com/antonio/visionsector/encoder/latent/BackgroundCompensator.java` | No | Interfaccia di compensazione del background latente (`compensate`, `id`) | Transcript Mistral `session-ses_eb08.md` rr. 19097–19386; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/Base64Codec.java` | No | Encode/decode Base64 per record float32 | Transcript Mistral; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/BicubicResize.java` | No | Resize bicubico Android Bitmap (`A=-0.75`) | Transcript Mistral; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureCapture.java` | No | Cattura embedding/statistiche/compensatore del padding nero | Transcript Mistral; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureStats.java` | No | Statistiche (norma, media, std, min, max) di un embedding | Transcript Mistral; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentBoundingBox.java` | No | Value object bounding box normalizzato `[0,1]` | Transcript Mistral; timestamp 08:38:07 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizer.java` | No | Normalizzazione L2 e calcolo norma | Transcript Mistral; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoder.java` | No | Encoder TFLite latente con fallback GPU, preprocessing, contratto e tempi | Transcript Mistral rr. 12783–12810, 19097–19386; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoderContract.java` | No | Contratto modello: ID/SHA, geometria input, dtype, dimensione embedding | Transcript Mistral r. 19750+; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentExportHook.java` | No | Scrittura record JSONL in `filesDir/latent_exports` | Transcript Mistral; timestamp 08:38:07 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentPreprocessing.java` | No | Preprocessing ImageNet RGB/bicubico 224, mean/std, factory MobileNetV4 | Transcript Mistral r. 19735+; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecord.java` | No | Schema record latente con metadati e embedding | Transcript Mistral; timestamp 08:38:07 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecordCodec.java` | No | Codec Gson/Base64 little-endian float32 | Transcript Mistral; timestamp 08:38:07 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentSmokeBenchmark.java` | No | Benchmark preprocessing/inferenza/L2/compensazione, con classe nested | Transcript Mistral; timestamp 08:38:07 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LatentTensorValidator.java` | No | Validazione input NHWC `[1,H,W,C]` e output rank-2 `[1,D]` | Transcript Mistral; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/LegacyBlackPaddingCompensator.java` | No | Compensazione legacy black-subtraction + ReLU (`legacy_black_subtraction_relu_v1`) | Transcript Mistral; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/ModelFingerprint.java` | No | SHA-256 di byte/file del modello | Transcript Mistral; timestamp 08:34:36 | No | `SAFE_TO_REMOVE` |
| `app/src/main/java/com/antonio/visionsector/encoder/latent/NoBackgroundCompensator.java` | No | Compensatore no-op (`ID="none"`) e validazione padding ratio | Transcript Mistral; timestamp 08:36:46 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/BackgroundCompensatorTest.java` | No | Test compensatori no-op/legacy | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/Base64CodecTest.java` | No | Test Base64 su vettori noti | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/BlackSignatureStatsTest.java` | No | Test statistiche firma nera | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizerTest.java` | No | Test normalizzazione L2 | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/LatentEncoderContractTest.java` | No | Test contratto/dimensioni dinamiche | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/LatentPreprocessingTest.java` | No | Test contratto preprocessing MobileNetV4 ImageNet | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/LatentRecordCodecTest.java` | No | Test round-trip record/codec float32 | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/LatentTensorValidatorTest.java` | No | Test validazione shape NHWC/rank-2 | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `app/src/test/java/com/antonio/visionsector/encoder/latent/ModelFingerprintTest.java` | No | Test SHA-256 su vettori noti | Transcript Mistral; timestamp 08:40:55 | No | `SAFE_TO_REMOVE` |
| `tools/export_mobilenetv4_latent.py` | No | Vecchio export con resize/preprocessing e `forward_head(..., pre_logits=True)` → 1280-D; superato dal nuovo `tools/mobilenetv4_export/export_encoder.py` | Transcript Mistral r. 20372; timestamp 08:34; contiene il path errato 1280-D | No | `SAFE_TO_REMOVE` |
| `tools/requirements-latent-export.txt` | No | Vecchi pin `torch==2.8.0` / `torchvision==0.23.0`, senza CPU extra-index; superato dal nuovo requirements | Transcript Mistral r. 20387; timestamp 08:34 | No | `SAFE_TO_REMOVE` |
| `tools/__pycache__/export_mobilenetv4_latent.cpython-313.pyc` | No | Bytecode CPython 3.13 generato dal vecchio script | Timestamp 08:41:05; `pyc` del vecchio `export_mobilenetv4_latent.py` | No | `SAFE_TO_REMOVE` |
| `session-ses_eb08.md` | No | Transcript completo della sessione Mistral Large (audit + implementazione latent) | 112 occorrenze `Mistral`; header `Build · Mistral Large`; rr. 12765, 19341, 20398; timestamp 2026-10-18/19 | No (solo transcript) | `UNCERTAIN` |

Nota: la rimozione dell'intero package latent elimina anche l'input di compilazione che Gradle includerebbe automaticamente dal source set `app/src/main/java`, ma nessun file tracked dipende da queste classi.

## 3. File da conservare

### Export MobileNetV4 validato

- `tools/mobilenetv4_export/export_encoder.py` (tracked): script definitivo backbone `forward_features` + GAP → `[N,960]`.
- `tools/mobilenetv4_export/requirements.txt` (tracked): pin `torch==2.11.0+cpu` / `torchvision==0.26.0+cpu`.
- `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.manifest.json` (tracked): manifest del modello validato.
- `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.tflite` (untracked): artefatto TFLite validato, SHA-256 `7F5D8D7796C5A48987748BB1CAD6C9E20788E6B92FD7D4D5F9C011D8CAA90F15`.
- `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite` (tracked): copia Android con lo stesso SHA-256.
- `docs/MOBILENETV4_EXPORT_SESSION_REPORT.md` (tracked): report definitivo dell'export e della parity.

### Documentazione e audit protetti

- `docs/CURRENT_ANDROID_PIPELINE_AUDIT.md`: audit storico della pipeline Android, esplicitamente protetto.
- `Android_060726.md`, `Android_070726.md`, `Android_VSA_Description.md`, `DettagliAndroid.md`, `Pulizia_10072026.md`, `ResocontoAndroid_11072026.md`: documentazione storica Android.
- `app/release/app-release.apk`, `app/release/baselineProfiles/0/app-release.dm`, `app/release/baselineProfiles/1/app-release.dm`, `app/release/output-metadata.json`: artefatti di release preesistenti.
- `app/src/test/java/com/antonio/vsatest/RetrofitClientTest.java`: test preesistente, non collegato al tentativo latent.

### Session log generici

- `session-ses_0ce3.md`, `session-ses_0e75.md`, `session-ses_0fbf.md`, `app/src/main/session-ses_0b43.md`: nessuna occorrenza di `Mistral`, `latent` o dei path del tentativo; non sono attribuibili con evidenza al tentativo Mistral.

### File tracked modificati

I 13 file tracked modificati elencati nella sezione 1 restano invariati. Sono modifiche preesistenti, non artefatti Mistral e non candidati alla rimozione.

## 4. File incerti

### `session-ses_eb08.md`

- **Tracked?** No, untracked.
- **Scopo:** transcript completo della sessione Mistral Large, inclusa la fase di audit read-only e la successiva implementazione latent.
- **Evidenza:** 112 occorrenze di `Mistral`; header `Build · Mistral Large`; contiene la pianificazione e le tool call che hanno creato esattamente i path del package latent e del vecchio tooling.
- **Referenziato?** No da codice o build.
- **Decisione:** `UNCERTAIN`.

L'attribuzione al tentativo è certa. L'incertezza riguarda solo l'opportunità di rimuoverlo: è l'unica prova storica del tentativo e del perché è stato interrotto. Non è incluso nella lista `SAFE_TO_REMOVE`.

## 5. Lista esatta proposta per la rimozione

Solo i path classificati `SAFE_TO_REMOVE`. Nessuna rimozione è stata eseguita.

```text
app/src/main/java/com/antonio/visionsector/encoder/latent/BackgroundCompensator.java
app/src/main/java/com/antonio/visionsector/encoder/latent/Base64Codec.java
app/src/main/java/com/antonio/visionsector/encoder/latent/BicubicResize.java
app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureCapture.java
app/src/main/java/com/antonio/visionsector/encoder/latent/BlackSignatureStats.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentBoundingBox.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizer.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoder.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentEncoderContract.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentExportHook.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentPreprocessing.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecord.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentRecordCodec.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentSmokeBenchmark.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LatentTensorValidator.java
app/src/main/java/com/antonio/visionsector/encoder/latent/LegacyBlackPaddingCompensator.java
app/src/main/java/com/antonio/visionsector/encoder/latent/ModelFingerprint.java
app/src/main/java/com/antonio/visionsector/encoder/latent/NoBackgroundCompensator.java
app/src/test/java/com/antonio/visionsector/encoder/latent/BackgroundCompensatorTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/Base64CodecTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/BlackSignatureStatsTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/LatentCanonicalizerTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/LatentEncoderContractTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/LatentPreprocessingTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/LatentRecordCodecTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/LatentTensorValidatorTest.java
app/src/test/java/com/antonio/visionsector/encoder/latent/ModelFingerprintTest.java
tools/export_mobilenetv4_latent.py
tools/requirements-latent-export.txt
tools/__pycache__/export_mobilenetv4_latent.cpython-313.pyc
```

## 6. Verifica dipendenze

La rimozione proposta lascerebbe intatti:

- **Encoder MobileNetV4 validato:** `app/src/main/assets/mobilenetv4_conv_small_encoder.tflite` è tracked, non è referenziato dal package latent e resta invariato.
- **Export script nuovo:** `tools/mobilenetv4_export/export_encoder.py` è tracked, non importa il vecchio script né classi latent, e resta invariato.
- **Manifest:** `tools/mobilenetv4_export/out/mobilenetv4_conv_small_encoder.manifest.json` è tracked e resta invariato.
- **Report export:** `docs/MOBILENETV4_EXPORT_SESSION_REPORT.md` è tracked e resta invariato.
- **Vecchia pipeline Android tracked:** i 13 file tracked modificati restano invariati; nessuna classe latent è referenziata da `FeatureEncoder`, `MainActivity`, manifest o altri file tracked.
- **`build.gradle.kts`:** nessun riferimento a `latent`, `Latent`, `MobileNetV4` o ai path del vecchio tooling; resta invariato.
- **`MainActivity`:** nessun riferimento a classi latent o al nuovo asset; resta invariato.
- **`FeatureEncoder`:** nessun riferimento a classi latent; resta invariato.

Le ricerche `git grep` e `Select-String` su codice tracked non hanno trovato riferimenti a `encoder.latent`, `LatentEncoder`, `LatentRecord`, `LatentPreprocessing`, `LatentCanonicalizer`, `ModelFingerprint`, `BackgroundCompensator`, `LatentExportHook`, `LatentTensorValidator` o `LatentEncoderContract`. I file latent risultano quindi completamente scollegati dal codice tracked; si collegano solo tra loro e ai test untracked dello stesso package.

La rimozione non tocca la cronologia Git, perché tutti i 30 path proposti sono untracked e non compaiono in nessun commit (`git log --all` e `git ls-files` non restituiscono nulla).

## 7. Prossimo passo minimo

“rimuovere i file classificati SAFE_TO_REMOVE dopo revisione umana”
