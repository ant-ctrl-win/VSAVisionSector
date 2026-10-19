# MobileNetV4 Android Runtime Smoke Test — Report

Branch: `feat/latent-v4`

Data: verifica eseguita nella sessione corrente.

## 1. File creati

- `app/src/androidTest/java/com/antonio/vsatest/MobileNetV4EncoderRuntimeTest.java`
- `docs/MOBILENETV4_ANDROID_RUNTIME_SMOKE_REPORT.md`

Nessun altro file modificato. `MobileNetV4Encoder.java`, `MainActivity.java`, `FeatureEncoder.java`, Gradle, manifest, pipeline VSA, networking, storage e layout sono rimasti invariati.

## 2. Tipo di test scelto

Test instrumentation Android minimale (`androidTest`), eseguito con `AndroidJUnit4`.

Il progetto aveva già l'infrastruttura necessaria senza modifiche a Gradle o manifest:

- `app/src/androidTest/` presente;
- `testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"` in `app/build.gradle.kts`;
- dipendenze `androidx.test.ext:junit` e `espresso-core` già presenti.

Il test istanzia direttamente `MobileNetV4Encoder` con il `Context` target dell'apparecchio, senza passare da `MainActivity`.

## 3. Input sintetico utilizzato

Array `float[224 * 224 * 3]` (150528 elementi) generato in modo deterministico e riproducibile:

```java
input[i] = ((i * 31) % 256) / 255f;
```

Non rappresenta un'immagine reale e non viene applicato alcun preprocessing.

## 4. Controlli implementati

Il test `encodeSyntheticInputTwice` verifica:

1. caricamento reale del modello dall'asset (la costruzione di `MobileNetV4Encoder` avviene dal `Context` target e non lancia eccezioni);
2. costruzione del wrapper senza eccezioni;
3. completamento dell'inferenza (nessuna eccezione da `encode`);
4. `output.length == 960` su entrambe le inferenze;
5. tutti i valori di output finiti (`Float.isFinite`, quindi né `NaN` né `Infinity`);
6. norma L2 dell'output `> 0`;
7. due inferenze consecutive sullo stesso input con scarto massimo assoluto `<= 1e-6`;
8. tempi separati della prima e della seconda inferenza misurati con `System.nanoTime()`.

Output diagnostico previsto (stampato con `Log.i` e `System.out`): lunghezza output, min, max, mean, norma L2, tempo prima inferenza, tempo seconda inferenza, massimo scarto assoluto. I 960 valori non vengono stampati.

La costruzione del wrapper (caricamento modello) avviene prima dell'avvio del cronometro, quindi i tempi riportati riguardano la sola inferenza.

## 5. Comando di compilazione eseguito

```text
.\gradlew.bat :app:compileDebugAndroidTestJavaWithJavac --console=plain
```

Risultato reale:

```text
BUILD SUCCESSFUL in 4s
28 actionable tasks: 10 executed, 18 up-to-date
```

Stato: **COMPILE PASS**.

## 6. Comando runtime eseguito

Comando realmente eseguito:

```text
.\gradlew.bat :app:connectedDebugAndroidTest --console=plain "-Pandroid.testInstrumentationRunnerArguments.class=com.antonio.vsatest.MobileNetV4EncoderRuntimeTest"
```

Dispositivo: `Pixel_7(AVD) - 17`.

Risultato reale:

```text
BUILD SUCCESSFUL in 34s
Starting 1 tests on Pixel_7(AVD) - 17
Finished 1 tests on Pixel_7(AVD) - 17
```

Risultato XML: tests = 1, failures = 0, errors = 0, skipped = 0.

Informazioni runtime osservate:

- `libtensorflowlite_jni.so` caricata correttamente;
- TensorFlow Lite runtime inizializzato;
- XNNPACK delegate CPU creato;
- 64 nodi su 65 delegati a XNNPACK;
- il mancato caricamento opzionale di `libtensorflowlite_jni_gms_client.so` non ha impedito l'esecuzione.

## 7. Output diagnostico reale

- outputLength = `960`
- min = `0.0`
- max = `0.7206121`
- mean = `0.0129930985814705`
- l2Norm = `1.7090111329405644`
- firstInferenceMs = `88.9196`
- secondInferenceMs = `58.3333`
- maxAbsDiff = `0.0`

I tempi provengono da un emulatore Android x86_64 e NON devono essere considerati benchmark rappresentativi di un dispositivo fisico ARM.

## 8. Stato finale

**COMPILE PASS** — **RUNTIME PASS**

Il test instrumentation è stato realmente eseguito su `Pixel_7(AVD) - 17` con 1 test, 0 failure, 0 error, 0 skipped. Il modello è stato caricato dall'asset, l'inferenza è stata completata, l'output ha lunghezza 960 con valori finiti, norma L2 > 0 e due inferenze consecutive identiche (`maxAbsDiff = 0.0`).

## 9. Problemi reali

- Nessun problema bloccante: test e inferenza completati con successo.
- Il caricamento opzionale di `libtensorflowlite_jni_gms_client.so` non è avvenuto, ma non ha impedito l'esecuzione.
- I tempi misurati provengono da un emulatore x86_64 e non sono rappresentativi di un dispositivo fisico ARM.

## 10. Prossimo passo

Implementare e validare il preprocessing crop/Bitmap -> NHWC float per MobileNetV4.
