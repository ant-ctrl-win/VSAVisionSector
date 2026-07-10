# Resoconto Android — VsaVisionSector (09/07/2026)

Applicazione Android in Java per la gestione di modelli VSA (Vector Symbolic Architectures)
locali tramite TensorFlow Lite, con comunicazione REST verso un backend locale via Wi-Fi.

---

## 1. Configurazione di Rete

### 1.1 AndroidManifest.xml — Permessi e Cleartext

**File:** `app/src/main/AndroidManifest.xml`

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

```xml
<application
    android:usesCleartextTraffic="true"
    ...>
```

- Solo due permessi: `CAMERA` e `INTERNET`.
- `usesCleartextTraffic="true"` abilita HTTP in chiaro verso il backend locale `http://192.168.178.42:8080/`.
- Nessun `network_security_config` XML separato — tutto gestito via attributo diretto nel manifest.
- Camera dichiarata come `required="true"`.

### 1.2 Dipendenze di Rete — build.gradle.kts

**File:** `app/build.gradle.kts`

```kotlin
// Retrofit per le chiamate REST
implementation("com.squareup.retrofit2:retrofit:2.9.0")
// Convertitore JSON (Gson)
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
```

- Retrofit `2.9.0` con converter Gson.
- OkHttp e Gson sono dipendenze transitive (non dichiarate esplicitamente).
- Compile SDK: 34, Min SDK: 24, Target SDK: 34, Java 11.

---

## 2. Struttura del Package `network`

### 2.1 DTO

#### `ModuleBundleDTO` (`network/dto/ModuleBundleDTO.java`)

| Campo             | Tipo                  | SerializedName       | Descrizione                          |
|-------------------|-----------------------|----------------------|--------------------------------------|
| `moduleName`      | `String`              | `"moduleName"`       | Nome del modulo VSA                  |
| `visibility`      | `String`              | `"visibility"`       | Visibilità (es. "public"/"private")  |
| `encoderFileName` | `String`              | `"encoderFileName"`  | Nome file del modello encoder TFLite |
| `statsFileName`   | `String`              | `"statsFileName"`    | Nome file delle statistiche di dataset |
| `rademacherSeed`  | `Long`                | `"rademacherSeed"`   | Seed per la proiezione Rademacher    |
| `classes`         | `List<ClassEntryDTO>` | `"classes"`          | Lista delle classi apprese           |

#### `ClassEntryDTO` (`network/dto/ClassEntryDTO.java`)

| Campo         | Tipo      | SerializedName  | Descrizione                                   |
|---------------|-----------|-----------------|-----------------------------------------------|
| `name`        | `String`  | `"name"`        | Nome della classe                             |
| `sampleCount` | `Integer` | `"sampleCount"` | Numero di campioni per la classe              |
| `vectorBase64`| `String`  | `"vectorBase64"`| Vettore VSA (ipervettore) codificato in Base64 |

### 2.2 RetrofitClient — Configurazione Singleton

**File:** `network/RetrofitClient.java`

```java
private static final String BASE_URL = "http://192.168.178.42:8080/";

OkHttpClient client = new OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build();

retrofit = new Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build();
```

- **BASE_URL:** IP locale sulla porta 8080, protocollo HTTP.
- **Timeout:** connect 10s, read 15s.
- **Retry:** abilitato su fallimento di connessione.
- **Converter:** Gson per deserializzazione JSON automatica.
- Pattern Singleton: `retrofit` è `static`, creato una sola volta.

### 2.3 Interfaccia VSHubApi

**File:** `network/VSHubApi.java`

```java
public interface VSHubApi {

    @GET("api/vi/modules")
    Call<List<ModuleBundleDTO>> getAllModules();

    @GET("api/v1/modules/{id}/bundle")
    Call<ModuleBundleDTO> getModuleBundle(@Path("id") Long moduleId);
}
```

- **`getAllModules()`** — `GET api/vi/modules` — restituisce l'elenco di tutti i moduli.
- **`getModuleBundle(id)`** — `GET api/v1/modules/{id}/bundle` — restituisce un modulo specifico con i suoi vettori.
- Nota: il primo endpoint usa `vi` (possibile typo per `v1`), il secondo `v1`.

---

## 3. Integrazione nella UI — MainActivity

**File:** `ui/MainActivity.java` (righe ~789–823)

```java
private void testCloudConnection() {
    Log.d(TAG, "Avvio del test di connessione al Cloud...");

    RetrofitClient.getApi().getAllModules().enqueue(new Callback<List<ModuleBundleDTO>>() {

        @Override
        public void onResponse(Call<List<ModuleBundleDTO>> call,
                               Response<List<ModuleBundleDTO>> response) {
            if (response.isSuccessful() && response.body() != null) {
                List<ModuleBundleDTO> listaModuli = response.body();
                Log.d(TAG, "CONNESSO! Ricevuti " + listaModuli.size()
                    + " moduli dal database.");
                for (ModuleBundleDTO modulo : listaModuli) {
                    Log.d(TAG, " -> Modulo ID: " + modulo.moduleName
                        + " | Visibilità: " + modulo.visibility);
                }
                Toast.makeText(MainActivity.this,
                    "Connesso a VSHub Cloud!", Toast.LENGTH_SHORT).show();
            } else {
                Log.e(TAG, "Il server ha risposto, ma con un errore. Codice: "
                    + response.code());
            }
        }

        @Override
        public void onFailure(Call<List<ModuleBundleDTO>> call, Throwable t) {
            Log.e(TAG, "ERRORE CRITICO DI CONNESSIONE: " + t.getMessage());
            Toast.makeText(MainActivity.this,
                "Impossibile raggiungere il server Cloud",
                Toast.LENGTH_LONG).show();
        }
    });
}
```

- Chiamata asincrona con **`enqueue`** (non bloccante).
- Eseguita in `onCreate()`: testa la connettività all'avvio dell'app.
- In caso di successo: logga il numero di moduli ricevuti e i loro nomi/visibilità, mostra un Toast.
- In caso di errore: logga il codice HTTP o l'eccezione di rete, mostra un Toast di errore.
- **N.B.:** I moduli scaricati **non** vengono attualmente persistiti o utilizzati localmente — è un test di connettività puro.

---

## 4. Struttura del Motore VSA (`engine`)

### Package `engine.vsa` (3 classi)

| Classe                  | Ruolo |
|-------------------------|-------|
| `VsaVector`             | Ipervettore BSC (Binary Sparse Code) a 10.000 dimensioni, organizzato in 157 blocchi lunghi. Operazioni: binding (XOR), distanza di Hamming, similarità coseno. |
| `RandomProjectionBridge`| Proiezione Rademacher dalle feature CNN (1280-dim) allo spazio VSA (10.000-dim), con normalizzazione L2 e binarizzazione. |
| `DatasetStatTracker`    | Standardizzazione Z-score (media/deviazione standard) caricata da file binario in asset. |

### Package `engine.learning` (1 classe)

| Classe                   | Ruolo |
|--------------------------|-------|
| `LearningSessionManager` | Gestione sessione di apprendimento multi-shot per bundling additivo. Enum interno `State`: `IDLE`, `COLLECTING`, `PROCESSING`, `CANCELLED`. |

---

## 5. Mappa Completa delle Classi del Progetto

| Package           | Classe                      | Descrizione |
|-------------------|-----------------------------|-------------|
| `detector`        | `SsdExtractor`              | Wrapper MediaPipe ObjectDetector per rilevamento SSD |
| `encoder`         | `FeatureEncoder`            | Encoder CNN TFLite (CPU/GPU), estrae feature da crop 224×224 |
| `engine.vsa`      | `VsaVector`                 | Ipervettore BSC con operazioni algebriche VSA |
| `engine.vsa`      | `RandomProjectionBridge`    | Proiezione Rademacher feature → spazio VSA |
| `engine.vsa`      | `DatasetStatTracker`        | Statistiche Z-score per standardizzazione |
| `engine.learning` | `LearningSessionManager`    | Sessione di apprendimento multi-shot |
| `network`         | `RetrofitClient`            | Singleton Retrofit + OkHttp |
| `network`         | `VSHubApi`                  | Interfaccia endpoint REST |
| `network.dto`     | `ModuleBundleDTO`           | DTO per bundle di modulo |
| `network.dto`     | `ClassEntryDTO`             | DTO per classe appresa |
| `ui`              | `MainActivity`              | Activity principale (825 righe): camera, detection, learning, test di rete |
| `ui`              | `TrackingOverlayView`       | View custom per bounding box e label |
| `ui`              | `DetectionStabilizer`       | Stabilizzazione temporale/spaziale (EMA smoothing, majority voting) |
| `utils`           | `AntiGlareFilter`           | Filtro anti-riflesso basato su LUT gamma (non usato) |

---

## 6. Riepilogo Tecnologico

| Componente          | Tecnologia                              |
|---------------------|-----------------------------------------|
| Linguaggio          | Java 11                                 |
| ML on-device        | TensorFlow Lite 2.16.1 (CPU + GPU)      |
| Object Detection    | MediaPipe tasks-vision 0.10.14          |
| Networking          | Retrofit 2.9.0 + OkHttp (transitivo) + Gson |
| Camera              | CameraX 1.3.4                           |
| UI                  | AppCompat, Material Design, CardView    |
| Backend target      | `http://192.168.178.42:8080/` (REST locale) |
| Dimensione VSA      | 10.000 dimensioni (BSC)                 |
