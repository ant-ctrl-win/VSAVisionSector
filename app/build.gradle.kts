plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.antonio.visionsector"
    compileSdk = 34 // <-- SINTASSI CORRETTA

    androidResources {
        noCompress.add("tflite")
    }

    defaultConfig {
        applicationId = "com.antonio.visionsector" // <-- ALLINEATO AL NAMESPACE
        minSdk = 24
        targetSdk = 34 // <-- ABBASSATO A 34 (36 causerebbe errori)
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.activity.ktx)
    implementation(libs.appcompat)
    implementation(libs.constraintlayout)
    implementation(libs.litert.gpu.api)
    // Rimosso libs.litert.api che causava l'errore
    implementation(libs.material)

    // Componenti di Design Material
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.cardview:cardview:1.0.0")

    // ── TensorFlow Lite stack (CPU + GPU) ──────────────────────────────
    //  · tensorflow-lite:2.16.1 fornisce org.tensorflow.lite.* (Interpreter, Delegate, etc.)
    //  · tensorflow-lite-gpu:2.16.1 fornisce GpuDelegate + librerie native OpenCL/GLES
    //  · Escludiamo tensorflow-lite-api dal GPU delegate perché le classi API
    //    sono già dentro tensorflow-lite (evita duplicate class).
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1") {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
    }

    // ── MediaPipe Object Detector ──────────────────────────────────────
    //  · tasks-vision:0.10.14 importa internamente com.google.ai.edge.litert:litert-api
    //    che ridefinisce org.tensorflow.lite.* creando conflitto con tensorflow-lite.
    //  · Escludiamo litert-api: MediaPipe userà le classi TFLite classiche.
    implementation("com.google.mediapipe:tasks-vision:0.10.14") {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }

    // ── CameraX ────────────────────────────────────────────────────────
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    testImplementation(libs.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.ext.junit)
}