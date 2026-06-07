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
    implementation(libs.material)

    // Componenti di Design Material
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.cardview:cardview:1.0.0")

    // Librerie Google CameraX
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // Il nostro motore offline
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.ext.junit)
}