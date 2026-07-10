package com.antonio.visionsector.utils;

import android.content.Context;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.antonio.visionsector.network.dto.ModuleBundleDTO;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Set;

public class LocalStorageManager {

    private static final String CATALOG_FILE = "catalog.json";
    private static final Gson gson = new Gson();

    // 1. Legge quali moduli sono già stati scaricati sul dispositivo
    public static Set<Long> getDownloadedModuleIds(Context context) {
        File file = new File(context.getFilesDir(), CATALOG_FILE);
        if (!file.exists()) {
            return new HashSet<>();
        }
        try (FileInputStream fis = context.openFileInput(CATALOG_FILE);
             InputStreamReader isr = new InputStreamReader(fis)) {
            Type setType = new TypeToken<HashSet<Long>>() {}.getType();
            Set<Long> ids = gson.fromJson(isr, setType);
            return ids != null ? ids : new HashSet<>();
        } catch (Exception e) {
            e.printStackTrace();
            return new HashSet<>();
        }
    }

    // 2. Aggiunge un ID modulo al catalogo locale
    public static void registerModuleInCatalog(Context context, Long moduleId) {
        Set<Long> downloadedIds = getDownloadedModuleIds(context);
        downloadedIds.add(moduleId);
        try (FileOutputStream fos = context.openFileOutput(CATALOG_FILE, Context.MODE_PRIVATE);
             OutputStreamWriter osw = new OutputStreamWriter(fos)) {
            gson.toJson(downloadedIds, osw);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 3. Ritorna la cartella specifica di un modulo (es. /data/user/0/pacchetto/files/modules/1)
    public static File getModuleDirectory(Context context, Long moduleId) {
        File baseDir = new File(context.getFilesDir(), "modules");
        File moduleDir = new File(baseDir, String.valueOf(moduleId));
        if (!moduleDir.exists()) {
            moduleDir.mkdirs();
        }
        return moduleDir;
    }

    // 4. Salva il DTO completo del modulo (vettori inclusi) come JSON locale
    public static boolean saveModuleBundle(Context context, ModuleBundleDTO bundle) {
        try {
            File moduleDir = getModuleDirectory(context, bundle.getModuleId());
            File bundleFile = new File(moduleDir, "bundle.json");

            try (FileOutputStream fos = new FileOutputStream(bundleFile);
                 OutputStreamWriter osw = new OutputStreamWriter(fos)) {
                gson.toJson(bundle, osw);
            }

            // Registriamo il successo nel catalogo globale dell'app
            registerModuleInCatalog(context, bundle.getModuleId());
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // 5. Carica il modulo dal disco quando l'utente decide di usarlo offline o attivarlo
    public static ModuleBundleDTO loadLocalModuleBundle(Context context, Long moduleId) {
        try {
            File moduleDir = getModuleDirectory(context, moduleId);
            File bundleFile = new File(moduleDir, "bundle.json");
            if (!bundleFile.exists()) return null;

            try (FileInputStream fis = new FileInputStream(bundleFile);
                 InputStreamReader isr = new InputStreamReader(fis)) {
                return gson.fromJson(isr, ModuleBundleDTO.class);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}