package com.antonio.visionsector.utils;

/**
 * Filtro ultra-veloce tramite Lookup Table (LUT)
 * Ottimizzato per ridurre i riflessi accecanti (schiaccia i bianchi)
 * prima di passare l'immagine a SSD.
 */
public class AntiGlareFilter {

    private static final int[] LUT = new int[256];

    // Blocco statico: viene eseguito una sola volta quando la classe viene caricata in memoria.
    static {
        // Prepariamo la tabella di conversione per una "Gamma Correction"
        // Valori di gamma > 1 scuriscono i toni medi e alti (i riflessi)
        double gamma = 1.4;
        for (int i = 0; i < 256; i++) {
            // Calcolo pesante fatto una sola volta!
            LUT[i] = (int) (255.0 * Math.pow(i / 255.0, gamma));
            // Sicurezza: blocchiamo il valore tra 0 e 255
            if (LUT[i] > 255) LUT[i] = 255;
            if (LUT[i] < 0) LUT[i] = 0;
        }
    }

    /**
     * Applica il filtro anti-riflesso direttamente sull'array di pixel.
     * Esecuzione stimata su 300x300: < 1 millisecondo.
     * @param pixels Array di pixel nel formato ARGB_8888. Viene modificato in place.
     */
    public static void applyFilterInPlace(int[] pixels) {
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            // Estrazione veloce dei canali RGB
            int r = (color >> 16) & 0xFF;
            int g = (color >> 8) & 0xFF;
            int b = color & 0xFF;

            // Mappatura istantanea tramite LUT
            r = LUT[r];
            g = LUT[g];
            b = LUT[b];

            // Ricomposizione del pixel (mantenendo il canale Alpha originale opaco)
            pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }
}