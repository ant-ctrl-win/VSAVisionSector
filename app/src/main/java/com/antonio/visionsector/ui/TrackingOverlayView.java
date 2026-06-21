package com.antonio.visionsector.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;

public class TrackingOverlayView extends View {

    private List<DetectedItem> items;
    private Paint boxPaint;
    private Paint textPaint;

    public TrackingOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initPaints();
    }

    private void initPaints() {
        // Stile del rettangolo di tracciamento
        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN); // Colore del box (puoi cambiarlo)
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8.0f);
        boxPaint.setAntiAlias(true);

        // Stile del testo (Etichetta e Confidenza)
        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(55.0f);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setAntiAlias(true);
        textPaint.setShadowLayer(6.0f, 0f, 0f, Color.BLACK); // Ombra nera per leggibilità
    }

    /**
     * Aggiorna gli oggetti rilevati.
     * I parametri w e h li manteniamo per retrocompatibilità con la firma di MainActivity,
     * ma ora usiamo direttamente le dimensioni reali della View nel momento del disegno.
     */
    public void updateDetections(List<DetectedItem> newItems, int w, int h) {
        this.items = newItems;
        invalidate(); // Forza Android a richiamare onDraw() a 60 FPS
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (items == null || items.isEmpty()) return;

        for (DetectedItem item : items) {
            // Le coordinate ora sono pixel esatti pronti per lo schermo
            RectF rect = new RectF(item.left, item.top, item.right, item.bottom);

            // Disegna il box verde
            canvas.drawRect(rect, boxPaint);

            // Disegna l'etichetta
            String labelText = item.label + " " + String.format("%.2f", item.confidence);
            canvas.drawText(labelText, item.left, item.top - 15, textPaint);
        }
    }

    /**
     * Classe DTO (Data Transfer Object) per passare i dati dalla MainActivity all'Overlay.
     * I campi left, top, right e bottom ORA SONO FLOAT!
     */
    public static class DetectedItem {
        public final String label;
        public final float left, top, right, bottom; // <--- Ecco i float che mancavano!
        public final double confidence;

        public DetectedItem(String label, float left, float top, float right, float bottom, double confidence) {
            this.label = label;
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.confidence = confidence;
        }
    }
}