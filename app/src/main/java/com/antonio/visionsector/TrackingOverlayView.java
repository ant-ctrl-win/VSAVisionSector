package com.antonio.visionsector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

public class TrackingOverlayView extends View {

    // La nuova struttura supporta rettangoli che coprono più celle!
    public static class DetectedItem {
        String name;
        int minCol, minRow, maxCol, maxRow;
        double confidence;

        public DetectedItem(String name, int minCol, int minRow, int maxCol, int maxRow, double confidence) {
            this.name = name;
            this.minCol = minCol;
            this.minRow = minRow;
            this.maxCol = maxCol;
            this.maxRow = maxRow;
            this.confidence = confidence;
        }
    }

    private List<DetectedItem> detectedItems = new ArrayList<>();
    private Paint boxPaint, textPaint, textBgPaint, gridPaint, viewfinderPaint, dimPaint;

    public TrackingOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint = new Paint();
        boxPaint.setColor(Color.parseColor("#00FF00"));
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8f);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(36f);
        textPaint.setFakeBoldText(true);

        textBgPaint = new Paint();
        textBgPaint.setColor(Color.parseColor("#CC000000"));
        textBgPaint.setStyle(Paint.Style.FILL);

        gridPaint = new Paint();
        gridPaint.setColor(Color.parseColor("#33FFFFFF"));
        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setStrokeWidth(2f);

        viewfinderPaint = new Paint();
        viewfinderPaint.setColor(Color.parseColor("#FFD700"));
        viewfinderPaint.setStyle(Paint.Style.STROKE);
        viewfinderPaint.setStrokeWidth(6f);

        // Oscura le parti di schermo fuori dal quadrato
        dimPaint = new Paint();
        dimPaint.setColor(Color.parseColor("#88000000"));
        dimPaint.setStyle(Paint.Style.FILL);
    }

    public void updateDetections(List<DetectedItem> items) {
        this.detectedItems = items;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int w = getWidth();
        int h = getHeight();

        // CALCOLO DEL QUADRATO PERFETTO CENTRALE
        int squareSize = Math.min(w, h);
        int offsetX = (w - squareSize) / 2;
        int offsetY = (h - squareSize) / 2;
        float cellW = (float) squareSize / 7;
        float cellH = (float) squareSize / 7;

        // Disegno bande nere (Sopra/Sotto o Destra/Sinistra) per far capire che l'IA guarda solo il centro
        if (offsetY > 0) {
            canvas.drawRect(0, 0, w, offsetY, dimPaint); // Top
            canvas.drawRect(0, offsetY + squareSize, w, h, dimPaint); // Bottom
        } else if (offsetX > 0) {
            canvas.drawRect(0, 0, offsetX, h, dimPaint); // Left
            canvas.drawRect(offsetX + squareSize, 0, w, h, dimPaint); // Right
        }

        // DISEGNO DELLA GRIGLIA 7x7 NEL QUADRATO
        for (int i = 0; i <= 7; i++) {
            canvas.drawLine(offsetX + i * cellW, offsetY, offsetX + i * cellW, offsetY + squareSize, gridPaint);
            canvas.drawLine(offsetX, offsetY + i * cellH, offsetX + squareSize, offsetY + i * cellH, gridPaint);
        }

        // MIRINO (Evidenzia l'area centrale 3x3)
        canvas.drawRect(offsetX + 2*cellW, offsetY + 2*cellH, offsetX + 5*cellW, offsetY + 5*cellH, viewfinderPaint);
        // Puntino centrale esatto (cella 3,3)
        canvas.drawCircle(offsetX + 3.5f*cellW, offsetY + 3.5f*cellH, 8f, viewfinderPaint);

        // DISEGNO DEI MACRO-RETTANGOLI
        if (detectedItems == null || detectedItems.isEmpty()) return;

        for (DetectedItem item : detectedItems) {
            float left = offsetX + item.minCol * cellW;
            float top = offsetY + item.minRow * cellH;
            float right = offsetX + (item.maxCol + 1) * cellW;
            float bottom = offsetY + (item.maxRow + 1) * cellH;

            canvas.drawRect(left, top, right, bottom, boxPaint);

            String label = item.name + " (" + String.format("%.2f", item.confidence) + ")";
            float textWidth = textPaint.measureText(label);
            canvas.drawRect(left, top - 50, left + textWidth + 20, top, textBgPaint);
            canvas.drawText(label, left + 10, top - 12, textPaint);
        }
    }
}