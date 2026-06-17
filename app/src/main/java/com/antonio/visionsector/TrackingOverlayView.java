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

    public static class DetectedItem {
        public final String name;
        public final int pixelLeft, pixelTop, pixelRight, pixelBottom;
        public final double confidence;

        public DetectedItem(String name, int pixelLeft, int pixelTop, int pixelRight, int pixelBottom, double confidence) {
            this.name = name;
            this.pixelLeft = pixelLeft;
            this.pixelTop = pixelTop;
            this.pixelRight = pixelRight;
            this.pixelBottom = pixelBottom;
            this.confidence = confidence;
        }
    }

    private List<DetectedItem> detectedItems = new ArrayList<>();
    private int frameW, frameH;

    private final Paint boxPaint;
    private final Paint textPaint;
    private final Paint textBgPaint;
    private final Paint viewfinderPaint;
    private final Paint dimPaint;

    public TrackingOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

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

        viewfinderPaint = new Paint();
        viewfinderPaint.setColor(Color.parseColor("#FFD700"));
        viewfinderPaint.setStyle(Paint.Style.STROKE);
        viewfinderPaint.setStrokeWidth(4f);

        dimPaint = new Paint();
        dimPaint.setColor(Color.parseColor("#88000000"));
        dimPaint.setStyle(Paint.Style.FILL);
    }

    public void updateDetections(List<DetectedItem> items, int frameW, int frameH) {
        this.detectedItems = (items != null) ? items : new ArrayList<>();
        this.frameW = frameW;
        this.frameH = frameH;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (frameW <= 0 || frameH <= 0) return;

        int w = getWidth();
        int h = getHeight();

        float scale = Math.min((float) w / frameW, (float) h / frameH);
        int drawW = (int) (frameW * scale);
        int drawH = (int) (frameH * scale);
        int offsetX = (w - drawW) / 2;
        int offsetY = (h - drawH) / 2;

        canvas.drawRect(0, 0, w, offsetY, dimPaint);
        canvas.drawRect(0, offsetY + drawH, w, h, dimPaint);
        if (offsetX > 0) {
            canvas.drawRect(0, offsetY, offsetX, offsetY + drawH, dimPaint);
            canvas.drawRect(offsetX + drawW, offsetY, w, offsetY + drawH, dimPaint);
        }

        canvas.drawRect(offsetX, offsetY, offsetX + drawW, offsetY + drawH, viewfinderPaint);

        if (detectedItems.isEmpty()) return;

        for (DetectedItem item : detectedItems) {
            float left = offsetX + item.pixelLeft * scale;
            float top = offsetY + item.pixelTop * scale;
            float right = offsetX + item.pixelRight * scale;
            float bottom = offsetY + item.pixelBottom * scale;

            canvas.drawRect(left, top, right, bottom, boxPaint);

            String label = item.name + " (" + String.format("%.2f", item.confidence) + ")";
            float textW = textPaint.measureText(label);
            canvas.drawRect(left, top - 50, left + textW + 20, top, textBgPaint);
            canvas.drawText(label, left + 10, top - 12, textPaint);
        }
    }
}
