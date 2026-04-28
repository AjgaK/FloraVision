package com.example.floravision.classifier

import android.graphics.Bitmap
import com.example.floravision.model.ClassificationResult

interface FlowerClassifier {
    fun classify(imageBitmap: Bitmap): List<ClassificationResult>
    fun close()
}