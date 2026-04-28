package com.example.floravision.classifier

import android.content.Context
import android.graphics.Bitmap
import com.example.floravision.ml.PlantRecognitionModel
import com.example.floravision.model.ClassificationResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class TfliteFlowerClassifier(
    context: Context,
    private val labels: List<String>
) : FlowerClassifier {

    private val imageSize = 224
    private val model = PlantRecognitionModel.newInstance(context)

    override fun classify(imageBitmap: Bitmap): List<ClassificationResult> {
        val resizedBitmap = Bitmap.createScaledBitmap(
            imageBitmap,
            imageSize,
            imageSize,
            false
        )

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)

        val inputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, imageSize, imageSize, 3),
            DataType.FLOAT32
        )

        inputBuffer.loadBuffer(tensorImage.buffer)

        val output = model.process(inputBuffer)
            .outputFeature0AsTensorBuffer
            .floatArray

        if (labels.size != output.size) {
            return emptyList()
        }

        return labels
            .zip(output.toList())
            .map { (label, confidence) ->
                ClassificationResult(label, confidence)
            }
            .sortedByDescending { it.confidence }
    }

    override fun close() {
        model.close()
    }
}