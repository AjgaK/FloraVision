package com.example.floravision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.floravision.ml.PlantRecognitionModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Classifier(private val context: Context, private val labels: List<String>) {

    //Required image size for the TensorFlow Lite model
    private val imageSize = 224

    /**
     * Classifies the input image using a TensorFlow Lite model.
     */
    fun classifyImage(imageBitmap: Bitmap): List<Pair<String, Float>> {
        // Resize the image to the dimensions required by the Tensorflow Lite model
        val resizedBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false)

        // Loading the image
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        // Initializing the model
        val model = PlantRecognitionModel.newInstance(context)
        // Preparing input data
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)
        // Processing the image through the model
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        val labeledResults = labels.zip(outputFeature0.toList())
        model.close()
        return labeledResults.sortedByDescending { it.second }
    }
}