package com.example.floravision

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.ActivityResult
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.floravision.ml.PlantRecognitionModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {

    // UI Components
    private lateinit var camera : Button
    private lateinit var gallery : Button
    private lateinit var imageView: ImageView
    private lateinit var result: TextView
    private lateinit var welcome: TextView

    // Bitmap of the image to classify
    private lateinit var imageBitmap : Bitmap

    // Activity result launchers
    private lateinit var takePhotoForResult: ActivityResultLauncher<Intent>
    private lateinit var pickImageFromGalleryForResult: ActivityResultLauncher<Intent>

    //List of classification labels
    private lateinit var labels : List<String>

    //Required image size for the TensorFlow Lite model
    private val imageSize = 224

    /**
     * Called when the activity is created.
     * Initializes the UI components and registers activity result handlers for camera and gallery.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        camera = findViewById(R.id.newPictureBtn)
        gallery = findViewById(R.id.selectBtn)
        result = findViewById(R.id.result)
        welcome = findViewById(R.id.welcome)
        imageView = findViewById(R.id.imageView)
        formatWelcomeMessage()
        // Load classification labels from the "labels.txt" file in the assets folder
        labels = application.assets.open("labels.txt").bufferedReader().readLines()

        // Register a launcher to handle camera results
        takePhotoForResult = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result: ActivityResult ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                imageBitmap = data?.extras!!.get("data") as Bitmap
                imageView.setImageBitmap(imageBitmap)
                welcome.text = ""
                // Resize the image to the dimensions required by the Tensorflow Lite model
                imageBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false)
                // Add a lighter border to an image after it was added
                updateBorder()
                classifyImage()
            }
        }

        pickImageFromGalleryForResult = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result: ActivityResult ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                val selectedPhotoUri = data?.data
                imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedPhotoUri)
                imageView.setImageBitmap(imageBitmap)
                welcome.text = ""
                // Resize the image to the dimensions required by the Tensorflow Lite model
                imageBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false)
                // Add a lighter border to an image after it was added
                updateBorder()
                classifyImage()
            }
        }

        gallery.setOnClickListener {
            selectImage()
        }

        camera.setOnClickListener {
            captureImage()
        }
    }

    /**
     * Checks for camera permissions.
     * Launches the camera application to capture an image.
     */
    private fun captureImage() {
        if(checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            takePhotoForResult.launch(cameraIntent)
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        }
    }

    /**
     * Opens the gallery selector.
     */
    private fun selectImage() {
        val galleryIntent = Intent(Intent.ACTION_GET_CONTENT)
        galleryIntent.setType("image/*")
        pickImageFromGalleryForResult.launch(galleryIntent)
    }

    /**
     * Updates the border width (strokeWidth) for the ShapeableImageView.
     */
    private fun updateBorder() {
        val shapeableImageView = findViewById<com.google.android.material.imageview.ShapeableImageView>(R.id.imageView)
        shapeableImageView.strokeWidth = resources.getDimension(R.dimen.stroke_width)
    }

    /**
     * Formats and displays the welcome message in the TextView with specific parts bolded and enlarged.
     */
    private fun formatWelcomeMessage(){
        val welcomeText = """
        Welcome to FloraVision! 🌸
        FloraVision makes it easy to identify flower species in just a few steps. Here's a quick guide to get started:
        
        1. Upload or capture a photo of a flower
        Use one of the buttons to upload a picture from your gallery or take a new one with your camera. Make sure the flower is clearly visible for the best results!
        2. Let the app work its magic
        Once the image is uploaded, FloraVision will analyse it and display predictions about the flower's species.
        3. View the results
        Check out the top predictions and learn more about the flowers around you!
        
        Enjoy exploring the world of flowers with FloraVision! 🌺
    """.trimIndent()

        val spannableBuilder = SpannableStringBuilder(welcomeText)
        val boldParts = listOf(
            "Welcome to FloraVision! 🌸",
            "1. Upload or capture a photo of a flower",
            "2. Let the app work its magic",
            "3. View the results",
            "Enjoy exploring the world of flowers with FloraVision! 🌺"
        )

        for (part in boldParts) {
            val startIndex = welcomeText.indexOf(part)
            if (startIndex != -1) {
                val endIndex = startIndex + part.length
                spannableBuilder.setSpan(
                    android.text.style.StyleSpan(android.graphics.Typeface.BOLD),
                    startIndex,
                    endIndex,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                spannableBuilder.setSpan(
                    android.text.style.RelativeSizeSpan(1.2f),
                    startIndex,
                    endIndex,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        welcome.text = spannableBuilder
    }

    /**
     * Classifies the input image using a TensorFlow Lite model.
     * Displays the top-3 classification results with their confidence scores.
     * The highest confidence result is displayed with larger, bold font.
     */
    private fun classifyImage() {
        // Loading the image
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(imageBitmap)
        // Initializing the model
        val model = PlantRecognitionModel.newInstance(this)
        // Preparing input data
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)
        // Processing the image through the model
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        // Formatting the result
        val labeledResults = labels.zip(outputFeature0.toList())
        val topResults = labeledResults.sortedByDescending { it.second }.take(3)

        val spannableBuilder = SpannableStringBuilder()

        topResults.forEachIndexed { index, (label, confidence) ->
            val confidenceText = "${"%.2f".format(confidence * 100)}%"
            val resultText = "$label: $confidenceText\n"

            if (index == 0) {
                val start = spannableBuilder.length
                spannableBuilder.append(resultText)
                spannableBuilder.setSpan(
                    android.text.style.StyleSpan(android.graphics.Typeface.BOLD),
                    start, spannableBuilder.length, 0
                )
                spannableBuilder.setSpan(
                    android.text.style.RelativeSizeSpan(1.2f),
                    start, spannableBuilder.length, 0
                )
            } else {
                spannableBuilder.append(resultText)
            }
        }

        result.text = spannableBuilder

        model.close()
    }
}