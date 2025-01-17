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


class MainActivity : AppCompatActivity() {

    // UI Components
    private lateinit var camera : Button
    private lateinit var gallery : Button
    private lateinit var imageView: ImageView
    private lateinit var result: TextView
    private lateinit var welcome: TextView

    // Image classifier
    private lateinit var classifier: Classifier

    // Bitmap of the image to classify
    private lateinit var imageBitmap : Bitmap

    // Activity result launchers
    private lateinit var takePhotoForResult: ActivityResultLauncher<Intent>
    private lateinit var pickImageFromGalleryForResult: ActivityResultLauncher<Intent>

    //List of classification labels
    private lateinit var labels : List<String>

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
        // Create classifier object
        classifier = Classifier(this, labels)

        // Register a launcher to handle camera results
        takePhotoForResult = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result: ActivityResult ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                imageBitmap = data?.extras?.get("data") as Bitmap
                imageView.setImageBitmap(imageBitmap)
                welcome.text = ""
                // Add a lighter border to an image after it was added
                updateBorder()
                formatResults()
            }
        }

        pickImageFromGalleryForResult = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result: ActivityResult ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                val selectedPhotoUri = data?.data
                imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedPhotoUri)
                welcome.text = ""
                imageView.setImageBitmap(imageBitmap)
                welcome.text = ""
                // Add a lighter border to an image after it was added
                updateBorder()
                formatResults()
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
        val welcomeText = getString(R.string.welcome_message)

        val spannableBuilder = SpannableStringBuilder(welcomeText)
        val boldParts = listOf(
            "Welcome to FloraVision!",
            "1. Upload or capture a photo of a flower",
            "2. Let the app work its magic",
            "3. View the results",
            "Enjoy exploring the world of flowers with FloraVision!"
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
     * Obtains the classification results from the Classifier
     * Displays the top-3 classification results with their confidence scores.
     * The highest confidence result is displayed with larger, bold font.
     */
    private fun formatResults() {
        // Get top 3 results with confidence score > 0.00%
        val topResults = classifier.classifyImage(imageBitmap).filter { confidence ->
            val formattedConfidence = "%.2f".format(confidence.second * 100).toDouble()
            formattedConfidence > 0.00 }.take(3)

        if (topResults.isEmpty()) {
            result.text = getString(R.string.no_predictions)
            return
        }

        val spannableBuilder = SpannableStringBuilder()

        // Combine the confidence scores with labels and format the first result to be bigger
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
    }
}