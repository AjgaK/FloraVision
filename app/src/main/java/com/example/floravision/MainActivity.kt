package com.example.floravision.ui

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.floravision.R
import com.example.floravision.classifier.FlowerClassifier
import com.example.floravision.classifier.TfliteFlowerClassifier
import com.example.floravision.databinding.ActivityMainBinding
import com.example.floravision.image.BitmapLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: FlowerClassifier
    private lateinit var bitmapLoader: BitmapLoader

    private lateinit var takePhotoLauncher: ActivityResultLauncher<Void?>
    private lateinit var pickImageLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupDependencies()
        setupWelcomeMessage()
        setupActivityResultLaunchers()
        setupClickListeners()
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }

    private fun setupDependencies() {
        val labels = assets.open("labels.txt").bufferedReader().use {
            it.readLines()
        }

        classifier = TfliteFlowerClassifier(this, labels)
        bitmapLoader = BitmapLoader(this)
    }

    private fun setupWelcomeMessage() {
        binding.welcome.text = WelcomeTextFormatter.format(
            getString(R.string.welcome_message)
        )
    }

    private fun setupActivityResultLaunchers() {
        takePhotoLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicturePreview()
        ) { bitmap ->
            if (bitmap != null) { handleSelectedImage(bitmap)
            }
        }

        pickImageLauncher = registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->
            if(uri != null) {
                val bitmap = bitmapLoader.loadFromUri(uri)

                if (bitmap != null) {
                    handleSelectedImage(bitmap)
                }
            }
        }
    }

    private fun setupClickListeners() {
        binding.selectBtn.setOnClickListener {
            selectImage()
        }

        binding.newPictureBtn.setOnClickListener {
            captureImage()
        }
    }

    private fun handleSelectedImage(bitmap: Bitmap) {
        binding.imageView.setImageBitmap(bitmap)
        binding.welcome.visibility = View.GONE
        binding.imageView.strokeWidth = resources.getDimension(R.dimen.stroke_width)

        classifySelectedImage(bitmap)
    }

    private fun captureImage() {
        takePhotoLauncher.launch(null)
    }

    private fun selectImage() {
        pickImageLauncher.launch("image/*")
    }

    private fun classifySelectedImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            val results = withContext(Dispatchers.Default) {
                classifier.classify(bitmap)
            }

            binding.result.text = ResultFormatter.format(results)
        }
    }
}