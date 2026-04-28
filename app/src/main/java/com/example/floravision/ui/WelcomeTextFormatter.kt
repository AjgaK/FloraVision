package com.example.floravision.ui

import android.graphics.Typeface
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan

object WelcomeTextFormatter {

    fun format(welcomeText: String): CharSequence {
        val builder = SpannableStringBuilder(welcomeText)

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

                builder.setSpan(
                    StyleSpan(Typeface.BOLD),
                    startIndex,
                    endIndex,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )

                builder.setSpan(
                    RelativeSizeSpan(1.2f),
                    startIndex,
                    endIndex,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        return builder
    }
}