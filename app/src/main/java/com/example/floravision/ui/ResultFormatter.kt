package com.example.floravision.ui

import android.graphics.Typeface
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import com.example.floravision.model.ClassificationResult

object ResultFormatter {

    fun format(results: List<ClassificationResult>): CharSequence {
        val topResults = results
            .filter { it.confidence > 0f }
            .take(3)

        val builder = SpannableStringBuilder()

        topResults.forEachIndexed { index, result ->
            val confidenceText = "%.2f".format(result.confidence * 100)
            val resultText = "${result.label}: $confidenceText%\n"

            val start = builder.length
            builder.append(resultText)

            if (index == 0) {
                builder.setSpan(
                    StyleSpan(Typeface.BOLD),
                    start,
                    builder.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                builder.setSpan(
                    RelativeSizeSpan(1.2f),
                    start,
                    builder.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        return builder
    }
}