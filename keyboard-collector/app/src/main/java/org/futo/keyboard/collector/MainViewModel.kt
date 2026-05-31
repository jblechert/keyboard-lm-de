package org.futo.keyboard.collector

import android.app.Application
import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import org.futo.keyboard.collector.data.AppDatabase
import org.futo.keyboard.collector.data.CorrectionCase
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val dao = AppDatabase.get(app).correctionDao()

    val cases = dao.allCases().stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        emptyList()
    )

    fun submit(recognized: String, expected: String, context: String) {
        if (recognized.isBlank() || expected.isBlank()) return
        viewModelScope.launch {
            dao.insert(CorrectionCase(
                recognized = recognized.trim(),
                expected = expected.trim(),
                context = context.trim(),
            ))
        }
    }

    fun delete(id: Int) = viewModelScope.launch { dao.delete(id) }

    fun exportJson(context: Context): Intent {
        val array = JSONArray()
        cases.value.forEach { case ->
            array.put(JSONObject().apply {
                put("recognized", case.recognized)
                put("expected", case.expected)
                if (case.context.isNotBlank()) put("context", case.context)
                put("ts", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.ROOT)
                    .format(Date(case.timestamp)))
            })
        }
        val json = JSONObject().put("cases", array).toString(2)

        val file = File(context.cacheDir, "futo_corrections.json")
        file.writeText(json)

        val uri = FileProvider.getUriForFile(
            context, "${context.packageName}.provider", file
        )
        return Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
