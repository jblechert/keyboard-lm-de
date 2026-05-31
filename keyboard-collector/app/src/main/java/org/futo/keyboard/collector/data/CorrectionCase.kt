package org.futo.keyboard.collector.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "corrections")
data class CorrectionCase(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val recognized: String,   // was die Tastatur ausgegeben hat
    val expected: String,     // was der User wollte
    val context: String = "", // optionaler Satzkontext
    val timestamp: Long = System.currentTimeMillis(),
)
