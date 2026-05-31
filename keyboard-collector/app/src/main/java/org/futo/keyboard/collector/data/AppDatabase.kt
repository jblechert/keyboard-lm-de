package org.futo.keyboard.collector.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [CorrectionCase::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun correctionDao(): CorrectionDao

    companion object {
        @Volatile private var instance: AppDatabase? = null

        fun get(context: Context): AppDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext,
                AppDatabase::class.java,
                "corrections.db"
            ).build().also { instance = it }
        }
    }
}
