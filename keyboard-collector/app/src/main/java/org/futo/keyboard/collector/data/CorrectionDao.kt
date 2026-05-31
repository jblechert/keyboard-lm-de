package org.futo.keyboard.collector.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface CorrectionDao {
    @Insert
    suspend fun insert(entry: CorrectionCase)

    @Query("SELECT * FROM corrections ORDER BY timestamp DESC")
    fun allCases(): Flow<List<CorrectionCase>>

    @Query("DELETE FROM corrections WHERE id = :id")
    suspend fun delete(id: Int)

    @Query("SELECT COUNT(*) FROM corrections")
    suspend fun count(): Int
}
