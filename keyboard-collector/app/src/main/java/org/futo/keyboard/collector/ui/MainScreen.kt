package org.futo.keyboard.collector.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import org.futo.keyboard.collector.MainViewModel
import org.futo.keyboard.collector.data.CorrectionCase
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(vm: MainViewModel) {
    val cases by vm.cases.collectAsState()
    val context = LocalContext.current

    var recognized by remember { mutableStateOf("") }
    var expected   by remember { mutableStateOf("") }
    var ctxText    by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("FUTO Collector  (${cases.size})") },
                actions = {
                    IconButton(onClick = {
                        context.startActivity(
                            android.content.Intent.createChooser(
                                vm.exportJson(context), "Export JSON"
                            )
                        )
                    }) {
                        Icon(Icons.Default.Share, contentDescription = "Export")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // ── Eingabeformular ───────────────────────────────────────────────
            OutlinedTextField(
                value = recognized,
                onValueChange = { recognized = it },
                label = { Text("Erkannt (Fehler)") },
                placeholder = { Text("Du bist schön ein guter") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = expected,
                onValueChange = { expected = it },
                label = { Text("Erwartet (korrekt)") },
                placeholder = { Text("Du bist schon ein guter") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = ctxText,
                onValueChange = { ctxText = it },
                label = { Text("Kontext (optional)") },
                placeholder = { Text("z.B. WhatsApp, Nachricht an Freund") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            Button(
                onClick = {
                    vm.submit(recognized, expected, ctxText)
                    recognized = ""; expected = ""; ctxText = ""
                },
                modifier = Modifier.align(Alignment.End),
                enabled = recognized.isNotBlank() && expected.isNotBlank(),
            ) {
                Text("Hinzufügen")
            }

            Text(
                "Export JSON → keyboard-lm-de@blechert.at",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )

            HorizontalDivider()

            // ── Liste ─────────────────────────────────────────────────────────
            if (cases.isEmpty()) {
                Box(Modifier.fillMaxWidth().padding(32.dp), Alignment.Center) {
                    Text("Noch keine Einträge.", style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(cases, key = { it.id }) { case ->
                        CaseCard(case, onDelete = { vm.delete(case.id) })
                    }
                }
            }
        }
    }
}

@Composable
private fun CaseCard(case: CorrectionCase, onDelete: () -> Unit) {
    val date = remember(case.timestamp) {
        SimpleDateFormat("dd.MM. HH:mm", Locale.GERMAN).format(Date(case.timestamp))
    }
    Card(Modifier.fillMaxWidth()) {
        Row(
            Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("✗ ${case.recognized}", style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.error)
                Text("✓ ${case.expected}", style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.primary)
                if (case.context.isNotBlank()) {
                    Text(case.context, style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text(date, style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Löschen",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}
