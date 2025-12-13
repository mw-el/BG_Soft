# Automatisierung BG-Soften mit OBS

Dieses Mini-Tool verbindet sich per OBS-WebSocket, trägt die gewünschte Datei in
eine vorbereitete Media-Source ein, startet die Aufnahme und stoppt sie
automatisch, sobald der Clip durchgelaufen ist.

## 1. OBS vorbereiten (einmalig)

1. **WebSocket aktivieren**
   - OBS → `Tools → WebSocket Server Settings`
   - Server aktivieren, Port (Standard `4455`) merken und ein Passwort setzen
2. **Szene aufsetzen**
   - Lege z. B. eine Szene `BR-Render` an
   - Füge eine Media-Source `BR-Clip` hinzu
   - Weise irgendeine Testdatei zu und aktiviere `Restart playback when source becomes active`
   - Richte auf der Quelle deinen Background-Removal-Filter ein

Während der Automatisierung muss OBS laufen und darf selbst nichts anderes
aufnehmen oder streamen.

## 2. Python-Abhängigkeiten

```bash
pip install -r requirements.txt
```

(Standardmäßig verbindet sich das Skript zu `localhost:4455` mit dem Passwort
`obsstudio`. Passe dies im Aufruf an, wenn du etwas anderes verwendest.)

## 3. Skript verwenden

```bash
python render_with_obs.py /pfad/zur/datei.mp4
```

Optionale Flags:

- `--host/--port/--password` – Verbindung zur OBS-WebSocket-Instanz
- `--scene` – Szenenname (Default `BR-Render`)
- `--input` – Name der Media-Source in dieser Szene (Default `BR-Clip`)
- `--poll` – Intervall in Sekunden für Statusabfragen (Default `0.5`)

## 4. PyQt5-GUI

Für eine komfortable Bedienung steht `bg_soft_gui.py` bereit:

```bash
python bg_soft_gui.py
```

Die GUI ermöglicht:

- Bearbeiten der OBS-Verbindung (Host, Port, Szene, Media-Source, Filter-Namen)
- Einstellen aller relevanten Background-Removal-Parameter sowie der Sharpen-Stärke
- Auswahl mehrerer Videodateien und Batch-Verarbeitung; Statusanzeige für jedes File
- Automatisches Öffnen der erzeugten Ausgabe im Standard-Videoplayer

OBS muss wie beim CLI-Skript vorab laufen und die Szene `BR-Render` mit den Filtern
enthalten.

## 5. Ergebnisdatei

OBS speichert die Aufnahme wie gewohnt (z. B. in deinem Standardaufnahmeordner).
Das Skript wartet auf den fertig gerenderten Clip, verschiebt ihn anschließend
zum Eingabeverzeichnis und benennt ihn nach dem Schema:

```
<original_stem>_soft_<YYYYmmdd-HHMMSS><obs_extension>
```

Beispiel: `video.mp4 → video_soft_20231212-150501.mkv`

## 6. Troubleshooting

- Wenn OBS noch aufnimmt, bricht das Skript ab, damit nichts überschrieben wird.
- Bei `OBS_MEDIA_STATE_ERROR` beendet das Skript ebenfalls mit Fehlermeldung.
- Stelle sicher, dass die Szene + Media-Source exakt so heißen wie im Skript
  bzw. wie du sie per Flags übergibst.
