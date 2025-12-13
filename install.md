# Installation

Diese Anleitung beschreibt, wie du die BG-Soft-Automatisierung auf einem neuen
Rechner startklar machst.

## Voraussetzungen

- OBS Studio inkl. WebSocket-Server (Tools → WebSocket Server Settings) ist
  installiert und konfiguriert.
- Miniconda/Anaconda ist verfügbar (`conda --version`).
- Dieses Repository ist bereits lokal geklont (`git clone ...`).

## Schnelleinrichtung mit `install.sh`

```bash
cd /pfad/zu/_AA_BG_Soft
./install.sh
```

Was das Script erledigt:
1. Liest `environment.yml` und erzeugt/aktualisiert das Conda-Environment
   `BG-Soft`.
2. Installiert die Python-Abhängigkeiten aus `requirements.txt` (z. B.
   `obsws-python`, `PyQt5`).
3. Gibt dir den Befehl aus, um das Environment zu aktivieren.

## Manuelle Einrichtung (Alternative)

Falls du die Schritte selbst ausführen möchtest:

```bash
cd /pfad/zu/_AA_BG_Soft
conda env create -f environment.yml    # oder: conda env update -f environment.yml --prune
conda activate BG-Soft
pip install -r requirements.txt
```

## Verwendung

```bash
conda activate BG-Soft
python render_with_obs.py /pfad/zur/datei.mp4
# oder die GUI starten:
python bg_soft_gui.py
```

OBS muss laufen und die Szene/Quelle (Standard: `BR-Render` + `BR-Clip`) bereits
mit deinem Background-Removal-Filter konfiguriert sein. Weitere Details findest
du in `README.md`.
