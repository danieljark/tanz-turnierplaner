# Tanzturnierplane

## Lokale Entwicklung starten

1. Virtuelle Umgebung anlegen und Abhängigkeiten installieren:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Flask-App starten (wählt automatisch `templates/` und `static/` aus dem Projekt):

   ```bash
   flask --app app run
   # oder
   python main.py  # startet Server und öffnet automatisch den Browser
   ```

3. Im Browser `http://localhost:5000` (oder den im Terminal angezeigten Port) öffnen.

### Funktionsumfang

- **Einstellungen**: Hinterlege Benutzername, Passwort, Accept-Header sowie den kompletten `User-Agent` (z. B. `TPS.net/1.0; Token=abcdef`). Optional kannst du einen Standardfilter für Veranstalter/Ausrichter speichern. Alles landet lokal in `settings.json` (Klartext!)
- **Veranstaltungen**: Lädt `/api/v1/veranstaltungen` mit deinem gespeicherten User-Agent (aus Name/Version & Token). Wird kein Filter angegeben, nutzt die Ansicht automatisch den Standardfilter aus den Einstellungen – lässt du das Filterfeld leer, werden alle Veranstaltungen geladen, die der API-User sehen darf. Ein Klick öffnet Details inkl. Turniere sowie Startliste (`/startliste/veranstaltung/{id}`).