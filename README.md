# Tanzturnierplaner

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

## Funktionsumfang

- **Einstellungen**: Hinterlege Benutzername, Passwort, Accept-Header sowie den kompletten `User-Agent` (z. B. `TPS.net/1.0; Token=abcdef`). Optional kannst du einen Standardfilter für Veranstalter/Ausrichter speichern. Alles landet lokal in `settings.json` (Klartext – nicht einchecken!).
- **Veranstaltungen**: Lädt `/api/v1/veranstaltungen` mit deinem gespeicherten User-Agent. Wird kein Filter angegeben, nutzt die Ansicht automatisch den Standardfilter aus den Einstellungen – lässt du das Feld leer, werden alle Veranstaltungen geladen, die der API-User sehen darf. Ein Klick öffnet Details inkl. Turniere sowie Startliste (`/startliste/veranstaltung/{id}`) und erlaubt per ID direkt in eine Veranstaltung zu springen.
- **Planer**: Eigener Tab, in dem du eine Veranstaltungs-ID und eine gewünschte Heat-Größe eingibst. Das Tool lädt Turnierdaten und Startliste, interpretiert anhand von `planner_rules.xml` die Rundenvorgaben, rechnet Dauer/Heats/Kreuze und stellt alle Turniere chronologisch pro Veranstaltungstag dar. Pläne lassen sich speichern (`plans.json`) und als JSON exportieren, so dass du sie später ohne API-Aufruf wieder abrufen kannst.
- **Funktionäre**: Suchmaske auf `/api/v1/funktionaere` plus eine Schaltfläche „Alle anzeigen“, die die komplette Liste lädt (sofern dein Account freigeschaltet ist).
## Turnierplaner anpassen

- Die Datei `planner_rules.xml` beschreibt Heatgrößen, Pausen, Rundenschwellen und Tanzdauern. Passe die Attribute an, wenn du andere Vorgaben (z. B. mehr Zwischenrunden oder andere Kreuzvorgaben) willst.
- Gespeicherte Pläne landen in `plans.json`. Dort kannst du sie archivieren oder per Tab „Planer“ erneut laden/als JSON exportieren. Möchtest du den Speicher zurücksetzen, lösche die Datei einfach.
