#!/usr/bin/env python3
"""
Mini-UI to browse events and tournaments from the ESV API.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, redirect, render_template, request, url_for
from requests.auth import HTTPBasicAuth

DEFAULT_BASE_URL = "https://ev.tanzsport-portal.de/api/v1"
QA_BASE_URL = "https://dtv-esv-qa.azurewebsites.net/api/v1"
DEFAULT_ACCEPT = "application/json"
ACCEPT_EVENT_DETAIL = "application/vnd.tanzsport.esv.v1.veranstaltung.l2+json"
ACCEPT_STARTLIST = "application/vnd.tanzsport.esv.v1.startliste.l2+json"
ACCEPT_FUNCTIONARIES = "application/vnd.tanzsport.esv.v1.funktionaere.l2+json"
REQUEST_TIMEOUT = 20
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
SETTINGS_FILE = Path("settings.json")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("ESV_UI_SECRET", "dev-secret-key")


@dataclass
class Settings:
    base_url: str
    username: str
    password: str
    accept: str
    user_agent: str
    default_organizer_filter: str


def default_settings() -> Settings:
    return Settings(
        base_url=DEFAULT_BASE_URL,
        username="",
        password="",
        accept=DEFAULT_ACCEPT,
        user_agent="",
        default_organizer_filter="",
    )


def load_settings() -> Settings:
    if not SETTINGS_FILE.exists():
        return default_settings()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_settings()
    merged = default_settings().__dict__
    merged.update({k: v for k, v in data.items() if k in merged})
    return Settings(**merged)


def save_settings(settings: Settings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def missing_credentials(settings: Settings) -> bool:
    return not settings.username or not settings.password or not settings.user_agent.strip()


def build_user_agent(settings: Settings) -> str:
    return settings.user_agent.strip()


def perform_get(path: str, settings: Settings, accept_override: str | None = None) -> Any:
    base = settings.base_url.rstrip("/")
    url = base + path if path.startswith("/") else f"{base}/{path}"
    headers = {
        "Accept": accept_override or settings.accept or DEFAULT_ACCEPT,
        "User-Agent": build_user_agent(settings),
        "Content-Type": "application/json",
    }
    response = requests.get(
        url,
        auth=HTTPBasicAuth(settings.username, settings.password),
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    if response.content:
        return response.json()
    return {}


def fetch_event_list(settings: Settings, organizer_filter: str) -> Tuple[List[Dict[str, Any]], str | None]:
    try:
        payload = perform_get("/veranstaltungen", settings)
    except requests.HTTPError as exc:
        return [], f"HTTP-Fehler: {exc.response.status_code} {exc.response.reason}"
    except requests.RequestException as exc:
        return [], f"Netzwerk- oder SSL-Fehler: {exc}"

    if isinstance(payload, dict):
        payload = payload.get("veranstaltungen", [])

    if not isinstance(payload, list):
        return [], "Unerwartetes Antwortformat für die Veranstaltungsliste."

    events = []
    for item in payload:
        if organizer_filter:
            haystack = " ".join(
                filter(
                    None,
                    [
                        str(item.get("veranstalter", {}).get("name", "")),
                        str(item.get("ausrichter", {}).get("name", "")),
                    ],
                )
            ).lower()
            if organizer_filter.lower() not in haystack:
                continue
        events.append(
            {
                "id": item.get("id"),
                "title": item.get("titel") or item.get("name") or f"Veranstaltung {item.get('id')}",
                "city": item.get("ort"),
                "date_from": item.get("datumVon"),
                "date_to": item.get("datumBis"),
                "ausrichter": (item.get("ausrichter") or {}).get("name"),
                "veranstalter": (item.get("veranstalter") or {}).get("name"),
            }
        )
    events.sort(key=lambda x: x.get("date_from") or "")
    return events, None


def fetch_event_detail(settings: Settings, event_id: str) -> Tuple[Dict[str, Any] | None, str | None]:
    try:
        payload = perform_get(f"/turniere/{event_id}", settings, accept_override=ACCEPT_EVENT_DETAIL)
        if not isinstance(payload, dict):
            return None, "Unerwartetes Antwortformat für die Veranstaltung."
        return payload, None
    except requests.HTTPError as exc:
        msg = f"HTTP-Fehler: {exc.response.status_code} {exc.response.reason}"
    except requests.RequestException as exc:
        msg = f"Netzwerk- oder SSL-Fehler: {exc}"
    return None, msg


def fetch_startlist(settings: Settings, event_id: str) -> Tuple[Dict[str, Any] | None, str | None]:
    try:
        payload = perform_get(f"/startliste/veranstaltung/{event_id}", settings, accept_override=ACCEPT_STARTLIST)
        if not isinstance(payload, dict):
            return None, "Unerwartetes Antwortformat für die Startliste."
        return payload, None
    except requests.HTTPError as exc:
        msg = f"HTTP-Fehler: {exc.response.status_code} {exc.response.reason}"
    except requests.RequestException as exc:
        msg = f"Netzwerk- oder SSL-Fehler: {exc}"
    return None, msg


def fetch_functionaries(settings: Settings) -> Tuple[List[Dict[str, Any]], str | None]:
    try:
        payload = perform_get("/funktionaere", settings, accept_override=ACCEPT_FUNCTIONARIES)
    except requests.HTTPError as exc:
        return [], f"HTTP-Fehler: {exc.response.status_code} {exc.response.reason}"
    except requests.RequestException as exc:
        return [], f"Netzwerk- oder SSL-Fehler: {exc}"
    if not isinstance(payload, list):
        return [], "Unerwartetes Antwortformat für die Funktionärsliste."
    return payload, None


@app.route("/")
def root() -> Any:
    return redirect(url_for("events"))


@app.route("/settings", methods=["GET", "POST"])
def settings_view() -> str:
    settings = load_settings()
    message = None
    error = None
    if request.method == "POST":
        base_url = request.form.get("base_url", DEFAULT_BASE_URL).strip()
        if base_url not in {DEFAULT_BASE_URL, QA_BASE_URL}:
            error = "Ungültige Basis-URL."
        else:
            data = Settings(
                base_url=base_url,
                username=request.form.get("username", "").strip(),
                password=request.form.get("password", "").strip(),
                accept=request.form.get("accept", DEFAULT_ACCEPT).strip() or DEFAULT_ACCEPT,
                user_agent=request.form.get("user_agent", settings.user_agent).strip(),
                default_organizer_filter=request.form.get("default_organizer_filter", "").strip(),
            )
            save_settings(data)
            settings = data
            message = "Einstellungen gespeichert."
    return render_template(
        "settings.html",
        settings=settings,
        message=message,
        error=error,
        active_tab="settings",
    )


@app.route("/events")
def events() -> str:
    settings = load_settings()
    event_id = request.args.get("event_id", "").strip()
    if event_id:
        return redirect(url_for("event_detail", event_id=event_id))
    organizer_filter_param = request.args.get("organizer")
    if organizer_filter_param is None:
        organizer_filter = settings.default_organizer_filter
    else:
        organizer_filter = organizer_filter_param
    events_data: List[Dict[str, Any]] = []
    error = None
    creds_missing = missing_credentials(settings)
    if not creds_missing:
        events_data, error = fetch_event_list(settings, organizer_filter)
    return render_template(
        "events.html",
        settings=settings,
        events=events_data,
        event_error=error,
        organizer_filter=organizer_filter or "",
        default_filter=settings.default_organizer_filter,
        creds_missing=creds_missing,
        active_tab="events",
    )


@app.route("/events/<event_id>")
def event_detail(event_id: str) -> str:
    settings = load_settings()
    if missing_credentials(settings):
        return redirect(url_for("settings_view"))

    event_data, event_error = fetch_event_detail(settings, event_id)
    startlist_data, startlist_error = fetch_startlist(settings, event_id)
    return render_template(
        "event.html",
        event=event_data,
        event_error=event_error,
        startlist=startlist_data,
        startlist_error=startlist_error,
        active_tab="events",
    )


@app.route("/functionaries")
def functionaries() -> str:
    settings = load_settings()
    query = request.args.get("q", "").strip()
    show_all = request.args.get("show_all") == "1"
    creds_missing = missing_credentials(settings)
    functionaries_list: List[Dict[str, Any]] = []
    error = None
    if not creds_missing and (query or show_all):
        payload, error = fetch_functionaries(settings)
        if not error:
            if show_all:
                functionaries_list = payload
            else:
                lowered = query.lower()
                for entry in payload:
                    blob = " ".join(
                        str(entry.get(field, "")) for field in ("id", "titel", "vorname", "nachname", "club", "staat")
                    ).lower()
                    if lowered in blob:
                        functionaries_list.append(entry)
    elif not query and not show_all:
        error = "Bitte Suchbegriff eingeben oder \"Alle anzeigen\" wählen."

    return render_template(
        "functionaries.html",
        settings=settings,
        functionaries=functionaries_list,
        error=error,
        query=query,
        show_all=show_all,
        creds_missing=creds_missing,
        active_tab="functionaries",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
