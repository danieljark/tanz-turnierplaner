#!/usr/bin/env python3
from __future__ import annotations

import json
import base64
import ftplib
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, time as time_cls
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, Response, redirect, render_template, request, url_for
from requests.auth import HTTPBasicAuth

DEFAULT_BASE_URL = "https://ev.tanzsport-portal.de/api/v1"
QA_BASE_URL = "https://dtv-esv-qa.azurewebsites.net/api/v1"
DEFAULT_ACCEPT = "application/json"
ACCEPT_EVENT_DETAIL = "application/vnd.tanzsport.esv.v1.veranstaltung.l2+json"
ACCEPT_STARTLIST = "application/vnd.tanzsport.esv.v1.startliste.l2+json"
ACCEPT_FUNCTIONARIES = "application/vnd.tanzsport.esv.v1.funktionaere.l2+json"
ACCEPT_STARTLIST_LEVEL1 = "application/vnd.tanzsport.esv.v1.startliste.l1+json"
REQUEST_TIMEOUT = 20
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
SETTINGS_FILE = Path("settings.json")
PLANS_FILE = Path("plans.json")
PLANNER_RULES_PATH = BASE_DIR / "planner_rules.xml"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("ESV_UI_SECRET", "dev-secret-key")


@app.context_processor
def inject_now() -> Dict[str, Any]:
    return {"current_year": datetime.utcnow().year}


@dataclass
class Settings:
    base_url: str
    username: str
    password: str
    accept: str
    user_agent: str
    default_organizer_filter: str
    ftp_host: str
    ftp_user: str
    ftp_password: str
    ftp_path: str


def default_settings() -> Settings:
    return Settings(
        base_url=DEFAULT_BASE_URL,
        username="",
        password="",
        accept=DEFAULT_ACCEPT,
        user_agent="",
        default_organizer_filter="",
        ftp_host="",
        ftp_user="",
        ftp_password="",
        ftp_path="",
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


@lru_cache(maxsize=1)
def load_planner_rules() -> Dict[str, Any]:
    def _parse_float(val: str, default: float = 0.0) -> float:
        try:
            return float(val.replace(",", "."))
        except Exception:
            return default

    if not PLANNER_RULES_PATH.exists():
        raise FileNotFoundError(f"Planner configuration missing at {PLANNER_RULES_PATH}")
    tree = ET.parse(PLANNER_RULES_PATH)
    root = tree.getroot()
    general_el = root.find("general")
    general = {
        "default_heat_size": int(general_el.get("defaultHeatSize", "6")) if general_el is not None else 6,
        "max_heat_size": int(general_el.get("maxHeatSize", "8")) if general_el is not None else 8,
        "break_minutes": int(general_el.get("breakMinutes", "10")) if general_el is not None else 10,
        "final_duration": int(general_el.get("finalDuration", "10")) if general_el is not None else 10,
        "gap_between_tournaments": int(general_el.get("gapBetweenTournaments", "5")) if general_el is not None else 5,
        "default_start": (general_el.get("defaultStart") if general_el is not None else "09:00"),
        "buffer_per_heat": _parse_float(general_el.get("bufferPerHeat", "0")) if general_el is not None else 0.0,
    }
    thresholds: List[Dict[str, Any]] = []
    for threshold in root.findall("./roundThresholds/threshold"):
        rounds = [r.strip() for r in threshold.get("rounds", "").split(",") if r.strip()]
        next_counts = [int(x.strip()) for x in threshold.get("nextCounts", "").split(",") if x.strip()]
        crosses = [x.strip() for x in threshold.get("crosses", "").split(",") if x.strip()]
        thresholds.append(
            {
                "min": int(threshold.get("min", "0")),
                "max": int(threshold.get("max")) if threshold.get("max") else None,
                "rounds": rounds,
                "next_counts": next_counts,
                "crosses": crosses,
            }
        )
    thresholds.sort(key=lambda item: item["min"], reverse=True)
    dance_durations: Dict[str, float] = {}
    for dance in root.findall("./danceDurations/dance"):
        code = dance.get("code")
        if not code:
            continue
        # use max duration to be on the safe side
        max_val = float(dance.get("max", "2.0"))
        dance_durations[code] = max_val
    return {
        "general": general,
        "thresholds": thresholds,
        "dance_durations": dance_durations,
    }


def determine_round_rule(starters: int, thresholds: List[Dict[str, Any]]) -> Dict[str, Any]:
    for rule in thresholds:
        if starters >= rule["min"] and (rule["max"] is None or starters <= rule["max"]):
            return rule
    return thresholds[-1]


def load_saved_plans() -> List[Dict[str, Any]]:
    if not PLANS_FILE.exists():
        return []
    try:
        return json.loads(PLANS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def persist_saved_plans(plans: List[Dict[str, Any]]) -> None:
    PLANS_FILE.write_text(json.dumps(plans, indent=2, ensure_ascii=False), encoding="utf-8")


def find_saved_plan(plan_id: str) -> Dict[str, Any] | None:
    for plan in load_saved_plans():
        if plan.get("id") == plan_id:
            return plan
    return None


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


def parse_date(value: str) -> datetime.date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def parse_time_value(value: str) -> time_cls | None:
    if not value:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    return None


STANDARD_DANCES = {
    "E": ["LW", "TG", "WW", "SF", "QU"],
    "D": ["LW", "TG", "QU"],
    "C": ["LW", "TG", "SF", "QU"],
    "B": ["LW", "TG", "WW", "SF", "QU"],
    "A": ["LW", "TG", "WW", "SF", "QU"],
    "S": ["LW", "TG", "WW", "SF", "QU"],
    "BSW": ["LW", "TG", "WW", "SF", "QU"],
}
LATIN_DANCES = {
    "E": ["SB", "CC", "RB", "PD", "JV"],
    "D": ["CC", "RB", "JV"],
    "C": ["SB", "CC", "RB", "JV"],
    "B": ["SB", "CC", "RB", "PD", "JV"],
    "A": ["SB", "CC", "RB", "PD", "JV"],
    "S": ["SB", "CC", "RB", "PD", "JV"],
    "BSW": ["SB", "CC", "RB", "PD", "JV"],
}
DEFAULT_DANCES = ["GL"]


def determine_dances(tournament: Dict[str, Any]) -> List[str]:
    art = (tournament.get("turnierart") or "").lower()
    startklasse = (tournament.get("startklasseLiga") or tournament.get("startklasse") or "").upper()
    if art in ("std", "standard"):
        return STANDARD_DANCES.get(startklasse, STANDARD_DANCES["B"])
    if art in ("lat", "latein"):
        return LATIN_DANCES.get(startklasse, LATIN_DANCES["B"])
    if art in ("kmb", "kombination"):
        return STANDARD_DANCES.get("B") + LATIN_DANCES.get("B")
    return DEFAULT_DANCES


STARTGRUPPE_LABELS = {
    "HGR": "Hauptgruppe",
    "HGRII": "Hauptgruppe II",
    "JUN": "Junioren",
    "JUNI": "Junioren I",
    "JUNII": "Junioren II",
    "JUG": "Jugend",
    "SENI": "Senioren I",
    "SENII": "Senioren II",
    "SENIII": "Senioren III",
    "SENIV": "Senioren IV",
}


def prettify_startgruppe(raw: str | None) -> str:
    if not raw:
        return ""
    key = raw.replace(" ", "").replace("/", "").upper()
    return STARTGRUPPE_LABELS.get(key, raw)


def generate_rounds_for_tournament(
    tournament: Dict[str, Any],
    starters: int,
    heat_size: int,
    rules: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    heat_size = max(1, heat_size)
    thresholds = rules["thresholds"]
    general = rules["general"]
    rule = determine_round_rule(max(starters, 0), thresholds)
    counts = [max(starters, 0)]
    for target in rule["next_counts"]:
        counts.append(min(target, counts[-1]) if counts[-1] else target)
    while len(counts) < len(rule["rounds"]):
        counts.append(max(6, counts[-1] if counts else 6))
    dances = determine_dances(tournament)
    durations = rules["dance_durations"]
    dance_minutes = sum(durations.get(code, 1.5) for code in dances) or 1.5
    rounds: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    total_minutes = 0.0
    for idx, round_name in enumerate(rule["rounds"]):
        couples = counts[idx] if idx < len(counts) else counts[-1]
        couples = int(max(couples, 0))
        heats = max(1, math.ceil(couples / heat_size)) if couples else 1
        duration = math.ceil(heats * (dance_minutes + general.get("buffer_per_heat", 0)))
        rounds.append(
            {
                "name": round_name,
                "couples": couples,
                "heats": heats,
                "dances": dances,
                "minutes": duration,
                "cross_info": rule["crosses"][idx] if idx < len(rule["crosses"]) else None,
            }
        )
        total_minutes += duration
        if idx < len(rule["rounds"]) - 1:
            pause_duration = math.ceil(general["break_minutes"])
            total_minutes += pause_duration
            blocks.append(
                {
                    "type": "pause",
                    "label": "Pause",
                    "minutes": pause_duration,
                }
            )

    if general.get("final_duration", 0) > 0:
        final_duration = math.ceil(float(general["final_duration"]))
        blocks.append({"type": "final", "label": "Siegerehrung", "minutes": final_duration})
        total_minutes += final_duration

    # prepend the rounds as blocks
    combined_blocks: List[Dict[str, Any]] = []
    block_iter = iter(blocks)
    for idx, round_data in enumerate(rounds):
        combined_blocks.append(
            {
                "type": "round",
                "label": round_data["name"],
                "minutes": round_data["minutes"],
                "couples": round_data["couples"],
                "heats": round_data["heats"],
                "dances": round_data["dances"],
                "cross_info": round_data["cross_info"],
            }
        )
        # insert pause after round if present at corresponding index
        try:
            pause_block = next(block_iter)
            combined_blocks.append(pause_block)
        except StopIteration:
            pass
    # any remaining blocks (e.g., finale)
    for block in block_iter:
        combined_blocks.append(block)

    # recalc total from combined_blocks
    total_minutes = sum(float(b.get("minutes", 0)) for b in combined_blocks)

    return rounds, total_minutes, combined_blocks


def build_event_plan(
    settings: Settings,
    event_id: str,
    heat_size: int,
    cancel_enabled: bool = True,
    cancel_under: int = 4,
    previous_canceled: List[str] | None = None,
) -> Tuple[Dict[str, Any] | None, str | None]:
    rules = load_planner_rules()
    heat_size = max(1, min(heat_size, rules["general"]["max_heat_size"]))
    event_data, error = fetch_event_detail(settings, event_id)
    if error:
        return None, error
    startlist_data, startlist_error = fetch_startlist(settings, event_id)
    warnings: List[str] = []
    if startlist_error:
        warnings.append(f"Startliste konnte nicht geladen werden: {startlist_error}")
        startlist_data = None
    starter_counts: Dict[str, int] = {}
    if event_data.get("turniere"):
        for t in event_data["turniere"]:
            starter_counts[str(t.get("id"))] = 0
    if startlist_data and startlist_data.get("starter"):
        for starter in startlist_data["starter"]:
            for meldung in starter.get("meldungen", []):
                tid = meldung.get("turnierId") or meldung.get("turnierID") or meldung.get("turnierid")
                if tid is None:
                    continue
                tid_str = str(tid)
                if tid_str not in starter_counts:
                    continue
                if "meldung" in meldung and not meldung.get("meldung"):
                    continue
                starter_counts[tid_str] += 1
    days: Dict[str, List[Dict[str, Any]]] = {}
    if not event_data.get("turniere"):
        return None, "Keine Turniere im Datensatz gefunden."

    canceled_list: List[Dict[str, Any]] = []
    prev_canceled_set = set(previous_canceled or [])

    for tournament in event_data["turniere"]:
        tid = str(tournament.get("id"))
        starters = starter_counts.get(tid, 0)
        was_canceled_before = tid in prev_canceled_set
        canceled_flag = cancel_enabled and starters < cancel_under
        was_canceled_before = tid in prev_canceled_set
        reactivated_flag = was_canceled_before and not canceled_flag
        if canceled_flag:
            canceled_list.append(
                {
                    "id": tid,
                    "title": tournament.get("titel") or tournament.get("turnierart"),
                    "starters": starters,
                    "reason": f"Abgesagt (< {cancel_under} Starter)",
                }
            )
        rounds, total_minutes, blocks = generate_rounds_for_tournament(tournament, starters, heat_size, rules)
        if canceled_flag:
            blocks = []
            total_minutes = 0
        date_key = tournament.get("datumVon") or event_data.get("datumVon")
        day_list = days.setdefault(date_key or "unbekannt", [])
        plan_entry = {
            "id": tid,
            "title": tournament.get("titel") or tournament.get("turnierart"),
            "startgruppe": prettify_startgruppe(tournament.get("startgruppe")),
            "startklasse": tournament.get("startklasseLiga"),
            "wettbewerbsart": tournament.get("wettbewerbsart"),
            "turnierart": tournament.get("turnierart"),
            "starters": starters,
            "rounds": rounds,
            "total_minutes": total_minutes,
            "startzeitPlan": tournament.get("startzeitPlan"),
            "heat_size": heat_size,
            "blocks": blocks,
            "reactivated": reactivated_flag,
            "canceled": canceled_flag,
        }
        day_list.append(plan_entry)

    general = rules["general"]
    scheduled_days: List[Dict[str, Any]] = []
    for date_key, tournaments in sorted(days.items()):
        date_obj = parse_date(date_key) or datetime.today().date()
        day_start_time = parse_time_value(general["default_start"]) or time_cls(hour=9, minute=0)
        current_time = datetime.combine(date_obj, day_start_time)
        tournaments.sort(key=lambda x: parse_time_value(x.get("startzeitPlan") or "") or day_start_time)
        for tournament in tournaments:
            explicit_time = parse_time_value(tournament.get("startzeitPlan"))
            if explicit_time:
                explicit_dt = datetime.combine(date_obj, explicit_time)
                if explicit_dt > current_time:
                    current_time = explicit_dt
            tournament["start_time"] = current_time.strftime("%H:%M")
            end_time = current_time + timedelta(minutes=tournament["total_minutes"])
            tournament["end_time"] = end_time.strftime("%H:%M")
            current_time = end_time + timedelta(minutes=general["gap_between_tournaments"])
        scheduled_days.append(
            {
                "date": date_key,
                "display_date": format_date_label(date_key),
                "start_time": day_start_time.strftime("%H:%M"),
                "tournaments": tournaments,
            }
        )

    plan = {
        "event": {
            "id": event_data.get("id"),
            "name": event_data.get("name") or event_data.get("titel"),
            "ort": event_data.get("ort"),
            "datumVon": event_data.get("datumVon"),
            "datumBis": event_data.get("datumBis"),
        },
        "heat_size": heat_size,
        "cancel_under": cancel_under,
        "cancel_enabled": cancel_enabled,
        "generated_at": datetime.utcnow().isoformat(),
        "warnings": warnings,
        "days": scheduled_days,
        "canceled": canceled_list,
        "plan_name": event_data.get("name") or event_data.get("titel") or f"Veranstaltung {event_data.get('id')}",
    }
    return plan, None


def build_startlists_by_tournament(startlist: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    result: Dict[str, List[Dict[str, str]]] = {}
    if not startlist or not startlist.get("starter"):
        return result
    for starter in startlist["starter"]:
        meldungen = starter.get("meldungen") or []
        personen = starter.get("personen") or []
        names = []
        if personen:
            if len(personen) >= 2:
                names.append(f"{personen[0].get('nachname','')} {personen[0].get('vorname','')}".strip())
                names.append(f"{personen[1].get('nachname','')} {personen[1].get('vorname','')}".strip())
                names = [" & ".join(names)]
            else:
                names.append(f"{personen[0].get('nachname','')} {personen[0].get('vorname','')}".strip())
        display_name = names[0] if names else f"Starter {starter.get('id')}"
        club = None
        ltv = None
        if starter.get("club"):
            club = starter["club"].get("name")
            if starter["club"].get("ltv"):
                ltv = starter["club"]["ltv"].get("name")
        for meldung in meldungen:
            tid = meldung.get("turnierId") or meldung.get("turnierID") or meldung.get("turnierid")
            if tid is None:
                continue
            tid_str = str(tid)
            result.setdefault(tid_str, []).append(
                {
                    "names": display_name,
                    "club": club,
                    "ltv": ltv,
                }
            )
    return result


def format_date_label(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return date_str
    weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    return f"{dt.strftime('%d.%m.%Y')} - {weekdays[dt.weekday()]}"


def render_schedule_html(plan: Dict[str, Any], startlists: Dict[str, List[Dict[str, str]]] | None, variant: str = "internal") -> str:
    # Normalize dates for display
    for day in plan.get("days", []):
        day["display_date"] = format_date_label(day.get("date"))
    plan["event"]["range_label"] = ""
    if plan["event"].get("datumVon"):
        von = format_date_label(plan["event"]["datumVon"])
        bis = format_date_label(plan["event"].get("datumBis") or plan["event"]["datumVon"])
        plan["event"]["range_label"] = f"{von} – {bis}" if bis and bis != von else von

    template_name = "static_schedule.html" if variant == "internal" else "static_schedule_public.html"
    return render_template(template_name, plan=plan, startlists=startlists or {})


def save_plan_record(plan: Dict[str, Any]) -> Dict[str, Any]:
    plans = load_saved_plans()
    plan_id = f"{plan['event']['id']}-{int(time.time())}"
    record = {
        "id": plan_id,
        "event_id": plan["event"]["id"],
        "event_name": plan["event"]["name"],
        "created_at": plan["generated_at"],
        "heat_size": plan["heat_size"],
        "cancel_under": plan.get("cancel_under"),
        "cancel_enabled": plan.get("cancel_enabled", True),
        "plan_name": plan.get("plan_name") or plan["event"]["name"],
        "content": plan,
    }
    plans.append(record)
    persist_saved_plans(plans)
    return record


def recompute_plan_entries(
    plan: Dict[str, Any],
    rules: Dict[str, Any],
    heat_size: int,
    startlists: Dict[str, List[Dict[str, str]]] | None = None,
) -> None:
    for day in plan.get("days", []):
        for entry in day.get("tournaments", []):
            entry_id = str(entry.get("id") or "")
            has_blocks = bool(entry.get("blocks"))
            if entry_id.startswith("BLOCK") or not entry_id.isdigit():
                # custom block
                if not has_blocks:
                    entry["blocks"] = []
                if startlists is not None:
                    entry["startlist"] = startlists.get(entry_id, [])
                continue
            tournament_stub = {
                "turnierart": entry.get("turnierart"),
                "wettbewerbsart": entry.get("wettbewerbsart"),
                "startklasseLiga": entry.get("startklasse"),
                "startklasse": entry.get("startklasse"),
                "startgruppe": entry.get("startgruppe"),
            }
            starters = int(entry.get("starters") or 0)
            rounds, total_minutes, blocks = generate_rounds_for_tournament(
                tournament_stub,
                starters,
                heat_size,
                rules,
            )
            entry["rounds"] = rounds
            if entry.get("canceled"):
                entry["blocks"] = []
                entry["total_minutes"] = 0
            else:
                entry["blocks"] = blocks
                entry["total_minutes"] = total_minutes
            if startlists is not None:
                entry["startlist"] = startlists.get(entry_id, [])


def update_plan_record(plan_id: str, plan: Dict[str, Any]) -> Dict[str, Any] | None:
    plans = load_saved_plans()
    updated = None
    for idx, rec in enumerate(plans):
        if rec.get("id") == plan_id:
            rec["content"] = plan
            rec["created_at"] = plan.get("generated_at") or rec.get("created_at")
            rec["heat_size"] = plan.get("heat_size", rec.get("heat_size"))
            rec["cancel_under"] = plan.get("cancel_under", rec.get("cancel_under"))
            rec["cancel_enabled"] = plan.get("cancel_enabled", rec.get("cancel_enabled"))
            rec["event_name"] = plan.get("event", {}).get("name", rec.get("event_name"))
            rec["plan_name"] = plan.get("plan_name", rec.get("plan_name", rec.get("event_name")))
            plans[idx] = rec
            updated = rec
            break
    if updated:
        persist_saved_plans(plans)
    return updated


def delete_plan_record(plan_id: str) -> bool:
    plans = load_saved_plans()
    new_plans = [rec for rec in plans if rec.get("id") != plan_id]
    if len(new_plans) == len(plans):
        return False
    persist_saved_plans(new_plans)
    return True


def rename_plan_record(plan_id: str, new_name: str) -> bool:
    plans = load_saved_plans()
    changed = False
    for rec in plans:
        if rec.get("id") == plan_id:
            rec["plan_name"] = new_name
            if "content" in rec:
                rec["content"]["plan_name"] = new_name
            changed = True
            break
    if changed:
        persist_saved_plans(plans)
    return changed


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
                ftp_host=request.form.get("ftp_host", "").strip(),
                ftp_user=request.form.get("ftp_user", "").strip(),
                ftp_password=request.form.get("ftp_password", "").strip(),
                ftp_path=request.form.get("ftp_path", "").strip(),
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


@app.route("/planner", methods=["GET", "POST"])
def planner() -> str:
    settings = load_settings()
    creds_missing = missing_credentials(settings)
    saved_plans = load_saved_plans()
    refresh = request.args.get("refresh") == "1"
    action = request.form.get("action") if request.method == "POST" else None
    selected_plan_id = request.args.get("plan_id") or (request.form.get("plan_id") if request.method == "POST" else None)
    try:
        planner_rules = load_planner_rules()
    except FileNotFoundError as exc:
        planner_rules = None
        rules_error = str(exc)
    else:
        rules_error = None
    default_heat_size = planner_rules["general"]["default_heat_size"] if planner_rules else 6
    heat_size_val = default_heat_size
    event_id = ""
    cancel_under = 4
    cancel_enabled = True
    plan_data = None
    message = None
    error = rules_error
    plan_source = None

    if request.method == "POST" and action in {"delete_plan", "rename_plan"}:
        target_plan_id = request.form.get("plan_id")
        if not target_plan_id:
            error = "Plan-ID fehlt."
        elif action == "delete_plan":
            if delete_plan_record(target_plan_id):
                message = "Plan gelöscht."
                if selected_plan_id == target_plan_id:
                    selected_plan_id = None
                    plan_data = None
            else:
                error = "Plan konnte nicht gelöscht werden."
        elif action == "rename_plan":
            new_name = request.form.get("plan_name", "").strip()
            if not new_name:
                error = "Bitte einen Namen für den Plan angeben."
            elif rename_plan_record(target_plan_id, new_name):
                message = "Plan umbenannt."
                if plan_data and plan_data.get("plan_id") == target_plan_id:
                    plan_data["plan_name"] = new_name
            else:
                error = "Plan konnte nicht umbenannt werden."
        saved_plans = load_saved_plans()
        action = None

    if selected_plan_id:
        record = find_saved_plan(selected_plan_id)
        if record:
            plan_data = record["content"]
            plan_data["plan_id"] = record["id"]
            plan_data["plan_name"] = record.get("plan_name") or plan_data.get("plan_name") or plan_data["event"].get("name")
            plan_source = "saved"
            heat_size_val = plan_data.get("heat_size", default_heat_size)
            event_id = str(record.get("event_id") or "")
            cancel_under = plan_data.get("cancel_under", cancel_under)
            cancel_enabled = plan_data.get("cancel_enabled", cancel_enabled)
            if refresh and not creds_missing:
                startlist_data, startlist_error = fetch_startlist(settings, event_id)
                if startlist_error:
                    error = f"Startliste konnte nicht geladen werden: {startlist_error}"
                elif planner_rules:
                    startlists = build_startlists_by_tournament(startlist_data)
                    plan_heat_size = int(plan_data.get("heat_size", heat_size_val))
                    recompute_plan_entries(plan_data, planner_rules, plan_heat_size, startlists=startlists)
                    plan_source = "refreshed"
        else:
            error = "Gespeicherter Plan wurde nicht gefunden."

    if request.method == "POST" and action == "save_current":
        payload_raw = request.form.get("plan_payload", "")
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            error = "Ungültiges Plan-JSON."
            payload = None
        if payload:
            record = find_saved_plan(selected_plan_id)
            if not record:
                error = "Plan zum Aktualisieren nicht gefunden."
            else:
                plan = record["content"]
                plan["plan_name"] = record.get("plan_name") or plan.get("plan_name") or plan["event"].get("name")
                plan["days"] = payload.get("days", plan.get("days", []))
                plan["generated_at"] = datetime.utcnow().isoformat()
                if planner_rules:
                    plan_heat_size = int(plan.get("heat_size", heat_size_val))
                    recompute_plan_entries(plan, planner_rules, plan_heat_size)
                canceled = []
                for day in plan["days"]:
                    for t in day.get("tournaments", []):
                        if t.get("canceled"):
                            canceled.append({"id": t.get("id"), "title": t.get("title"), "reason": "Manuell abgesagt"})
                plan["canceled"] = canceled
                updated = update_plan_record(record["id"], plan)
                if updated:
                    message = "Plan aktualisiert."
                    plan_data = updated["content"]
                    plan_data["plan_id"] = updated["id"]
                else:
                    error = "Plan konnte nicht aktualisiert werden."

    if request.method == "POST" and action != "save_current":
        action = request.form.get("action", "generate")
        event_id = request.form.get("event_id", "").strip()
        try:
            heat_size_val = int(request.form.get("heat_size", default_heat_size))
        except ValueError:
            heat_size_val = default_heat_size
        try:
            cancel_under = int(request.form.get("cancel_under", cancel_under))
        except ValueError:
            cancel_under = 4
        cancel_enabled = request.form.get("cancel_enabled") == "on"
        if planner_rules:
            heat_size_val = max(1, min(heat_size_val, planner_rules["general"]["max_heat_size"]))
        if not event_id:
            error = "Bitte eine Veranstaltungs-ID angeben."
        elif creds_missing:
            error = "Bitte zunächst Zugangsdaten im Tab Einstellungen hinterlegen."
        elif not planner_rules:
            error = rules_error or "Planer-Konfiguration konnte nicht geladen werden."
        elif action != "save_current":
            plan_data, plan_error = build_event_plan(
                settings,
                event_id,
                heat_size_val,
                cancel_enabled=cancel_enabled,
                cancel_under=cancel_under,
                previous_canceled=None,
            )
            if plan_error:
                error = plan_error
                plan_data = None
            else:
                plan_source = "generated"
                if action == "save":
                    record = save_plan_record(plan_data)
                    message = f"Plan gespeichert ({record['id']})."
                    plan_data = record["content"]
                    plan_data["plan_id"] = record["id"]
                    saved_plans = load_saved_plans()

    return render_template(
        "planner.html",
        settings=settings,
        creds_missing=creds_missing,
        plan=plan_data,
        heat_size=heat_size_val,
        default_heat_size=default_heat_size,
        saved_plans=saved_plans,
        message=message,
        error=error,
        event_id=event_id,
        cancel_under=cancel_under,
        cancel_enabled=cancel_enabled,
        active_tab="planner",
        plan_source=plan_source,
    )


@app.route("/publish", methods=["GET", "POST"])
def publish() -> str:
    settings = load_settings()
    saved_plans = load_saved_plans()
    selected_plan_id = None
    preview_html = None
    message = None
    error = None
    ftp_host = settings.ftp_host
    ftp_user = settings.ftp_user
    ftp_password = settings.ftp_password
    ftp_path = settings.ftp_path
    filename = "zeitplan.html"
    variant = "internal"

    if request.method == "POST":
        action = request.form.get("action")
        selected_plan_id = request.form.get("plan_id")
        variant = request.form.get("variant", variant)
        ftp_host = request.form.get("ftp_host", ftp_host)
        ftp_user = request.form.get("ftp_user", ftp_user)
        ftp_password = request.form.get("ftp_password", ftp_password)
        ftp_path = request.form.get("ftp_path", ftp_path)
        filename = request.form.get("filename", filename)
        record = find_saved_plan(selected_plan_id) if selected_plan_id else None
        if not record:
            error = "Bitte zuerst einen gespeicherten Plan wählen."
        else:
            plan = record["content"]
            startlist_data, startlist_error = fetch_startlist(settings, str(plan["event"]["id"]))
            startlists = build_startlists_by_tournament(startlist_data) if not startlist_error else {}
            if startlist_error:
                error = f"Startliste konnte nicht geladen werden: {startlist_error}"
            html = render_schedule_html(plan, startlists, variant=variant)
            if action == "preview":
                preview_html = base64.b64encode(html.encode("utf-8")).decode("utf-8")
                message = "Vorschau erzeugt."
            elif action == "upload":
                try:
                    with ftplib.FTP(ftp_host) as ftp:
                        ftp.login(ftp_user, ftp_password)
                        if ftp_path:
                            ftp.cwd(ftp_path)
                        with ftp.transfercmd(f"STOR {filename}") as conn:
                            conn.sendall(html.encode("utf-8"))
                        message = "Upload erfolgreich."
                except Exception as exc:  # noqa: BLE001
                    error = f"FTP-Upload fehlgeschlagen: {exc}"

    return render_template(
        "publish.html",
        saved_plans=saved_plans,
        selected_plan_id=selected_plan_id,
        preview_html=preview_html,
        message=message,
        error=error,
        ftp_host=ftp_host,
        ftp_user=ftp_user,
        ftp_password=ftp_password,
        ftp_path=ftp_path,
        filename=filename,
        variant=variant,
        active_tab="publish",
    )


@app.route("/planner/export/<plan_id>")
def planner_export(plan_id: str) -> Response:
    record = find_saved_plan(plan_id)
    if not record:
        return Response("Plan nicht gefunden.", status=404)
    payload = json.dumps(record["content"], ensure_ascii=False, indent=2)
    filename = f"plan-{plan_id}.json"
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
