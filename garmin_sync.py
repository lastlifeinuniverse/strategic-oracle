"""
Garmin Connect Sync  Activity import via unofficial garminconnect library.

Fetches recent activities, daily stats (steps, sleep, heart rate, calories)
and maps them to Five Element categories for use by the recommender agent.

Install:  pip install garminconnect
"""

from __future__ import annotations

import csv
import datetime
import io
import os
from typing import Optional

# OAuth token cache  reused across sessions so you only authenticate once
TOKENSTORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".garmin_tokens")

# Five Element mapping for Garmin activity types
ACTIVITY_ELEMENT_MAP: dict[str, tuple[str, str]] = {
    # Running / cardio   Fire (high energy, Yang movement)
    "running":              ("Fire",  "Wood"),
    "trail_running":        ("Fire",  "Wood"),
    "treadmill_running":    ("Fire",  "Wood"),
    "cycling":              ("Fire",  "Metal"),
    "indoor_cycling":       ("Fire",  "Metal"),
    "swimming":             ("Water", "Fire"),
    "open_water_swimming":  ("Water", "Fire"),
    # Yoga / Pilates  Wood (growth, flexibility)
    "yoga":                 ("Wood",  "Water"),
    "pilates":              ("Wood",  "Water"),
    "stretching":           ("Wood",  "Water"),
    # Strength / HIIT  Metal (discipline, structure)
    "strength_training":    ("Metal", "Fire"),
    "hiit":                 ("Metal", "Fire"),
    "cardio":               ("Fire",  "Metal"),
    # Walking / hiking  Wood (nature, grounding)
    "walking":              ("Wood",  "Earth"),
    "hiking":               ("Wood",  "Earth"),
    # Racket sports  Fire (social, dynamic)
    "racket_sports":        ("Fire",  "Metal"),
    "tennis":               ("Fire",  "Metal"),
    "pickleball":           ("Fire",  "Metal"),
    "badminton":            ("Fire",  "Metal"),
    # Meditation / breathwork  Water (stillness, introspection)
    "breathwork":           ("Water", "Metal"),
    "meditation":           ("Water", "Metal"),
    # Other
    "other":                ("Earth", "Wood"),
}

INTENSITY_THRESHOLDS = [
    (0.0,  30,  "light"),
    (30,   60,  "moderate"),
    (60,   999, "high"),
]


def _intensity_from_hr(avg_hr: Optional[float], max_hr: int = 190) -> str:
    if not avg_hr:
        return "moderate"
    pct = (avg_hr / max_hr) * 100
    for lo, hi, label in INTENSITY_THRESHOLDS:
        if lo <= pct < hi:
            return label
    return "high"


def _map_activity_type(garmin_type: str) -> tuple[str, str]:
    key = garmin_type.lower().replace(" ", "_")
    return ACTIVITY_ELEMENT_MAP.get(key, ACTIVITY_ELEMENT_MAP["other"])


def _parse_duration(duration_seconds: Optional[float]) -> int:
    if not duration_seconds:
        return 0
    return int(duration_seconds / 60)


class GarminSync:
    """
    Wraps the unofficial garminconnect library for ORACLE health integration.

    Usage:
        gs = GarminSync()
        ok, msg = gs.connect(email, password)
        activities = gs.get_recent_activities(days=7)
        stats = gs.get_today_stats()
    """

    def __init__(self):
        self.client = None
        self._connected = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def connect(self, email: str, password: str, mfa_code: str = "") -> tuple[bool, str]:
        try:
            from garminconnect import Garmin, GarminConnectAuthenticationError
        except ImportError:
            return False, "garminconnect not installed. Run: pip install garminconnect"

        #  1. Try cached OAuth tokens first (no password needed) 
        if os.path.exists(TOKENSTORE):
            try:
                self.client = Garmin(email=email, password=password)
                self.client.login(tokenstore=TOKENSTORE)
                self._connected = True
                display = self.client.get_full_name() or email
                return True, f"Connected as {display} (cached session)"
            except Exception:
                pass  # Tokens expired  fall through to fresh login

        #  2. Fresh login with optional MFA 
        mfa_trimmed = mfa_code.strip()

        def _prompt_mfa() -> str:
            if mfa_trimmed:
                return mfa_trimmed
            raise RuntimeError("MFA_REQUIRED")

        try:
            self.client = Garmin(email=email, password=password, prompt_mfa=_prompt_mfa)
            self.client.login()
            # Save tokens so next login skips password entirely
            try:
                self.client.garth.dump(TOKENSTORE)
            except Exception:
                pass
            self._connected = True
            display = self.client.get_full_name() or email
            return True, f"Connected as {display}"
        except RuntimeError as exc:
            if "MFA_REQUIRED" in str(exc):
                self._connected = False
                return False, "MFA_REQUIRED"
            self._connected = False
            return False, f"Connection error: {exc}"
        except GarminConnectAuthenticationError:
            self._connected = False
            return False, "Authentication failed  check your Garmin email and password."
        except Exception as exc:
            self._connected = False
            return False, f"Connection error: {exc}"

    @staticmethod
    def clear_token_cache() -> None:
        """Delete saved OAuth tokens (forces fresh login next time)."""
        if os.path.exists(TOKENSTORE):
            import shutil
            shutil.rmtree(TOKENSTORE, ignore_errors=True)
            if os.path.isfile(TOKENSTORE):
                os.remove(TOKENSTORE)

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------
    def get_recent_activities(self, days: int = 7) -> list[dict]:
        """Return parsed activity dicts for the past `days` days."""
        if not self._connected or not self.client:
            return []

        try:
            raw = self.client.get_activities_by_date(
                startdate=(datetime.date.today() - datetime.timedelta(days=days)).isoformat(),
                enddate=datetime.date.today().isoformat(),
            )
        except Exception as exc:
            print(f"[garmin] get_activities_by_date error: {exc}")
            return []

        results = []
        for act in raw:
            garmin_type = (
                act.get("activityType", {}).get("typeKey", "other")
                if isinstance(act.get("activityType"), dict)
                else str(act.get("activityType", "other"))
            )
            primary_el, secondary_el = _map_activity_type(garmin_type)
            avg_hr = act.get("averageHR") or act.get("avgHr")
            duration_sec = act.get("duration") or act.get("elapsedDuration")

            results.append({
                "activity_id":       str(act.get("activityId", "")),
                "activity_date":     act.get("startTimeLocal", "")[:10],
                "activity_name":     act.get("activityName", garmin_type.replace("_", " ").title()),
                "activity_type":     garmin_type,
                "primary_element":   primary_el,
                "secondary_element": secondary_el,
                "duration_minutes":  _parse_duration(duration_sec),
                "distance_km":       round((act.get("distance") or 0) / 1000, 2),
                "avg_hr":            int(avg_hr) if avg_hr else None,
                "max_hr":            int(act.get("maxHR") or act.get("maxHr") or 0) or None,
                "calories":          int(act.get("calories") or 0) or None,
                "intensity":         _intensity_from_hr(avg_hr),
            })

        return results

    def enrich_activities_with_zones(self, activities: list[dict],
                                      custom_zones: dict) -> list[dict]:
        """
        For each activity that has an activity_id and avg_hr but no zone data yet,
        fetch actual time-in-zone from Garmin and add z1_minutes–z5_minutes.

        Makes one extra API call per activity — call this during manual re-sync only.
        Returns the same list with zone data added in-place.
        """
        if not self._connected or not self.client:
            return activities

        max_hr = custom_zones.get("max_hr") or 190

        enriched = 0
        for act in activities:
            act_id = act.get("activity_id", "")
            if not act_id or act_id.startswith("csv_"):
                continue  # CSV imports have no Garmin activity ID
            if act.get("z1_minutes") is not None:
                continue  # Already have zone data

            try:
                raw_zones = self.client.get_activity_hr_in_timezones(int(act_id))
                print(f"[garmin] raw zone data for {act_id}: {raw_zones}")
                if raw_zones:
                    zone_mins = redistribute_garmin_zones_to_custom(
                        raw_zones, custom_zones, max_hr=max_hr
                    )
                    if zone_mins and any(v > 0 for v in zone_mins.values()):
                        act.update(zone_mins)
                        enriched += 1
                        print(f"[garmin] ✓ zone enriched {act.get('activity_name','?')}: {zone_mins}")
                    else:
                        print(f"[garmin] zone redistribution returned zeros for {act_id} — using avg HR fallback")
                else:
                    print(f"[garmin] empty zone response for {act_id}")
            except Exception as exc:
                print(f"[garmin] zone fetch error for {act_id}: {exc}")

        print(f"[garmin] zone enrichment complete: {enriched}/{len(activities)} activities enriched")

        return activities

    # ------------------------------------------------------------------
    # Daily stats
    # ------------------------------------------------------------------
    def get_today_stats(self) -> dict:
        """Return today's steps, sleep, resting HR, stress, and body battery."""
        if not self._connected or not self.client:
            return {}

        today = datetime.date.today().isoformat()
        stats: dict = {"date": today}

        # Steps
        try:
            steps_data = self.client.get_steps_data(today)
            if steps_data:
                total = sum(s.get("steps", 0) for s in steps_data if isinstance(s, dict))
                stats["steps"] = total
        except Exception as exc:
            print(f"[garmin] steps error: {exc}")

        # Sleep (previous night).
        # Garmin sometimes reports last night's sleep under today's date, sometimes yesterday's.
        # Try today first; fall back to yesterday if no sleep data found.
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        for _sleep_date in [today, yesterday]:
            try:
                sleep = self.client.get_sleep_data(_sleep_date)
                if not sleep:
                    continue
                daily_sleep = sleep.get("dailySleepDTO", sleep)
                secs = daily_sleep.get("sleepTimeSeconds") or daily_sleep.get("sleepDuration")
                if not secs:
                    continue
                stats["sleep_hours"] = round(secs / 3600, 1)
                scores = daily_sleep.get("sleepScores", {})
                stats["sleep_score"] = (scores.get("overall", {}).get("value")
                                        if isinstance(scores, dict) else daily_sleep.get("sleepScore"))
                # Sleep stages (seconds → minutes)
                for key, field in [
                    ("deepSleepSeconds",  "sleep_deep_minutes"),
                    ("remSleepSeconds",   "sleep_rem_minutes"),
                    ("lightSleepSeconds", "sleep_light_minutes"),
                    ("awakeSleepSeconds", "sleep_awake_minutes"),
                ]:
                    val = daily_sleep.get(key)
                    if val:
                        stats[field] = round(val / 60)
                print(f"[garmin] sleep data fetched for {_sleep_date}: {stats.get('sleep_hours')}h")
                break  # got sleep data, stop trying
            except Exception as exc:
                print(f"[garmin] sleep error ({_sleep_date}): {exc}")

        # Heart rate / resting HR — try today then yesterday
        for _hr_date in [today, yesterday]:
            try:
                hr = self.client.get_heart_rates(_hr_date)
                if hr and hr.get("restingHeartRate"):
                    stats["resting_hr"] = hr.get("restingHeartRate")
                    print(f"[garmin] resting HR {stats['resting_hr']} bpm from {_hr_date}")
                    break
            except Exception as exc:
                print(f"[garmin] hr error ({_hr_date}): {exc}")

        # Stress — try today then yesterday
        for _stress_date in [today, yesterday]:
            try:
                stress = self.client.get_stress_data(_stress_date)
                if stress:
                    val = stress.get("avgStressLevel") or stress.get("averageStressLevel")
                    if val and val > 0:
                        stats["avg_stress"] = int(val)
                        break
            except Exception as exc:
                print(f"[garmin] stress error ({_stress_date}): {exc}")

        # Body battery — try today then yesterday
        for _bb_date in [today, yesterday]:
            try:
                bb = self.client.get_body_battery(_bb_date)
                if bb and isinstance(bb, list) and bb:
                    charged = [e.get("charged") for e in bb if isinstance(e, dict) and "charged" in e]
                    if charged:
                        stats["body_battery"] = max(charged)
                        stats["body_battery_morning"] = charged[0]
                        print(f"[garmin] body battery from {_bb_date}: morning={charged[0]}, max={max(charged)}")
                        break
            except Exception as exc:
                print(f"[garmin] body battery error ({_bb_date}): {exc}")

        # Calories (daily summary)
        try:
            summary = self.client.get_daily_summary_for_range(today, today)
            if summary and isinstance(summary, list) and summary:
                s = summary[0]
                stats["total_calories"] = s.get("totalKilocalories")
                stats["active_calories"] = s.get("activeKilocalories")
        except Exception as exc:
            print(f"[garmin] daily summary error: {exc}")

        return stats

    def get_comprehensive_health_data(self) -> dict:
        """
        Pull all available health data from Garmin in one comprehensive call.
        This is expensive (makes ~15+ API calls) but gives maximum context.
        Call once per sync session, not every recommendation.
        """
        if not self._connected or not self.client:
            return {}

        today = datetime.date.today().isoformat()
        data = {"pulled_at": today}

        # 
        # TRAINING & FITNESS STATUS
        # 
        try:
            readiness = self.client.get_morning_training_readiness()
            if readiness:
                data["training_readiness"] = {
                    "score": readiness.get("readinessScore"),  # 0-100
                    "status": readiness.get("readinessStatus"),  # e.g. "ready", "low", "high"
                    "factors": readiness.get("factors", {}),  # Training load, HRV, sleep, etc.
                }
                print(f"  [garmin] Training readiness: {readiness.get('readinessScore', '?')}")
        except Exception as e:
            print(f"  [garmin] training_readiness error: {e}")

        try:
            train_status = self.client.get_training_status()
            if train_status:
                data["training_status"] = {
                    "load": train_status.get("trainingLoadFocus"),  # e.g. "balanced", "unbalanced", "overreaching"
                    "training_load": train_status.get("trainingLoadFocusValue"),
                    "stress_balance": train_status.get("stressBalance"),
                }
                print(f"  [garmin] Training status: {train_status.get('trainingLoadFocus', '?')}")
        except Exception as e:
            print(f"  [garmin] training_status error: {e}")

        try:
            endurance = self.client.get_endurance_score()
            if endurance:
                data["endurance_score"] = endurance.get("score")  # Aerobic fitness
                print(f"  [garmin] Endurance score: {endurance.get('score', '?')}")
        except Exception as e:
            print(f"  [garmin] endurance_score error: {e}")

        try:
            max_metrics = self.client.get_max_metrics()
            if max_metrics:
                data["max_metrics"] = {
                    "vo2_max": max_metrics.get("vo2Max"),
                    "max_pace": max_metrics.get("maxRunningPace"),
                    "max_hr": max_metrics.get("maxHeartRate"),
                    "max_power": max_metrics.get("maxPower"),  # Cycling
                }
                print(f"  [garmin] VO2 Max: {max_metrics.get('vo2Max', '?')}")
        except Exception as e:
            print(f"  [garmin] max_metrics error: {e}")

        # 
        # HEART RATE VARIABILITY & STRESS RESILIENCE
        # 
        try:
            hrv = self.client.get_hrv_data(today)
            if hrv:
                data["hrv"] = {
                    "last_night_avg": hrv.get("lastNightAverage"),  # ms (higher = better recovery)
                    "last_night_5min": hrv.get("lastNight5MinHigh"),
                    "last_night_status": hrv.get("status"),  # "balanced", "low", "high"
                    "trend": hrv.get("trend"),  # "balanced", "improving", "declining"
                }
                print(f"  [garmin] HRV: {hrv.get('lastNightAverage', '?')}ms ({hrv.get('status', '?')})")
        except Exception as e:
            print(f"  [garmin] hrv error: {e}")

        try:
            respiration = self.client.get_respiration_data(today)
            if respiration:
                data["respiration"] = {
                    "avg_breathing_rate": respiration.get("avgBreathingRate"),  # breaths/min (12-20 normal)
                    "status": respiration.get("status"),  # elevated = stress/illness
                }
                print(f"  [garmin] Respiration: {respiration.get('avgBreathingRate', '?')} breaths/min")
        except Exception as e:
            print(f"  [garmin] respiration error: {e}")

        # 
        # BODY COMPOSITION & WEIGHT
        # 
        try:
            body_comp = self.client.get_body_composition()
            if body_comp and isinstance(body_comp, list) and body_comp:
                latest = body_comp[0]  # Most recent
                data["body_composition"] = {
                    "weight_kg": latest.get("weight"),
                    "body_fat_percent": latest.get("bodyFatPercent"),
                    "muscle_percent": latest.get("musclePercent"),
                    "bone_density": latest.get("boneDensity"),
                    "measured_date": latest.get("date"),
                }
                print(f"  [garmin] Body comp: {latest.get('weight')}kg, {latest.get('bodyFatPercent', '?')}% fat")
        except Exception as e:
            print(f"  [garmin] body_composition error: {e}")

        # 
        # WEEKLY SUMMARIES & TRAINING LOAD
        # 
        try:
            # get_intensity_minutes_data returns the TRUE weekly rolling totals
            # (weeklyModerate, weeklyVigorous, weeklyTotal) — much more accurate
            # than get_user_summary which only returns today's minutes.
            im_data = self.client.get_intensity_minutes_data(today)
            if im_data:
                data["weekly_summary"] = {
                    "intensity_minutes":          im_data.get("weeklyTotal"),
                    "moderate_intensity_minutes": im_data.get("weeklyModerate"),
                    "vigorous_intensity_minutes": im_data.get("weeklyVigorous"),
                    "week_goal":                  im_data.get("weekGoal"),
                    "day_of_goal_met":            im_data.get("dayOfGoalMet"),
                }
                print(f"  [garmin] Weekly intensity: mod={im_data.get('weeklyModerate')} "
                      f"vig={im_data.get('weeklyVigorous')} total={im_data.get('weeklyTotal')}")
        except Exception as e:
            print(f"  [garmin] intensity_minutes error: {e}")

        # 
        # PERSONAL RECORDS & PROGRESS
        # 
        try:
            prs = self.client.get_personal_record()
            if prs:
                data["personal_records"] = prs.get("personalBests", {})  # Dict of activity -> best time/distance
                print(f"  [garmin] Loaded {len(prs.get('personalBests', {}))} personal records")
        except Exception as e:
            print(f"  [garmin] personal_record error: {e}")

        # 
        # ADDITIONAL OPTIONAL DATA
        # 
        try:
            blood_pressure = self.client.get_blood_pressure()
            if blood_pressure and isinstance(blood_pressure, list) and blood_pressure:
                latest_bp = blood_pressure[0]
                data["blood_pressure"] = {
                    "systolic": latest_bp.get("systolic"),
                    "diastolic": latest_bp.get("diastolic"),
                    "measured_date": latest_bp.get("date"),
                }
        except Exception:
            pass  # Not everyone has a BP cuff

        try:
            hydration = self.client.get_hydration_data(today)
            if hydration:
                data["hydration"] = hydration.get("value")  # mL of water consumed
        except Exception:
            pass

        try:
            devices = self.client.get_devices()
            if devices:
                data["devices"] = [
                    {
                        "model": d.get("displayName"),
                        "serial": d.get("serialNumber"),
                        "battery": d.get("batteryLevel"),
                    }
                    for d in devices if isinstance(devices, list)
                ]
                print(f"  [garmin] {len(devices)} devices connected")
        except Exception:
            pass

        try:
            goals = self.client.get_goals()
            if goals:
                data["goals"] = goals
                print(f"  [garmin] Loaded {len(goals)} goals")
        except Exception:
            pass

        print(f"\n  [garmin] Comprehensive pull complete: {len(data)} data groups")
        return data

    # ------------------------------------------------------------------
    # Human-readable summary for the recommender prompt
    # ------------------------------------------------------------------
    @staticmethod
    def format_summary(activities: list[dict], today_stats: dict,
                       stats_age_days: int | None = None,
                       comprehensive: dict | None = None) -> str:
        """
        Format Garmin data as a clinical-grade health context string for LLM consumption.
        Covers sleep architecture, HRV, respiration, body battery, stress, training load,
        and recent activity patterns — everything needed for morning wellbeing assessment.
        """
        lines = ["=== GARMIN HEALTH DATA ==="]

        # ── YESTERDAY'S RECOVERY DATA ──────────────────────────────────────────
        # Check if stats have any meaningful values (not all null)
        _has_any_stats = today_stats and any(
            today_stats.get(k) for k in (
                "sleep_hours", "sleep_score", "resting_hr", "body_battery",
                "avg_stress", "body_battery_morning", "steps"
            )
        )
        if today_stats and stats_age_days is not None and _has_any_stats:
            stat_date = today_stats.get("stat_date", "")
            date_label = "Today" if stats_age_days == 0 else "Yesterday" if stats_age_days == 1 else f"{stats_age_days} days ago ({stat_date})"
            lines.append(f"\n[DAILY SNAPSHOT — {date_label}]")

            # Body battery (morning waking value is most diagnostically meaningful)
            bb_morning = today_stats.get("body_battery_morning")
            bb_max = today_stats.get("body_battery")
            if bb_morning is not None:
                bb_status = "good" if bb_morning >= 70 else "moderate" if bb_morning >= 50 else "low — high fatigue risk"
                lines.append(f"  Body battery at wake: {bb_morning}% ({bb_status})")
                if bb_max and bb_max != bb_morning:
                    lines.append(f"  Body battery daily max: {bb_max}%")
            elif bb_max is not None:
                bb_status = "good" if bb_max >= 70 else "moderate" if bb_max >= 50 else "low — high fatigue risk"
                lines.append(f"  Body battery: {bb_max}% ({bb_status})")

            # Sleep architecture
            sleep_h = today_stats.get("sleep_hours")
            sleep_score = today_stats.get("sleep_score")
            deep_min = today_stats.get("sleep_deep_minutes")
            rem_min  = today_stats.get("sleep_rem_minutes")
            light_min = today_stats.get("sleep_light_minutes")
            awake_min = today_stats.get("sleep_awake_minutes")

            if sleep_h:
                sleep_status = "optimal" if sleep_h >= 7.5 else "adequate" if sleep_h >= 6.5 else "insufficient — cognitive and recovery impact expected"
                lines.append(f"  Sleep duration: {sleep_h}h ({sleep_status})")
            if sleep_score:
                score_label = "good" if sleep_score >= 75 else "fair" if sleep_score >= 60 else "poor"
                lines.append(f"  Sleep score: {sleep_score}/100 ({score_label})")

            if deep_min is not None or rem_min is not None:
                lines.append("  Sleep stages:")
                total_sleep_min = sleep_h * 60 if sleep_h else None
                if deep_min is not None:
                    deep_pct = f" ({round(deep_min/total_sleep_min*100)}% of night)" if total_sleep_min else ""
                    deep_status = "good" if deep_min >= 60 else "low — physical repair may be compromised"
                    lines.append(f"    Deep (N3): {deep_min} min{deep_pct} — {deep_status}")
                if rem_min is not None:
                    rem_pct = f" ({round(rem_min/total_sleep_min*100)}% of night)" if total_sleep_min else ""
                    rem_status = "good" if rem_min >= 60 else "low — emotional regulation and memory consolidation affected"
                    lines.append(f"    REM: {rem_min} min{rem_pct} — {rem_status}")
                if light_min is not None:
                    lines.append(f"    Light (N1/N2): {light_min} min")
                if awake_min is not None:
                    awake_status = "normal" if awake_min <= 20 else "elevated — sleep fragmentation"
                    lines.append(f"    Awake: {awake_min} min ({awake_status})")

            # Resting HR
            rhr = today_stats.get("resting_hr")
            if rhr:
                rhr_status = "excellent" if rhr < 55 else "good" if rhr < 65 else "elevated — possible incomplete recovery or stress"
                lines.append(f"  Resting HR: {rhr} bpm ({rhr_status})")

            # Stress
            stress = today_stats.get("avg_stress")
            if stress:
                stress_status = "low — good parasympathetic tone" if stress < 25 else "moderate" if stress < 50 else "high — sympathetic dominance, recovery impaired"
                lines.append(f"  Avg stress: {stress}/100 ({stress_status})")

            # Steps
            steps = today_stats.get("steps")
            if steps:
                lines.append(f"  Steps: {steps:,}")

        elif not today_stats or not _has_any_stats:
            lines.append("\n[DAILY SNAPSHOT] Daily health metrics not yet available — "
                         "tap 🔄 Re-sync Garmin to pull latest data. "
                         "(Garmin sometimes takes 15–30 min after device sync to process sleep/HRV data.)")

        # ── HRV & AUTONOMIC NERVOUS SYSTEM ─────────────────────────────────────
        if comprehensive:
            hrv = comprehensive.get("hrv", {})
            hrv_avg = hrv.get("last_night_avg")
            hrv_5min = hrv.get("last_night_5min")
            hrv_status = hrv.get("status", "")
            hrv_trend = hrv.get("trend", "")

            if hrv_avg:
                lines.append(f"\n[HRV — Autonomic Nervous System]")
                status_label = {
                    "balanced": "balanced — ANS well-regulated, good readiness",
                    "low":      "low — ANS under stress, prioritise recovery",
                    "unbalanced": "unbalanced — elevated sympathetic tone",
                }.get(hrv_status, hrv_status or "")
                lines.append(f"  Last night avg HRV: {hrv_avg} ms ({status_label})")
                if hrv_5min:
                    lines.append(f"  Peak 5-min HRV: {hrv_5min} ms")
                if hrv_trend:
                    trend_note = {
                        "balanced":  "stable — no change in ANS load",
                        "improving": "improving — recovery trend positive",
                        "declining": "declining — cumulative stress or fatigue building",
                    }.get(hrv_trend, hrv_trend)
                    lines.append(f"  HRV trend: {trend_note}")

            # ── RESPIRATION ───────────────────────────────────────────────────
            resp = comprehensive.get("respiration", {})
            rr = resp.get("avg_breathing_rate")
            if rr:
                lines.append(f"\n[RESPIRATION]")
                if rr <= 14:
                    rr_note = "excellent — calm, efficient breathing"
                elif rr <= 18:
                    rr_note = "normal"
                elif rr <= 20:
                    rr_note = "slightly elevated — monitor for stress or overtraining"
                else:
                    rr_note = "elevated — possible illness, stress, or respiratory strain"
                lines.append(f"  Avg breathing rate: {rr} breaths/min ({rr_note})")
                if resp.get("status") and resp["status"] != "normal":
                    lines.append(f"  Status flag: {resp['status']}")

            # ── TRAINING READINESS ────────────────────────────────────────────
            readiness = comprehensive.get("training_readiness", {})
            r_score = readiness.get("score")
            if r_score is not None:
                lines.append(f"\n[TRAINING READINESS]")
                r_label = "high — ready for quality training" if r_score >= 70 else \
                          "moderate — light/moderate session advisable" if r_score >= 50 else \
                          "low — rest or very light activity only"
                lines.append(f"  Score: {r_score}/100 ({r_label})")
                factors = readiness.get("factors", {})
                if factors:
                    lines.append(f"  Contributing factors: {factors}")

            # ── TRAINING STATUS & LOAD ────────────────────────────────────────
            train_status = comprehensive.get("training_status", {})
            if train_status.get("load"):
                lines.append(f"\n[TRAINING LOAD]")
                lines.append(f"  Status: {train_status['load']}")
                sb = train_status.get("stress_balance")
                if sb:
                    lines.append(f"  Stress balance: {sb}")

            # ── FITNESS METRICS ───────────────────────────────────────────────
            vo2 = comprehensive.get("max_metrics", {}).get("vo2_max")
            endurance = comprehensive.get("endurance_score")
            if vo2 or endurance:
                lines.append(f"\n[FITNESS METRICS]")
                if vo2:
                    lines.append(f"  VO2 Max: {vo2} mL/kg/min")
                if endurance:
                    lines.append(f"  Endurance score: {endurance}")

            # ── WEEKLY INTENSITY MINUTES (from get_intensity_minutes_data) ──────
            weekly = comprehensive.get("weekly_summary", {})
            wk_total = weekly.get("intensity_minutes")
            wk_mod   = weekly.get("moderate_intensity_minutes")
            wk_vig   = weekly.get("vigorous_intensity_minutes")
            wk_goal  = weekly.get("week_goal") or 300
            if wk_total:
                lines.append(f"\n[GARMIN WEEKLY INTENSITY MINUTES]")
                goal_note = ("✅ met" if wk_total >= wk_goal
                             else f"⚠ {wk_goal - wk_total} min below weekly goal ({wk_goal} min)")
                lines.append(f"  Weekly total (Garmin-tracked): {wk_total} min — {goal_note}")
                if wk_mod is not None:
                    lines.append(f"  Moderate: {wk_mod} min")
                if wk_vig is not None:
                    lines.append(f"  Vigorous: {wk_vig} min")

            # ── HYDRATION ─────────────────────────────────────────────────────
            hydration = comprehensive.get("hydration_ml")
            if hydration:
                lines.append(f"\n[HYDRATION] Yesterday: {hydration} mL consumed")

        # ── RECENT ACTIVITIES ─────────────────────────────────────────────────
        if activities:
            lines.append(f"\n[RECENT ACTIVITIES — last {len(activities)} sessions]")
            for act in activities:
                duration = f"{act['duration_minutes']}min" if act.get("duration_minutes") else ""
                hr_str = f" | avg HR {act['avg_hr']} bpm" if act.get("avg_hr") else ""
                dist = f" | {act['distance_km']} km" if act.get("distance_km") else ""
                lines.append(
                    f"  {act['activity_date']} | {act['activity_name']}"
                    f" | {duration}{dist}{hr_str} [{act.get('intensity','?')}]"
                )

            # Training load pattern
            from collections import Counter
            intensities = [a.get("intensity", "") for a in activities]
            high_n = intensities.count("high")
            mod_n  = intensities.count("moderate")
            light_n = intensities.count("light")
            lines.append(f"\n  Load pattern: {high_n}x high / {mod_n}x moderate / {light_n}x light")
            if high_n >= 3 and high_n > light_n + mod_n:
                lines.append("  ⚠ Heavy high-intensity load — monitor for overreaching")

        return "\n".join(lines)


# =============================================================================
# CSV IMPORT (Garmin Connect website export)
# =============================================================================
# Column names as exported by Garmin Connect  Activities  Export CSV
_CSV_COL_TYPE     = "Activity Type"
_CSV_COL_DATE     = "Date"
_CSV_COL_TITLE    = "Title"
_CSV_COL_DIST     = "Distance"
_CSV_COL_CALORIES = "Calories"
_CSV_COL_TIME     = "Time"
_CSV_COL_AVG_HR   = "Avg HR"
_CSV_COL_MAX_HR   = "Max HR"


def _parse_hhmmss(time_str: str) -> int:
    """Convert HH:MM:SS or MM:SS to total minutes."""
    if not time_str or time_str.strip() == "--":
        return 0
    parts = time_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 2:
            return int(parts[0])
    except ValueError:
        pass
    return 0


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_int(val: str) -> Optional[int]:
    f = _safe_float(val)
    return int(f) if f is not None else None


def parse_garmin_csv(csv_bytes: bytes) -> tuple[list[dict], str]:
    """
    Parse a Garmin Connect Activities CSV export.

    Returns (activities, error_message). On success error_message is "".
    """
    try:
        text = csv_bytes.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        text = csv_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []

    if _CSV_COL_TYPE not in headers and _CSV_COL_DATE not in headers:
        return [], (
            "This doesn't look like a Garmin Connect CSV. "
            "Export from: Garmin Connect  Activities  top-right menu  Export CSV."
        )

    activities = []
    for row in reader:
        raw_type = row.get(_CSV_COL_TYPE, "other").strip()
        raw_date = row.get(_CSV_COL_DATE, "")[:10]  # keep YYYY-MM-DD
        if not raw_date:
            continue

        primary_el, secondary_el = _map_activity_type(raw_type)
        dist_raw = _safe_float(row.get(_CSV_COL_DIST, ""))
        avg_hr   = _safe_int(row.get(_CSV_COL_AVG_HR, ""))
        duration = _parse_hhmmss(row.get(_CSV_COL_TIME, ""))

        activities.append({
            "activity_id":       f"csv_{raw_date}_{raw_type[:8]}_{len(activities)}",
            "activity_date":     raw_date,
            "activity_name":     row.get(_CSV_COL_TITLE, raw_type).strip() or raw_type,
            "activity_type":     raw_type.lower().replace(" ", "_"),
            "primary_element":   primary_el,
            "secondary_element": secondary_el,
            "duration_minutes":  duration,
            "distance_km":       round(dist_raw, 2) if dist_raw else None,
            "avg_hr":            avg_hr,
            "max_hr":            _safe_int(row.get(_CSV_COL_MAX_HR, "")),
            "calories":          _safe_int(row.get(_CSV_COL_CALORIES, "")),
            "intensity":         _intensity_from_hr(avg_hr),
        })

    return activities, ""


# =============================================================================
# PERSONALIZED ZONE CALCULATION
# =============================================================================
def _classify_activity_zone(avg_hr: int, zones: dict) -> tuple[str, str]:
    """
    Classify a single activity into the VO2 Master 5-zone system.
    Returns (zone_label, short_name) e.g. ("Z2 Fat Burn (140–156 bpm)", "Z2")
    """
    z1 = zones.get("zone_1")
    z2 = zones.get("zone_2")
    z3 = zones.get("zone_3")
    z4 = zones.get("zone_4")
    vt1 = zones.get("vt1_bpm")
    vt2 = zones.get("vt2_bpm")

    if z1 and z2 and z3 and z4:
        if avg_hr <= z1[1]:
            return f"Z1 Warm-Up (<{z1[1]} bpm)", "Z1"
        elif avg_hr <= z2[1]:
            return f"Z2 Fat Burn ({z1[1]+1}–{z2[1]} bpm)", "Z2"
        elif avg_hr <= z3[1]:
            return f"Z3 Endurance ({z2[1]+1}–{z3[1]} bpm)", "Z3"
        elif avg_hr <= z4[1]:
            return f"Z4 Vigorous ({z3[1]+1}–{z4[1]} bpm)", "Z4"
        else:
            return f"Z5 Maximal (>{z4[1]} bpm)", "Z5"
    elif vt1 and vt2:
        # Fallback to VT1/VT2
        if avg_hr < vt1:
            return f"Sub-VT1 aerobic (<{vt1} bpm)", "Z1/Z2"
        elif avg_hr <= vt2:
            return f"Threshold (VT1–VT2, {vt1}–{vt2} bpm)", "Z3/Z4"
        else:
            return f"High-intensity (>{vt2} bpm)", "Z5"
    return "Unknown zone", "?"


def redistribute_garmin_zones_to_custom(garmin_zones_raw: list, custom_zones: dict,
                                         max_hr: int = 190) -> dict | None:
    """
    Convert Garmin's per-activity time-in-zone data into our custom VO2 Master zone boundaries.

    Garmin returns zones as:
      [{'zoneNumber': 1, 'secsInZone': 252.0, 'zoneLowBoundary': 95},
       {'zoneNumber': 2, 'secsInZone': 1274.0, 'zoneLowBoundary': 114}, ...]

    The high boundary of each zone = the next zone's low - 1.  Last zone → 999.
    We proportionally redistribute each Garmin bucket into our VO2 Master zones
    by computing BPM overlap and assuming uniform HR distribution within each bucket.

    Returns {z1_minutes, z2_minutes, z3_minutes, z4_minutes, z5_minutes} or None.
    """
    if not garmin_zones_raw:
        return None

    # Sort by zoneNumber (ascending) to guarantee order
    sorted_zones = sorted(garmin_zones_raw, key=lambda z: z.get("zoneNumber", 0))

    # Build (low_bpm, high_bpm, secs) tuples
    # High = next zone's low - 1;  last zone → 999
    parsed = []
    for i, z in enumerate(sorted_zones):
        secs = z.get("secsInZone") or 0
        low_raw = z.get("zoneLowBoundary")
        if low_raw is None:
            continue
        low_bpm = int(low_raw)

        if i + 1 < len(sorted_zones):
            next_low = sorted_zones[i + 1].get("zoneLowBoundary")
            high_bpm = int(next_low) - 1 if next_low is not None else 999
        else:
            high_bpm = 999

        if secs > 0:
            parsed.append({"low": low_bpm, "high": high_bpm, "secs": secs})

    if not parsed:
        return None

    # Custom zone boundaries from VO2 Master
    z1 = custom_zones.get("zone_1") or (0, 139)
    z2 = custom_zones.get("zone_2") or (140, 156)
    z3 = custom_zones.get("zone_3") or (157, 164)
    z4 = custom_zones.get("zone_4") or (165, 173)
    custom = [
        (z1[0], z1[1], "z1"),
        (z2[0], z2[1], "z2"),
        (z3[0], z3[1], "z3"),
        (z4[0], z4[1], "z4"),
        (z4[1] + 1, 999, "z5"),
    ]

    result = {k: 0.0 for k in ("z1", "z2", "z3", "z4", "z5")}

    for g in parsed:
        g_range = g["high"] - g["low"]
        if g_range <= 0:
            continue
        for c_low, c_high, key in custom:
            overlap = min(g["high"], c_high) - max(g["low"], c_low)
            if overlap > 0:
                fraction = overlap / g_range
                result[key] += g["secs"] * fraction / 60.0  # → minutes

    return {f"{k}_minutes": round(v, 1) for k, v in result.items()}


def analyze_training_vs_zones(activities: list[dict], zones: dict) -> str:
    """
    Analyze activities against personalized VO2 Master training zones.

    Uses the actual 5-zone boundaries (Z1–Z5) for per-activity classification.
    Uses VT1/VT2 as the polarized model boundary for the volume assessment:
      - Aerobic base (sub-VT1, Z1+Z2): should be ~70-80% of training
      - Threshold/high (above VT1): the remaining 20-30%
    """
    if not activities or not zones.get("vt1_bpm") or not zones.get("vt2_bpm"):
        return ""

    vt1 = zones["vt1_bpm"]
    vt2 = zones["vt2_bpm"]
    vo2_max = zones.get("vo2_max")

    # Per-activity 5-zone classification
    activity_zone_lines = []
    # Polarized buckets: aerobic (sub-VT1 = Z1+Z2), threshold (VT1–VT2 = Z3+Z4), high (>VT2 = Z5)
    aerobic_minutes = 0    # <VT1 — Z1+Z2
    threshold_minutes = 0  # VT1–VT2 — Z3+Z4
    high_minutes = 0       # >VT2 — Z5
    no_hr_minutes = 0
    total_sessions = 0
    total_minutes = 0

    for act in activities:
        avg_hr   = act.get("avg_hr")
        duration = act.get("duration_minutes", 0)
        date     = act.get("activity_date", "")
        name     = act.get("activity_name", "Activity")
        total_sessions += 1
        total_minutes  += duration

        # ── Use actual time-in-zone if available ────────────────────────
        has_zone_data = act.get("z1_minutes") is not None
        if has_zone_data:
            z1m = act.get("z1_minutes") or 0
            z2m = act.get("z2_minutes") or 0
            z3m = act.get("z3_minutes") or 0
            z4m = act.get("z4_minutes") or 0
            z5m = act.get("z5_minutes") or 0
            sub_vt1 = z1m + z2m          # aerobic (Z1+Z2 = below VT1)
            above_vt1 = z3m + z4m        # threshold (Z3+Z4 = VT1→VT2)
            above_vt2 = z5m              # high-intensity (Z5 = above VT2)
            aerobic_minutes   += sub_vt1
            threshold_minutes += above_vt1
            high_minutes      += above_vt2
            # Build a compact zone breakdown string
            parts = []
            if z1m: parts.append(f"Z1:{z1m:.0f}m")
            if z2m: parts.append(f"Z2:{z2m:.0f}m")
            if z3m: parts.append(f"Z3:{z3m:.0f}m")
            if z4m: parts.append(f"Z4:{z4m:.0f}m")
            if z5m: parts.append(f"Z5:{z5m:.0f}m")
            zone_total = z1m + z2m + z3m + z4m + z5m
            # Any gap between duration and zone total = time below lowest zone floor
            # (HR below Garmin's Zone 1 threshold ~95 bpm, e.g. first 1-3 min of warm-up)
            pre_zone = round(duration - zone_total)
            if pre_zone > 0:
                parts.insert(0, f"pre-zone:{pre_zone}m")
            zone_breakdown = " | ".join(parts) if parts else "no zone split"
            hr_note = f"avg {avg_hr} bpm" if avg_hr else ""
            activity_zone_lines.append(
                f"  {date} | {name} | {duration}min | {hr_note} | actual zones: {zone_breakdown}"
            )
        # ── Fall back to avg HR estimate ────────────────────────────────
        elif avg_hr:
            zone_label, zone_short = _classify_activity_zone(avg_hr, zones)
            activity_zone_lines.append(
                f"  {date} | {name} | {duration}min | avg {avg_hr} bpm → {zone_label} (estimated — re-sync for actual zones)"
            )
            if avg_hr < vt1:
                aerobic_minutes += duration
            elif avg_hr <= vt2:
                threshold_minutes += duration
            else:
                high_minutes += duration
        else:
            activity_zone_lines.append(
                f"  {date} | {name} | {duration}min | no HR data"
            )
            no_hr_minutes += duration

    hr_minutes = aerobic_minutes + threshold_minutes + high_minutes
    if hr_minutes == 0:
        return ""

    aerobic_pct   = aerobic_minutes / hr_minutes * 100
    threshold_pct = threshold_minutes / hr_minutes * 100
    high_pct      = high_minutes / hr_minutes * 100
    aerobic_hrs   = aerobic_minutes / 60
    threshold_hrs = threshold_minutes / 60

    # Build assessment block
    vo2_pct   = zones.get("vo2_pct")
    max_hr    = zones.get("max_hr")
    vt1_power = zones.get("vt1_power")
    vt2_power = zones.get("vt2_power")

    assessment = f"\n[ZONE ANALYSIS — VO2 Master GXT 15 Oct 2025]\n"

    # Context header
    if vo2_max:
        pct_label = (f" ({int(vo2_pct)}th percentile — Very Poor for 42yo male)" if vo2_pct and vo2_pct <= 10
                     else f" ({int(vo2_pct)}th percentile)" if vo2_pct else "")
        assessment += f"VO2max: {vo2_max} mL/kg/min{pct_label}\n"
        assessment += f"  Context: For 42yo male — >40.0=Good, >45.0=Excellent, <31.5=Very Poor\n"
        assessment += f"  Improvement potential: VO2max is highly trainable; +10-25% with structured training\n"
    if max_hr:
        assessment += f"Measured max HR: {max_hr} bpm (lab-verified)\n"
    if vt1_power and vt2_power:
        assessment += f"Threshold power: VT1={vt1_power}W | VT2={vt2_power}W | Max={zones.get('max_power','?')}W\n"

    # 5-zone legend
    z1 = zones.get("zone_1", (0, vt1 - 1))
    z2 = zones.get("zone_2", (vt1, vt2))
    z3 = zones.get("zone_3")
    z4 = zones.get("zone_4")
    if z3 and z4:
        assessment += (
            f"5-Zone system: Z1 Warm-Up <{z1[1]} | Z2 Fat Burn {z1[1]+1}–{z2[1]} | "
            f"Z3 Endurance {z2[1]+1}–{z3[1]} | Z4 Vigorous {z3[1]+1}–{z4[1]} | Z5 Maximal >{z4[1]} bpm\n"
        )

    # Per-activity log
    assessment += f"\nPer-activity zone classification:\n"
    assessment += "\n".join(activity_zone_lines) + "\n"

    # ── Per-zone breakdown: which activities contributed to each zone ─────────
    # Build contribution lists for every zone so the LLM can cite sources
    z1_contrib: list[str] = []
    z2_contrib: list[str] = []
    z3_contrib: list[str] = []
    z4_contrib: list[str] = []
    z5_contrib: list[str] = []

    z1 = zones.get("zone_1", (0, vt1 - 1))
    z2 = zones.get("zone_2", (z1[1] + 1, vt1))

    for act in activities:
        _dur  = act.get("duration_minutes", 0)
        _name = act.get("activity_name", "Activity")
        _date = act.get("activity_date", "")
        _avg  = act.get("avg_hr")
        _has  = act.get("z1_minutes") is not None

        if _has:
            # Actual zone data
            _z1m = act.get("z1_minutes") or 0
            _z2m = act.get("z2_minutes") or 0
            _z3m = act.get("z3_minutes") or 0
            _z4m = act.get("z4_minutes") or 0
            _z5m = act.get("z5_minutes") or 0
            if _z1m: z1_contrib.append(f"{_name} ({_date}): {_z1m:.0f} min")
            if _z2m: z2_contrib.append(f"{_name} ({_date}): {_z2m:.0f} min")
            if _z3m: z3_contrib.append(f"{_name} ({_date}): {_z3m:.0f} min")
            if _z4m: z4_contrib.append(f"{_name} ({_date}): {_z4m:.0f} min")
            if _z5m: z5_contrib.append(f"{_name} ({_date}): {_z5m:.0f} min")
        elif _avg:
            # Avg-HR estimate — assign full duration to one zone
            if _avg <= z1[1]:
                z1_contrib.append(f"{_name} ({_date}): ~{_dur} min (avg {_avg} bpm, est.)")
            elif _avg <= z2[1]:
                z2_contrib.append(f"{_name} ({_date}): ~{_dur} min (avg {_avg} bpm, est.)")
            elif _avg <= (zones.get("zone_3") or (z2[1]+1, vt2))[1]:
                z3_contrib.append(f"{_name} ({_date}): ~{_dur} min (avg {_avg} bpm, est.)")
            elif _avg <= vt2:
                z4_contrib.append(f"{_name} ({_date}): ~{_dur} min (avg {_avg} bpm, est.)")
            else:
                z5_contrib.append(f"{_name} ({_date}): ~{_dur} min (avg {_avg} bpm, est.)")

    assessment += "\nZone-by-zone breakdown (which activities contributed):\n"
    _has_any_actual = any(a.get("z1_minutes") is not None for a in activities)
    if not _has_any_actual:
        assessment += "  ⚠ All values estimated from avg HR (one zone per activity). Re-sync for actual per-minute zone splits.\n"
        assessment += "  ⚠ NOTE: pickleball/sports sessions may have Z2 peaks during rallies that are hidden in the avg HR.\n"

    def _zone_line(label: str, total_min: float, contrib: list[str]) -> str:
        total_h = total_min / 60
        lines_out = [f"  {label}: {total_min:.0f} min ({total_h:.1f} hrs)"]
        for c in contrib:
            lines_out.append(f"    → {c}")
        if not contrib:
            lines_out.append(f"    → no contributions this period")
        return "\n".join(lines_out)

    z_total_1 = sum(
        (a.get("z1_minutes") or 0) if a.get("z1_minutes") is not None
        else (a.get("duration_minutes", 0) if (a.get("avg_hr") or 0) <= z1[1] else 0)
        for a in activities
    )
    z_total_2 = sum(
        (a.get("z2_minutes") or 0) if a.get("z1_minutes") is not None
        else (a.get("duration_minutes", 0) if z1[1] < (a.get("avg_hr") or 0) <= z2[1] else 0)
        for a in activities
    )
    z_total_3 = sum(
        (a.get("z3_minutes") or 0) if a.get("z1_minutes") is not None
        else (a.get("duration_minutes", 0)
              if z2[1] < (a.get("avg_hr") or 0) <= (zones.get("zone_3") or (0, vt2))[1] else 0)
        for a in activities
    )
    z_total_4 = sum(
        (a.get("z4_minutes") or 0) if a.get("z1_minutes") is not None
        else (a.get("duration_minutes", 0)
              if (zones.get("zone_3") or (0, vt2))[1] < (a.get("avg_hr") or 0) <= vt2 else 0)
        for a in activities
    )
    z_total_5 = sum(
        (a.get("z5_minutes") or 0) if a.get("z1_minutes") is not None
        else (a.get("duration_minutes", 0) if (a.get("avg_hr") or 0) > vt2 else 0)
        for a in activities
    )

    z3_label = f"zone_3" ; z3_tup = zones.get("zone_3")
    z4_tup = zones.get("zone_4")
    z3_range = f"{z2[1]+1}–{z3_tup[1]}" if z3_tup else f">{z2[1]}"
    z4_range = f"{z3_tup[1]+1}–{z4_tup[1]}" if z3_tup and z4_tup else f">{z2[1]}"

    assessment += _zone_line(f"Z1 Warm-Up        (<{z1[1]} bpm)", z_total_1, z1_contrib) + "\n"
    assessment += _zone_line(f"Z2 Fat Burn       ({z1[1]+1}–{z2[1]} bpm)", z_total_2, z2_contrib) + "\n"
    assessment += _zone_line(f"Z3 Endurance      ({z3_range} bpm)", z_total_3, z3_contrib) + "\n"
    assessment += _zone_line(f"Z4 Vigorous       ({z4_range} bpm)", z_total_4, z4_contrib) + "\n"
    assessment += _zone_line(f"Z5 Maximal        (>{vt2} bpm)", z_total_5, z5_contrib) + "\n"

    # Polarized summary (uses VT1/VT2 as the two key thresholds)
    _actual_zone_count = sum(1 for a in activities if a.get("z1_minutes") is not None)
    _data_quality = (
        f"actual per-second zone data ({_actual_zone_count}/{total_sessions} activities)"
        if _actual_zone_count == total_sessions
        else f"ESTIMATED from avg HR — {total_sessions - _actual_zone_count}/{total_sessions} activities missing zone data. "
             f"Tap 🔄 Re-sync Garmin to fetch actual time-in-zone from Garmin servers."
        if _actual_zone_count < total_sessions
        else "avg HR estimate"
    )
    assessment += f"\nPolarized training summary (VT1={vt1} / VT2={vt2} bpm boundaries):\n"
    assessment += f"  ⚠ Data quality: {_data_quality}\n"
    assessment += (
        f"  Aerobic/easy (Z1+Z2, <{vt1} bpm):   {aerobic_minutes} min ({aerobic_pct:.0f}%) = {aerobic_hrs:.1f} hrs\n"
        f"  Threshold (Z3+Z4, {vt1}–{vt2} bpm): {threshold_minutes} min ({threshold_pct:.0f}%) = {threshold_hrs:.1f} hrs\n"
        f"  High-intensity (Z5, >{vt2} bpm):     {high_minutes} min ({high_pct:.0f}%)\n"
        f"  Total training (HR-tracked):         {hr_minutes} min ({hr_minutes/60:.1f} hrs)\n"
    )
    if no_hr_minutes:
        assessment += f"  No HR data: {no_hr_minutes} min (excluded from zone analysis)\n"

    # ── Per-zone role guide + personalised prescription ──────────────────────
    # Targets are evidence-based for VO2max improvement from a low baseline.
    # Polarized model: 80% sub-VT1 (Z1+Z2) / 15% threshold (Z3+Z4) / 5% maximal (Z5)

    Z2_TARGET_MIN  = 180   # 3 hrs — minimum weekly Z2 for aerobic adaptation
    Z2_TARGET_MAX  = 240   # 4 hrs — upper end before diminishing returns per week
    Z34_TARGET_MIN = 45    # 45 min threshold work to stimulate VO2max
    Z34_TARGET_MAX = 75    # 75 min upper end — beyond this, recovery debt accumulates
    Z5_TARGET_MIN  = 0     # none required until aerobic base is established
    Z5_TARGET_MAX  = 20    # 20 min max — high reward but high injury/overtraining risk

    z_total_2_f = float(z_total_2)
    z_total_34  = float(z_total_3) + float(z_total_4)
    z_total_5_f = float(z_total_5)

    def _status(val, lo, hi):
        if val < lo * 0.5: return "CRITICALLY LOW"
        if val < lo:       return "BELOW TARGET"
        if val <= hi:      return "ON TARGET ✅"
        return "ABOVE TARGET ⚠"

    vo2_str = f"{vo2_max} mL/kg/min" if vo2_max else "unknown"

    assessment += f"""
[ZONE PRESCRIPTION — based on VO2max {vo2_str} and VT1/VT2 from VO2 Master GXT]

Understanding each zone and why it matters:

  Z1 – WARM-UP (<{z1[1]} bpm)
    Role:   Active recovery; flushes metabolic waste; primes aerobic enzymes.
            Essential bookend to every session — skipping it raises injury risk.
    Too little: Joints and muscles under-prepared; injury risk up.
    Too much:  No training stimulus — purely recovery, won't move VO2max.
    This week: {z_total_1:.0f} min — Z1 is not a target, it's structural filler.

  Z2 – FAT BURN / AEROBIC BASE ({z1[1]+1}–{z2[1]} bpm, below VT1={vt1} bpm)
    Role:   Builds mitochondrial density, improves fat oxidation, expands
            cardiac stroke volume. This is the PRIMARY driver of VO2max gains
            for someone at VO2max {vo2_str}. Every hour here is compounding.
    Too little: VO2max stagnates. Mitochondria don't multiply. Fat engine stays weak.
                You rely on carbs for everything → bonk faster, recover slower.
    Too much:  Not really possible at this stage. The risk is NONE from over-doing Z2.
               More Z2 = more benefit, up to ~10 hrs/week for advanced athletes.
    TARGET: {Z2_TARGET_MIN}–{Z2_TARGET_MAX} min/week ({Z2_TARGET_MIN//60}–{Z2_TARGET_MAX//60} hrs)
    This week: {z_total_2_f:.0f} min — {_status(z_total_2_f, Z2_TARGET_MIN, Z2_TARGET_MAX)}
    Gap: {max(0, Z2_TARGET_MIN - z_total_2_f):.0f} min short of minimum target
    How to hit it: {max(0, Z2_TARGET_MIN - z_total_2_f):.0f} min ÷ 3 sessions = ~{max(0, Z2_TARGET_MIN - z_total_2_f)/3:.0f} min/session of steady running/cycling at {z1[1]+1}–{z2[1]} bpm.
    NOTE: Your VT1={vt1} bpm means Z2 is not "easy jogging" — it's a brisk aerobic effort.
          You must run SLOWER than feels natural to stay under {z2[1]} bpm.

  Z3 – ENDURANCE / TEMPO ({z2[1]+1}–{(zones.get('zone_3') or (0,164))[1]} bpm, above VT1)
  Z4 – VIGOROUS / THRESHOLD ({(zones.get('zone_3') or (0,164))[1]+1}–{vt2} bpm, approaching VT2)
    Role:   Lactate threshold training. Raises the ceiling of how fast you can
            go before lactic acid accumulates. Also directly stimulates VO2max.
            This is where pickleball rallies and hard running intervals land.
    Too little: VO2max ceiling stays low. Fat base alone won't push you past a plateau.
    Too much:  Chronic fatigue, elevated resting HR, suppressed HRV, overtraining.
               Z3+Z4 stress the body significantly — need Z1+Z2 recovery sessions between.
    TARGET: {Z34_TARGET_MIN}–{Z34_TARGET_MAX} min/week
    This week: {z_total_34:.0f} min — {_status(z_total_34, Z34_TARGET_MIN, Z34_TARGET_MAX)}
    Your pickleball provides organic Z3/Z4 via rally bursts — this is effective.

  Z5 – MAXIMAL (>{vt2} bpm, above VT2)
    Role:   Highest VO2max stimulus per minute. Short intervals (30s–4 min) at
            maximum effort force cardiac output to its ceiling, driving adaptations.
    Too little: Misses the most powerful VO2max signal — but not needed yet at your stage.
    Too much:  Overtraining risk is HIGH. Immune suppression, injury, HRV crash.
               Rule: no more than 1 session/week, only after Z2 base is established.
    TARGET: {Z5_TARGET_MIN}–{Z5_TARGET_MAX} min/week (add ONLY once weekly Z2 reaches 3+ hrs)
    This week: {z_total_5_f:.0f} min — {"hold off until Z2 base is built" if z_total_2_f < Z2_TARGET_MIN else _status(z_total_5_f, Z5_TARGET_MIN, Z5_TARGET_MAX)}

PRIORITY ACTIONS THIS WEEK:
  1. Z2 deficit = {max(0, Z2_TARGET_MIN - z_total_2_f):.0f} min. Add {max(1, round((Z2_TARGET_MIN - z_total_2_f)/45))} × 45-min run/cycle at {z1[1]+1}–{z2[1]} bpm.
  2. Z3+Z4 = {z_total_34:.0f} min — {"adequate, keep current pickleball cadence." if z_total_34 >= Z34_TARGET_MIN else f"add 1 × 20–30 min tempo effort above {vt1} bpm."}
  3. Z5 = {z_total_5_f:.0f} min — hold off. Build Z2 base first.
  Expected VO2max improvement: {f'+2–3 mL/kg/min in 8–12 weeks with consistent Z2 volume' if vo2_max and vo2_max < 35 else '+1–2 mL/kg/min in 8–12 weeks'}
"""

    return assessment


def get_personalized_zones(db_path: str = "oracle.db") -> dict:
    """
    Load personalized training zones from VO2 Master GXT lab results.

    Returns a 5-zone system (matching VO2 Master report) plus VT1/VT2 for polarized analysis.
    Zone boundaries from the DB (GXT_ZONE1-4_HR_MAX) take priority; VT1/VT2 used as fallback.

    5-zone system (VO2 Master):
      Zone 1 Warm-Up:    <zone1_max bpm
      Zone 2 Fat Burn:   zone1_max – vt1 bpm   (up to VT1)
      Zone 3 Endurance:  vt1 – zone3_max bpm    (above VT1, approaching VT2)
      Zone 4 Vigorous:   zone3_max – vt2 bpm    (below VT2)
      Zone 5 Maximal:    >vt2 bpm
    """
    import sqlite3

    def _get(cursor, code):
        cursor.execute(
            "SELECT value FROM lab_markers WHERE marker_code = ? ORDER BY collected_at DESC LIMIT 1",
            (code,)
        )
        r = cursor.fetchone()
        return r[0] if r else None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        vt1_bpm      = _get(cursor, "VT1_HR")
        vt2_bpm      = _get(cursor, "VT2_HR")
        vo2_max      = _get(cursor, "VO2_MAX")
        vo2_pct      = _get(cursor, "VO2_MAX_PERCENTILE")
        max_hr       = _get(cursor, "MAX_HR_MEASURED")
        max_power    = _get(cursor, "MAX_POWER")
        vt1_power    = _get(cursor, "VT1_POWER")
        vt2_power    = _get(cursor, "VT2_POWER")
        zone1_hr_max = _get(cursor, "GXT_ZONE1_HR_MAX")
        zone2_hr_max = _get(cursor, "GXT_ZONE2_HR_MAX")  # = VT1
        zone3_hr_max = _get(cursor, "GXT_ZONE3_HR_MAX")
        zone4_hr_max = _get(cursor, "GXT_ZONE4_HR_MAX")  # = VT2

        conn.close()

        # Prefer explicit zone boundaries; fall back to VT1/VT2
        if vt1_bpm and vt2_bpm:
            z1_max = int(zone1_hr_max) if zone1_hr_max else int(vt1_bpm) - 16
            z2_max = int(zone2_hr_max) if zone2_hr_max else int(vt1_bpm)
            z3_max = int(zone3_hr_max) if zone3_hr_max else int((vt1_bpm + vt2_bpm) / 2)
            z4_max = int(zone4_hr_max) if zone4_hr_max else int(vt2_bpm)

            return {
                # VT1/VT2 for polarized 3-zone analysis (backward-compat)
                "vt1_bpm": int(vt1_bpm),
                "vt2_bpm": int(vt2_bpm),
                # Full 5-zone system from VO2 Master
                "zone_1": (0,        z1_max),          # Warm-Up
                "zone_2": (z1_max+1, z2_max),          # Fat Burning / Aerobic Base
                "zone_3": (z2_max+1, z3_max),          # Endurance (above VT1)
                "zone_4": (z3_max+1, z4_max),          # Vigorous (approaching VT2)
                "zone_5": (z4_max+1, 999),             # Maximal (above VT2)
                # Legacy 3-zone keys (kept for backward compat)
                "zone_3_plus": (z4_max+1, 999),
                # Performance metrics
                "vo2_max":    float(vo2_max) if vo2_max else None,
                "vo2_pct":    float(vo2_pct) if vo2_pct else None,
                "max_hr":     int(max_hr) if max_hr else None,
                "max_power":  int(max_power) if max_power else None,
                "vt1_power":  int(vt1_power) if vt1_power else None,
                "vt2_power":  int(vt2_power) if vt2_power else None,
                "status": (
                    f"VO2max: {vo2_max} mL/kg/min ({int(vo2_pct)}th %ile) | "
                    f"Max HR: {max_hr} bpm | "
                    f"VT1: {int(vt1_bpm)} bpm ({int(vt1_power)}W) | "
                    f"VT2: {int(vt2_bpm)} bpm ({int(vt2_power)}W) | "
                    f"5-zone: <{z1_max} / {z1_max+1}-{z2_max} / {z2_max+1}-{z3_max} / {z3_max+1}-{z4_max} / >{z4_max}"
                    if vt1_power and vt2_power and max_hr and vo2_pct
                    else f"VT1: {int(vt1_bpm)} bpm | VT2: {int(vt2_bpm)} bpm | VO2max: {vo2_max}"
                ),
            }
    except Exception as e:
        print(f"[WARNING] Could not load personalized zones: {e}")

    return {
        "vt1_bpm": None, "vt2_bpm": None,
        "zone_1": None, "zone_2": None, "zone_3": None, "zone_4": None, "zone_5": None,
        "zone_3_plus": None,
        "vo2_max": None, "vo2_pct": None, "max_hr": None,
        "status": "No VO2 Master test data found — upload your GXT report to enable personalized zones"
    }
