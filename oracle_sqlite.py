"""
Strategic Oracle  SQLite Memory Integration
=============================================
Zero-setup persistent memory using SQLite (built into Python).
No server, no installation, just a single oracle.db file.
"""

import sqlite3
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pathlib import Path


# =============================================================================
# DATABASE CONNECTOR
# =============================================================================
class OracleDB:
    def __init__(self, db_path: str = "oracle.db"):
        self.db_path = db_path
        self._init_schema()
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                intent TEXT,
                day_pillar TEXT,
                stem TEXT,
                branch TEXT,
                officer_type TEXT,
                auspicious_sector TEXT,
                weather_condition TEXT,
                temperature_c INTEGER,
                humidity INTEGER,
                uv_index INTEGER,
                is_rainy INTEGER,
                venues_recommended TEXT,
                helpful INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS venue_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_date TEXT DEFAULT (date('now')),
                venue_name TEXT NOT NULL,
                venue_address TEXT,
                venue_sector TEXT,
                day_pillar TEXT,
                stem TEXT,
                branch TEXT,
                officer_type TEXT,
                was_auspicious_sector INTEGER,
                weather_condition TEXT,
                temperature_c INTEGER,
                energy_rating INTEGER CHECK(energy_rating BETWEEN 1 AND 10),
                mood_before TEXT,
                mood_after TEXT,
                alignment_score INTEGER CHECK(alignment_score BETWEEN 1 AND 10),
                would_return INTEGER,
                activity_type TEXT,
                duration_minutes INTEGER,
                accompanied INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS life_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                event_type TEXT,
                event_description TEXT NOT NULL,
                luck_cycle TEXT,
                year_pillar TEXT,
                month_pillar TEXT,
                day_pillar TEXT,
                emotional_state TEXT,
                decision_quality TEXT,
                outcome_sentiment TEXT,
                outcome_notes TEXT,
                reflection TEXT,
                lessons_learned TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_date TEXT DEFAULT (date('now')),
                activity_name TEXT NOT NULL,
                primary_element TEXT,
                secondary_element TEXT,
                category TEXT,
                duration_minutes INTEGER,
                intensity TEXT,
                meaning_score INTEGER CHECK(meaning_score BETWEEN 1 AND 10),
                energy_score INTEGER CHECK(energy_score BETWEEN 1 AND 10),
                flow_state INTEGER,
                location_type TEXT,
                accompanied INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_practice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                practice_date TEXT DEFAULT (date('now')),
                skill_name TEXT NOT NULL,
                category TEXT,
                hours REAL NOT NULL,
                proficiency_level TEXT,
                milestone_reached TEXT,
                certification TEXT,
                people_taught INTEGER,
                revenue_earned REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS element_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_name TEXT UNIQUE NOT NULL,
                primary_element TEXT NOT NULL,
                secondary_element TEXT,
                description TEXT,
                typical_duration INTEGER,
                energy_level TEXT,
                solo_or_social TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lab_reports (
                id TEXT PRIMARY KEY,
                collected_at TEXT NOT NULL,
                source TEXT,
                report_type TEXT,
                raw_pdf_name TEXT,
                extracted_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lab_markers (
                id TEXT PRIMARY KEY,
                lab_report_id TEXT REFERENCES lab_reports(id),
                marker_code TEXT NOT NULL,
                marker_name TEXT,
                value REAL,
                unit TEXT,
                ref_low REAL,
                ref_high REAL,
                flag TEXT,
                collected_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lab_marker_code ON lab_markers(marker_code, collected_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lab_report ON lab_markers(lab_report_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_id TEXT UNIQUE,
                activity_date TEXT NOT NULL,
                activity_name TEXT NOT NULL,
                activity_type TEXT,
                primary_element TEXT,
                secondary_element TEXT,
                duration_minutes INTEGER,
                distance_km REAL,
                avg_hr INTEGER,
                max_hr INTEGER,
                calories INTEGER,
                intensity TEXT,
                synced_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_date TEXT UNIQUE NOT NULL,
                steps INTEGER,
                sleep_hours REAL,
                sleep_score INTEGER,
                resting_hr INTEGER,
                body_battery INTEGER,
                avg_stress INTEGER,
                total_calories INTEGER,
                active_calories INTEGER,
                synced_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garmin_comprehensive_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                synced_date TEXT UNIQUE NOT NULL,
                training_readiness_score INTEGER,
                training_readiness_status TEXT,
                training_load TEXT,
                endurance_score REAL,
                vo2_max REAL,
                max_heart_rate INTEGER,
                hrv_last_night_avg REAL,
                hrv_status TEXT,
                respiration_rate REAL,
                body_weight_kg REAL,
                body_fat_percent REAL,
                muscle_percent REAL,
                intensity_minutes_weekly INTEGER,
                blood_pressure_systolic INTEGER,
                blood_pressure_diastolic INTEGER,
                raw_comprehensive_json TEXT,
                synced_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Current life context — user-editable snapshot for divination personalization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS life_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                situation TEXT,
                emotional_state TEXT,
                focus_area TEXT,
                recent_events TEXT,
                phase TEXT,
                goals TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Generated divination reflections history (4-field structure)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS divination_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reflection_date TEXT DEFAULT (date('now')),
                day_pillar TEXT,
                element TEXT,
                officer TEXT,
                user_context_json TEXT,
                headline TEXT,
                focus TEXT,
                watch TEXT,
                reflection_text TEXT NOT NULL,
                model_used TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add new columns to existing tables (safe on already-created DBs)
        for col_def in [
            ("garmin_daily_stats", "sleep_deep_minutes",  "INTEGER"),
            ("garmin_daily_stats", "sleep_rem_minutes",   "INTEGER"),
            ("garmin_daily_stats", "sleep_light_minutes", "INTEGER"),
            ("garmin_daily_stats", "sleep_awake_minutes", "INTEGER"),
            ("garmin_daily_stats", "body_battery_morning","INTEGER"),
            ("garmin_comprehensive_health", "hrv_trend",  "TEXT"),
            ("garmin_comprehensive_health", "hrv_5min_high", "REAL"),
            ("garmin_comprehensive_health", "hydration_ml", "INTEGER"),
            ("garmin_comprehensive_health", "stress_balance", "TEXT"),
            # Actual time-in-zone data (minutes per VO2 Master 5-zone system)
            ("garmin_activities", "z1_minutes", "REAL"),
            ("garmin_activities", "z2_minutes", "REAL"),
            ("garmin_activities", "z3_minutes", "REAL"),
            ("garmin_activities", "z4_minutes", "REAL"),
            ("garmin_activities", "z5_minutes", "REAL"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE {col_def[0]} ADD COLUMN {col_def[1]} {col_def[2]}")
            except Exception:
                pass  # Column already exists

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversation_log(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_intent ON conversation_log(intent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_venue_date ON venue_visits(visit_date DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_venue_name ON venue_visits(venue_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_date ON daily_activities(activity_date DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_element ON daily_activities(primary_element)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_garmin_date ON garmin_activities(activity_date DESC)")
        
        cursor.execute("SELECT COUNT(*) FROM element_catalog")
        if cursor.fetchone()[0] == 0:
            activities = [
                ('Yoga teaching', 'Wood', 'Fire', 'Guiding others in growth and movement', 60, 'moderate', 'social'),
                ('Yoga practice', 'Wood', 'Water', 'Personal practice, introspective', 45, 'moderate', 'solo'),
                ('Coffee brewing', 'Metal', 'Water', 'Precision, aesthetic craft, mindfulness', 20, 'light', 'solo'),
                ('AI study', 'Metal', 'Wood', 'Technical learning, structured thinking', 90, 'high', 'solo'),
                ('Reading Murakami', 'Water', 'Wood', 'Introspective, imaginative', 60, 'light', 'solo'),
                ('Jazz listening', 'Water', 'Fire', 'Emotional resonance, uplifting or mellow', 30, 'light', 'solo'),
                ('Deep conversation', 'Fire', 'Wood', 'Expression, connection, growth-oriented', 90, 'moderate', 'social'),
                ('Running', 'Fire', 'Wood', 'Yang movement, energizing', 40, 'high', 'solo'),
                ('Pickleball', 'Fire', 'Metal', 'Social sport, precision and fun', 60, 'high', 'social'),
                ('Thai massage', 'Earth', 'Water', 'Grounding, nurturing touch', 90, 'moderate', 'social'),
                ('Nature walk', 'Wood', 'Earth', 'Growth environment, grounding', 45, 'light', 'either'),
                ('Theatre/film', 'Fire', 'Water', 'Emotional expression, introspection', 150, 'light', 'either'),
                ('Visiting cats', 'Earth', 'Water', 'Nurturing, playful, grounding', 30, 'light', 'social'),
                ('Meditation', 'Water', 'Metal', 'Deep introspection, stillness', 20, 'light', 'solo'),
                ('Cooking vegetarian', 'Earth', 'Wood', 'Nurturing, creative nourishment', 45, 'moderate', 'either')
            ]
            cursor.executemany("""
                INSERT INTO element_catalog 
                (activity_name, primary_element, secondary_element, description, 
                 typical_duration, energy_level, solo_or_social)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, activities)
        
        conn.commit()
        conn.close()
        print(f"[OK] Database initialized: {self.db_path}")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Optional[List[Dict]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            if fetch:
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return None
        except Exception as e:
            conn.rollback()
            print(f"SQL Error: {e}")
            print(f"Query: {query}")
            return None
        finally:
            conn.close()
    
    def log_conversation(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        day_pillar: Optional[str] = None,
        stem: Optional[str] = None,
        branch: Optional[str] = None,
        officer_type: Optional[str] = None,
        auspicious_sector: Optional[str] = None,
        weather: Optional[dict] = None,
        venues_recommended: Optional[List[dict]] = None
    ) -> bool:
        query = """
            INSERT INTO conversation_log (
                role, content, intent, day_pillar, stem, branch, officer_type, 
                auspicious_sector, weather_condition, temperature_c, humidity, 
                uv_index, is_rainy, venues_recommended
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        weather_condition = weather.get("condition") if weather else None
        temperature_c = weather.get("temp_c") if weather else None
        humidity = weather.get("humidity") if weather else None
        uv_index = weather.get("uv_index") if weather else None
        is_rainy = 1 if (weather and weather.get("is_rainy")) else 0
        venues_json = json.dumps(venues_recommended) if venues_recommended else None
        params = (
            role, content, intent, day_pillar, stem, branch, officer_type,
            auspicious_sector, weather_condition, temperature_c, humidity,
            uv_index, is_rainy, venues_json
        )
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_conversation_history(self, last_n: int = 10) -> List[Dict]:
        query = """
            SELECT role, content, timestamp, intent, day_pillar, officer_type
            FROM conversation_log
            ORDER BY timestamp DESC
            LIMIT ?
        """
        results = self.execute_query(query, (last_n,))
        return list(reversed(results)) if results else []
    
    def search_conversations(self, keyword: str, limit: int = 10) -> List[Dict]:
        query = """
            SELECT timestamp, role, content, day_pillar, officer_type
            FROM conversation_log
            WHERE content LIKE ? COLLATE NOCASE
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return self.execute_query(query, (f"%{keyword}%", limit)) or []
    
    def log_venue_visit(
        self,
        venue_name: str,
        venue_address: str = None,
        venue_sector: str = None,
        day_pillar: str = None,
        officer_type: str = None,
        weather: dict = None,
        energy_rating: int = None,
        mood_before: str = None,
        mood_after: str = None,
        alignment_score: int = None,
        would_return: bool = None,
        activity_type: str = None,
        notes: str = None
    ) -> bool:
        query = """
            INSERT INTO venue_visits (
                venue_name, venue_address, venue_sector, day_pillar, officer_type,
                weather_condition, temperature_c, energy_rating, mood_before,
                mood_after, alignment_score, would_return, activity_type, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        weather_condition = weather.get("condition") if weather else None
        temperature_c = weather.get("temp_c") if weather else None
        would_return_int = 1 if would_return else (0 if would_return is False else None)
        params = (
            venue_name, venue_address, venue_sector, day_pillar, officer_type,
            weather_condition, temperature_c, energy_rating, mood_before,
            mood_after, alignment_score, would_return_int, activity_type, notes
        )
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_top_venues(self, limit: int = 5) -> List[Dict]:
        query = """
            SELECT venue_name, ROUND(AVG(energy_rating), 2) as avg_energy, COUNT(*) as visit_count
            FROM venue_visits
            WHERE energy_rating IS NOT NULL
            GROUP BY venue_name
            ORDER BY avg_energy DESC
            LIMIT ?
        """
        return self.execute_query(query, (limit,)) or []
    
    def get_all_venues(self, limit: int = 20) -> List[Dict]:
        """Get all recommended venues ordered by most recent."""
        query = """
            SELECT venue_name, venue_address, venue_sector, day_pillar,
                   officer_type, weather_condition, visit_date
            FROM venue_visits
            ORDER BY visit_date DESC, id DESC
            LIMIT ?
        """
        return self.execute_query(query, (limit,)) or []

    def get_officer_patterns(self) -> List[Dict]:
        query = """
            SELECT officer_type, COUNT(*) as total_visits,
                   ROUND(AVG(energy_rating), 2) as avg_energy,
                   ROUND(AVG(alignment_score), 2) as avg_alignment
            FROM venue_visits
            WHERE officer_type IS NOT NULL
            GROUP BY officer_type ORDER BY avg_energy DESC
        """
        return self.execute_query(query) or []
    
    def get_sector_patterns(self) -> List[Dict]:
        query = """
            SELECT venue_sector, COUNT(*) as visit_count,
                   ROUND(AVG(energy_rating), 2) as avg_energy,
                   ROUND(AVG(alignment_score), 2) as avg_alignment,
                   SUM(CASE WHEN would_return = 1 THEN 1 ELSE 0 END) as would_return_count
            FROM venue_visits
            WHERE venue_sector IS NOT NULL
            GROUP BY venue_sector ORDER BY avg_energy DESC
        """
        return self.execute_query(query) or []
    
    def get_element_balance(self, days_back: int = 30) -> List[Dict]:
        query = """
            SELECT primary_element, SUM(duration_minutes) as total_minutes,
                   COUNT(*) as activity_count,
                   ROUND(AVG(meaning_score), 2) as avg_meaning
            FROM daily_activities
            WHERE activity_date > date('now', ? || ' days')
            GROUP BY primary_element ORDER BY total_minutes DESC
        """
        return self.execute_query(query, (f"-{days_back}",)) or []
    
    def save_lab_report(self, report, pdf_name: str = "") -> str:
        """Persist an ExtractedReport + its markers. Returns the report UUID."""
        from marker_catalog import MARKER_CATALOG
        report_id = str(__import__("uuid").uuid4())
        markers_json = __import__("json").dumps(
            [m.model_dump() for m in report.markers], default=str
        )
        self.execute_query(
            """INSERT OR IGNORE INTO lab_reports
               (id, collected_at, source, report_type, raw_pdf_name, extracted_json)
               VALUES (?,?,?,?,?,?)""",
            (report_id, str(report.collected_at), report.source,
             report.report_type, pdf_name, markers_json),
            fetch=False,
        )
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for m in report.markers:
                cursor.execute(
                    """INSERT INTO lab_markers
                       (id, lab_report_id, marker_code, marker_name, value, unit,
                        ref_low, ref_high, flag, collected_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(__import__("uuid").uuid4()), report_id,
                        m.marker_code,
                        MARKER_CATALOG.get(m.marker_code, {}).get("name", m.marker_code),
                        m.value, m.unit, m.ref_low, m.ref_high,
                        m.flag, str(report.collected_at),
                    ),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[lab save] error: {e}")
        finally:
            conn.close()
        return report_id

    def get_latest_lab_markers(self, codes: List[str] = None) -> List[Dict]:
        """Return the most recent value for each marker (or filtered by codes)."""
        if codes:
            placeholders = ",".join("?" * len(codes))
            query = f"""
                SELECT m.marker_code, m.marker_name, m.value, m.unit,
                       m.flag, m.collected_at
                FROM lab_markers m
                INNER JOIN (
                    SELECT marker_code, MAX(collected_at) as latest
                    FROM lab_markers GROUP BY marker_code
                ) latest ON m.marker_code = latest.marker_code
                         AND m.collected_at = latest.latest
                WHERE m.marker_code IN ({placeholders})
                ORDER BY m.marker_code
            """
            return self.execute_query(query, tuple(codes)) or []
        else:
            query = """
                SELECT m.marker_code, m.marker_name, m.value, m.unit,
                       m.flag, m.collected_at
                FROM lab_markers m
                INNER JOIN (
                    SELECT marker_code, MAX(collected_at) as latest
                    FROM lab_markers GROUP BY marker_code
                ) latest ON m.marker_code = latest.marker_code
                         AND m.collected_at = latest.latest
                ORDER BY m.flag DESC, m.marker_code
            """
            return self.execute_query(query) or []

    def get_abnormal_markers(self) -> List[Dict]:
        """Return latest values for markers flagged abnormal or suboptimal."""
        query = """
            SELECT m.marker_code, m.marker_name, m.value, m.unit,
                   m.flag, m.ref_low, m.ref_high, m.collected_at
            FROM lab_markers m
            INNER JOIN (
                SELECT marker_code, MAX(collected_at) as latest
                FROM lab_markers GROUP BY marker_code
            ) latest ON m.marker_code = latest.marker_code
                     AND m.collected_at = latest.latest
            WHERE m.flag IN ('abnormal', 'suboptimal')
            ORDER BY CASE m.flag WHEN 'abnormal' THEN 0 ELSE 1 END, m.marker_code
        """
        return self.execute_query(query) or []

    def get_lab_summary(self) -> Dict[str, Any]:
        """Compact health summary for the recommender prompt."""
        abnormal = self.get_abnormal_markers()
        all_latest = self.get_latest_lab_markers()
        report_count = (self.execute_query(
            "SELECT COUNT(*) as c FROM lab_reports") or [{}])[0].get("c", 0)
        return {
            "report_count": report_count,
            "marker_count": len(all_latest),
            "abnormal": abnormal,
            "has_data": bool(all_latest),
        }

    def import_garmin_activities(self, activities: List[Dict]) -> int:
        """Upsert Garmin activities. Returns count of newly inserted rows.

        If an activity dict contains z1_minutes–z5_minutes (actual time-in-zone),
        those are saved. For existing records missing zone data, we update them.
        """
        if not activities:
            return 0
        inserted = 0
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for act in activities:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO garmin_activities
                      (activity_id, activity_date, activity_name, activity_type,
                       primary_element, secondary_element, duration_minutes,
                       distance_km, avg_hr, max_hr, calories, intensity,
                       z1_minutes, z2_minutes, z3_minutes, z4_minutes, z5_minutes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        act.get("activity_id"), act.get("activity_date"),
                        act.get("activity_name"), act.get("activity_type"),
                        act.get("primary_element"), act.get("secondary_element"),
                        act.get("duration_minutes"), act.get("distance_km"),
                        act.get("avg_hr"), act.get("max_hr"),
                        act.get("calories"), act.get("intensity"),
                        act.get("z1_minutes"), act.get("z2_minutes"),
                        act.get("z3_minutes"), act.get("z4_minutes"),
                        act.get("z5_minutes"),
                    ),
                )
                inserted += cursor.rowcount  # 1 if inserted, 0 if ignored (duplicate)

                # If this activity already existed (rowcount=0) but now has zone data, update zones
                if cursor.rowcount == 0 and act.get("activity_id") and any(
                    act.get(k) is not None for k in ("z1_minutes", "z2_minutes", "z3_minutes",
                                                       "z4_minutes", "z5_minutes")
                ):
                    cursor.execute(
                        """UPDATE garmin_activities
                           SET z1_minutes=?, z2_minutes=?, z3_minutes=?, z4_minutes=?, z5_minutes=?,
                               synced_at=CURRENT_TIMESTAMP
                           WHERE activity_id=? AND z1_minutes IS NULL""",
                        (
                            act.get("z1_minutes"), act.get("z2_minutes"),
                            act.get("z3_minutes"), act.get("z4_minutes"),
                            act.get("z5_minutes"), act.get("activity_id"),
                        ),
                    )

            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[garmin import] SQL error: {e}")
        finally:
            conn.close()
        return inserted

    def upsert_garmin_daily_stats(self, stats: Dict) -> bool:
        result = self.execute_query(
            """
            INSERT INTO garmin_daily_stats
              (stat_date, steps, sleep_hours, sleep_score, resting_hr,
               body_battery, avg_stress, total_calories, active_calories,
               sleep_deep_minutes, sleep_rem_minutes, sleep_light_minutes,
               sleep_awake_minutes, body_battery_morning)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(stat_date) DO UPDATE SET
              steps=excluded.steps, sleep_hours=excluded.sleep_hours,
              sleep_score=excluded.sleep_score, resting_hr=excluded.resting_hr,
              body_battery=excluded.body_battery, avg_stress=excluded.avg_stress,
              total_calories=excluded.total_calories,
              active_calories=excluded.active_calories,
              sleep_deep_minutes=excluded.sleep_deep_minutes,
              sleep_rem_minutes=excluded.sleep_rem_minutes,
              sleep_light_minutes=excluded.sleep_light_minutes,
              sleep_awake_minutes=excluded.sleep_awake_minutes,
              body_battery_morning=excluded.body_battery_morning,
              synced_at=CURRENT_TIMESTAMP
            """,
            (
                stats.get("date"), stats.get("steps"), stats.get("sleep_hours"),
                stats.get("sleep_score"), stats.get("resting_hr"),
                stats.get("body_battery"), stats.get("avg_stress"),
                stats.get("total_calories"), stats.get("active_calories"),
                stats.get("sleep_deep_minutes"), stats.get("sleep_rem_minutes"),
                stats.get("sleep_light_minutes"), stats.get("sleep_awake_minutes"),
                stats.get("body_battery_morning"),
            ),
            fetch=False,
        )
        return result is not None

    def upsert_garmin_comprehensive_health(self, comp_data: Dict) -> bool:
        """Store comprehensive health metrics (training readiness, HRV, body comp, etc.)."""
        try:
            readiness = comp_data.get("training_readiness", {})
            train_status = comp_data.get("training_status", {})
            hrv = comp_data.get("hrv", {})
            resp = comp_data.get("respiration", {})
            body_comp = comp_data.get("body_composition", {})
            max_metrics = comp_data.get("max_metrics", {})
            weekly = comp_data.get("weekly_summary", {})
            bp = comp_data.get("blood_pressure", {})

            result = self.execute_query(
                """
                INSERT INTO garmin_comprehensive_health
                  (synced_date, training_readiness_score, training_readiness_status,
                   training_load, endurance_score, vo2_max, max_heart_rate,
                   hrv_last_night_avg, hrv_status, hrv_trend, hrv_5min_high,
                   respiration_rate, body_weight_kg, body_fat_percent, muscle_percent,
                   intensity_minutes_weekly, blood_pressure_systolic,
                   blood_pressure_diastolic, hydration_ml, stress_balance,
                   raw_comprehensive_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(synced_date) DO UPDATE SET
                  training_readiness_score=excluded.training_readiness_score,
                  training_readiness_status=excluded.training_readiness_status,
                  training_load=excluded.training_load,
                  endurance_score=excluded.endurance_score,
                  vo2_max=excluded.vo2_max,
                  max_heart_rate=excluded.max_heart_rate,
                  hrv_last_night_avg=excluded.hrv_last_night_avg,
                  hrv_status=excluded.hrv_status,
                  hrv_trend=excluded.hrv_trend,
                  hrv_5min_high=excluded.hrv_5min_high,
                  respiration_rate=excluded.respiration_rate,
                  body_weight_kg=excluded.body_weight_kg,
                  body_fat_percent=excluded.body_fat_percent,
                  muscle_percent=excluded.muscle_percent,
                  intensity_minutes_weekly=excluded.intensity_minutes_weekly,
                  blood_pressure_systolic=excluded.blood_pressure_systolic,
                  blood_pressure_diastolic=excluded.blood_pressure_diastolic,
                  hydration_ml=excluded.hydration_ml,
                  stress_balance=excluded.stress_balance,
                  raw_comprehensive_json=excluded.raw_comprehensive_json,
                  synced_at=CURRENT_TIMESTAMP
                """,
                (
                    comp_data.get("pulled_at"),
                    readiness.get("score"),
                    readiness.get("status"),
                    train_status.get("load"),
                    comp_data.get("endurance_score"),
                    max_metrics.get("vo2_max"),
                    max_metrics.get("max_hr"),
                    hrv.get("last_night_avg"),
                    hrv.get("last_night_status") or hrv.get("status"),
                    hrv.get("trend"),
                    hrv.get("last_night_5min"),
                    resp.get("avg_breathing_rate"),
                    body_comp.get("weight_kg"),
                    body_comp.get("body_fat_percent"),
                    body_comp.get("muscle_percent"),
                    weekly.get("intensity_minutes"),
                    bp.get("systolic"),
                    bp.get("diastolic"),
                    comp_data.get("hydration"),
                    train_status.get("stress_balance"),
                    __import__("json").dumps(comp_data, default=str),
                ),
                fetch=False,
            )
            return result is not None
        except Exception as e:
            print(f"Error storing comprehensive health: {e}")
            return False

    def get_garmin_summary(self, days: int = 7,
                           start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Return activities + daily stats for a date range.

        Prefer start_date/end_date (ISO strings) for exact calendar ranges.
        Falls back to days-offset from today when only days is given.
        """
        import datetime as _dt

        today_str = _dt.date.today().isoformat()

        # Resolve date bounds
        if start_date and end_date:
            act_start = start_date
            act_end   = end_date
        else:
            # Legacy: days offset from today (always at least 7 for activity context)
            activity_days = max(days, 7)
            act_start = (_dt.date.today() - _dt.timedelta(days=activity_days)).isoformat()
            act_end   = today_str

        activities = self.execute_query(
            """
            SELECT activity_id, activity_date, activity_name, primary_element,
                   duration_minutes, distance_km, avg_hr, intensity,
                   z1_minutes, z2_minutes, z3_minutes, z4_minutes, z5_minutes
            FROM garmin_activities
            WHERE activity_date >= ? AND activity_date <= ?
            ORDER BY activity_date DESC
            LIMIT 30
            """,
            (act_start, act_end),
        ) or []

        _daily_cols = """stat_date, steps, sleep_hours, sleep_score,
                       resting_hr, body_battery, avg_stress,
                       sleep_deep_minutes, sleep_rem_minutes,
                       sleep_light_minutes, sleep_awake_minutes,
                       body_battery_morning,
                       cast(julianday('now') - julianday(stat_date) as integer) as days_old"""

        def _most_recent_stats(from_date: str):
            """Return the most recent daily stats record on or after from_date.
            Prefers rows that have at least one health metric populated."""
            rows = self.execute_query(
                f"""SELECT {_daily_cols} FROM garmin_daily_stats
                    WHERE stat_date >= ?
                    ORDER BY stat_date DESC LIMIT 5""",
                (from_date,)
            ) or []
            # Prefer rows with actual health data over empty shell rows
            for row in rows:
                if any(row.get(c) for c in ("sleep_hours", "resting_hr", "body_battery",
                                             "avg_stress", "body_battery_morning")):
                    return [row]
            return rows[:1] if rows else []

        if start_date and end_date:
            if start_date == end_date:
                # Single day (today/yesterday queries).
                # For stats: always look back up to 3 days — Garmin sleep/HRV data
                # often appears 1-2 days after the measurement date.
                three_days_ago = (_dt.date.today() - _dt.timedelta(days=3)).isoformat()
                daily = _most_recent_stats(three_days_ago)
            else:
                # Date range — average the stats across the window
                daily = self.execute_query(
                    f"""SELECT ? as stat_date,
                           ROUND(AVG(steps)) as steps,
                           ROUND(AVG(sleep_hours), 1) as sleep_hours,
                           ROUND(AVG(sleep_score)) as sleep_score,
                           ROUND(AVG(resting_hr)) as resting_hr,
                           ROUND(AVG(body_battery)) as body_battery,
                           ROUND(AVG(avg_stress)) as avg_stress,
                           ROUND(AVG(sleep_deep_minutes)) as sleep_deep_minutes,
                           ROUND(AVG(sleep_rem_minutes)) as sleep_rem_minutes,
                           ROUND(AVG(sleep_light_minutes)) as sleep_light_minutes,
                           ROUND(AVG(sleep_awake_minutes)) as sleep_awake_minutes,
                           ROUND(AVG(body_battery_morning)) as body_battery_morning,
                           0 as days_old
                    FROM garmin_daily_stats WHERE stat_date >= ? AND stat_date <= ?""",
                    (end_date, start_date, end_date)
                ) or []
        elif days <= 1:
            daily = self.execute_query(
                f"SELECT {_daily_cols} FROM garmin_daily_stats WHERE stat_date = date('now') ORDER BY stat_date DESC LIMIT 1"
            ) or []
        elif days == 2:
            daily = self.execute_query(
                f"SELECT {_daily_cols} FROM garmin_daily_stats WHERE stat_date >= date('now', '-1 days') ORDER BY stat_date DESC LIMIT 1"
            ) or []
        else:
            daily = self.execute_query(
                f"""SELECT date('now') as stat_date,
                       ROUND(AVG(steps)) as steps,
                       ROUND(AVG(sleep_hours), 1) as sleep_hours,
                       ROUND(AVG(sleep_score)) as sleep_score,
                       ROUND(AVG(resting_hr)) as resting_hr,
                       ROUND(AVG(body_battery)) as body_battery,
                       ROUND(AVG(avg_stress)) as avg_stress,
                       ROUND(AVG(sleep_deep_minutes)) as sleep_deep_minutes,
                       ROUND(AVG(sleep_rem_minutes)) as sleep_rem_minutes,
                       ROUND(AVG(sleep_light_minutes)) as sleep_light_minutes,
                       ROUND(AVG(sleep_awake_minutes)) as sleep_awake_minutes,
                       ROUND(AVG(body_battery_morning)) as body_battery_morning,
                       0 as days_old
                FROM garmin_daily_stats WHERE stat_date >= date('now', '-{days} days')"""
            ) or []

        latest = daily[0] if daily else {}
        stats_age = latest.get("days_old", None)

        # Fetch comprehensive health metrics (if available)
        comprehensive = self.execute_query(
            """
            SELECT training_readiness_score, training_readiness_status,
                   training_load, endurance_score, vo2_max, max_heart_rate,
                   hrv_last_night_avg, hrv_status, hrv_trend, hrv_5min_high,
                   respiration_rate, body_weight_kg, body_fat_percent, muscle_percent,
                   intensity_minutes_weekly, blood_pressure_systolic,
                   blood_pressure_diastolic, hydration_ml, stress_balance,
                   raw_comprehensive_json
            FROM garmin_comprehensive_health
            ORDER BY synced_date DESC
            LIMIT 1
            """,
        )
        comp_dict = None
        if comprehensive:
            row = comprehensive[0]
            # Parse raw JSON to recover any fields not in dedicated columns
            import json as _json
            raw = {}
            try:
                raw = _json.loads(row.get("raw_comprehensive_json") or "{}")
            except Exception:
                pass
            raw_weekly = raw.get("weekly_summary", {})
            raw_readiness = raw.get("training_readiness", {})

            comp_dict = {
                "training_readiness": {
                    "score": row.get("training_readiness_score"),
                    "status": row.get("training_readiness_status"),
                    "factors": raw_readiness.get("factors", {}),
                },
                "training_status": {
                    "load": row.get("training_load"),
                    "stress_balance": row.get("stress_balance") or raw.get("training_status", {}).get("stress_balance"),
                },
                "endurance_score": row.get("endurance_score"),
                "max_metrics": {
                    "vo2_max": row.get("vo2_max"),
                    "max_hr": row.get("max_heart_rate"),
                },
                "hrv": {
                    "last_night_avg": row.get("hrv_last_night_avg"),
                    "status": row.get("hrv_status"),
                    "trend": row.get("hrv_trend") or raw.get("hrv", {}).get("trend"),
                    "last_night_5min": row.get("hrv_5min_high") or raw.get("hrv", {}).get("last_night_5min"),
                },
                "respiration": {
                    "avg_breathing_rate": row.get("respiration_rate"),
                    "status": raw.get("respiration", {}).get("status"),
                },
                "body_composition": {
                    "weight_kg": row.get("body_weight_kg"),
                    "body_fat_percent": row.get("body_fat_percent"),
                    "muscle_percent": row.get("muscle_percent"),
                },
                "weekly_summary": {
                    "intensity_minutes": row.get("intensity_minutes_weekly"),
                    "moderate_intensity_minutes": raw_weekly.get("moderate_intensity_minutes"),
                    "vigorous_intensity_minutes": raw_weekly.get("vigorous_intensity_minutes"),
                    "avg_stress": raw_weekly.get("avg_stress"),
                },
                "blood_pressure": {
                    "systolic": row.get("blood_pressure_systolic"),
                    "diastolic": row.get("blood_pressure_diastolic"),
                },
                "hydration_ml": row.get("hydration_ml") or raw.get("hydration"),
            }

        # ── Compute WHO intensity minutes from activity data (reliable) ──────────
        # Garmin's API weekly_summary.intensityMinutesTotal is often unreliable
        # (returns today's value only, or 0). We calculate from our own data.
        # Using VT1/VT2 thresholds from lab_markers if available.
        try:
            _vt1 = _vt2 = None
            _markers = self.execute_query(
                "SELECT marker_code, value FROM lab_markers "
                "WHERE marker_code IN ('VT1_HR','VT2_HR') ORDER BY collected_at DESC LIMIT 4"
            ) or []
            for _m in _markers:
                if _m["marker_code"] == "VT1_HR" and _vt1 is None:
                    _vt1 = float(_m["value"])
                if _m["marker_code"] == "VT2_HR" and _vt2 is None:
                    _vt2 = float(_m["value"])
            # Defaults if no lab data
            _vt1 = _vt1 or 156
            _vt2 = _vt2 or 173
            _Z2_LOW = 130   # below VT1 but above easy aerobic — counts as WHO moderate

            _moderate_min = 0.0
            _vigorous_min = 0.0
            for _act in activities:
                _dur  = _act.get("duration_minutes") or 0
                _avghr = _act.get("avg_hr")
                _has_zones = _act.get("z1_minutes") is not None

                if _has_zones:
                    # Actual zone data: Z2 (140-VT1) = moderate; Z3-Z5 (>VT1) = vigorous
                    _moderate_min += (_act.get("z2_minutes") or 0)
                    _vigorous_min += ((_act.get("z3_minutes") or 0)
                                    + (_act.get("z4_minutes") or 0)
                                    + (_act.get("z5_minutes") or 0))
                elif _avghr:
                    if _avghr >= _vt1:       # above VT1 → vigorous
                        _vigorous_min += _dur
                    elif _avghr >= _Z2_LOW:  # Z2 range → moderate
                        _moderate_min += _dur

            _moderate_min = round(_moderate_min)
            _vigorous_min = round(_vigorous_min)
            # WHO formula: vigorous counts double toward the combined 150-min target
            _who_equiv = _moderate_min + (_vigorous_min * 2)
            computed_intensity = {
                "moderate_minutes": _moderate_min,
                "vigorous_minutes": _vigorous_min,
                "who_equivalent":   _who_equiv,
                "who_target":       150,
            }
        except Exception as _ie:
            print(f"[intensity calc] error: {_ie}")
            computed_intensity = None

        return {
            "activities":  activities,
            "latest_stats": latest,
            "stats_age_days": stats_age,          # None means no stats within last 2 days
            "comprehensive_health": comp_dict,    # Comprehensive health metrics (if available)
            "computed_intensity": computed_intensity,  # WHO minutes calculated from activities
            "has_data": bool(activities or daily),
        }

    def seed_vo2_master_data(self, markers: list, report_date: str, source: str = "VO2 Master GXT") -> int:
        """
        Directly write VO2 Master GXT results to lab_markers.
        markers: list of (marker_code, value, unit) tuples.
        Returns count of rows inserted.
        """
        import uuid as _uuid
        report_id = str(_uuid.uuid4())
        self.execute_query(
            "INSERT OR IGNORE INTO lab_reports (report_id, report_type, source, collected_at, raw_text) VALUES (?,?,?,?,?)",
            (report_id, "vo2_master", source, report_date, f"Seeded from known GXT values — {source}"),
            fetch=False,
        )
        n = 0
        for code, value, unit in markers:
            try:
                from marker_catalog import MARKER_CATALOG, compute_flag
                name = MARKER_CATALOG.get(code, {}).get("name", code)
                flag = compute_flag(code, float(value))
                self.execute_query(
                    """INSERT OR REPLACE INTO lab_markers
                       (report_id, marker_code, marker_name, value, unit, flag, collected_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (report_id, code, name, float(value), unit, flag, report_date),
                    fetch=False,
                )
                n += 1
            except Exception as e:
                print(f"  [seed] {code}: {e}")
        print(f"  [seed_vo2] Inserted {n} markers from {source}")
        return n

    def get_garmin_activity_count(self) -> int:
        result = self.execute_query("SELECT COUNT(*) as count FROM garmin_activities")
        return result[0]["count"] if result else 0

    # =========================================================================
    # DIVINATION SUPPORT — life context + reflection storage
    # =========================================================================
    def get_current_life_context(self) -> Dict[str, Any]:
        """
        Fetch the user's most recent life context for divination personalization.
        Falls back to sensible defaults if no context has been saved yet.
        """
        rows = self.execute_query(
            """SELECT situation, emotional_state, focus_area, recent_events,
                      phase, goals, updated_at
               FROM life_context
               ORDER BY id DESC LIMIT 1"""
        )

        if rows:
            return rows[0]

        # Default context derived from HARDCODED_PROFILE
        return {
            "situation": "Career transition from banking toward teaching, coaching, and meaningful AI work",
            "emotional_state": "Reflective, navigating tension between achievement and meaning",
            "focus_area": "Integrating Mind (AI), Body (yoga), Heart (teaching), and Sensory Craft (coffee)",
            "recent_events": "Mother's passing in 2025 deepened orientation toward presence and intentional living",
            "phase": "Bing Chen luck cycle (2025-2035) — self-amplification, identity-driven decisions",
            "goals": "Barista FIRE; sustainable meaningful work; teaching, guiding, deep AI applications",
            "updated_at": None
        }

    def save_life_context(self, context: Dict[str, str]) -> int:
        """Insert a new life context snapshot. Returns the new row id."""
        import datetime as _dt
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO life_context
               (situation, emotional_state, focus_area, recent_events,
                phase, goals, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                context.get("situation", ""),
                context.get("emotional_state", ""),
                context.get("focus_area", ""),
                context.get("recent_events", ""),
                context.get("phase", ""),
                context.get("goals", ""),
                _dt.datetime.now().isoformat()
            )
        )
        conn.commit()
        return cursor.lastrowid

    def save_divination_reflection(self, reflection_data: Dict[str, Any]) -> int:
        """Persist a generated divination reflection (4-field structure)."""
        import json as _json
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO divination_history
               (reflection_date, day_pillar, element, officer,
                user_context_json, headline, focus, watch,
                reflection_text, model_used)
               VALUES (date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                reflection_data.get("day_pillar", ""),
                reflection_data.get("element", ""),
                reflection_data.get("officer", ""),
                _json.dumps(reflection_data.get("user_context", {})),
                reflection_data.get("headline", ""),
                reflection_data.get("focus", ""),
                reflection_data.get("watch", ""),
                reflection_data.get("reflection_text", ""),
                reflection_data.get("model_used", "")
            )
        )
        conn.commit()
        return cursor.lastrowid

    def get_today_divination(self) -> Optional[Dict[str, Any]]:
        """Return today's most recent divination reflection if one exists (4-field structure)."""
        rows = self.execute_query(
            """SELECT id, reflection_date, day_pillar, element, officer,
                      headline, focus, watch, reflection_text, model_used, created_at
               FROM divination_history
               WHERE reflection_date = date('now')
               ORDER BY id DESC LIMIT 1"""
        )
        return rows[0] if rows else None

    def get_recent_divinations(self, limit: int = 7) -> List[Dict[str, Any]]:
        """Return the N most recent divination reflections (4-field structure)."""
        return self.execute_query(
            """SELECT reflection_date, day_pillar, element, officer,
                      headline, focus, watch, reflection_text, model_used, created_at
               FROM divination_history
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        ) or []

    def get_daily_recovery_snapshot(self) -> Dict[str, Any]:
        """
        Fetch TODAY's recovery data (sleep, body battery, HRV, resting HR, stress).
        Returns formatted dictionary for display in sidebar card.
        """
        import datetime as _dt
        today = _dt.date.today().isoformat()

        # Get today's daily stats
        daily = self.execute_query(
            """SELECT stat_date, sleep_hours, sleep_score, body_battery, body_battery_morning,
                      resting_hr, avg_stress, sleep_deep_minutes, sleep_rem_minutes
               FROM garmin_daily_stats
               WHERE stat_date >= date('now', '-2 days')
               ORDER BY stat_date DESC LIMIT 1""",
        )

        # Get comprehensive health metrics
        comp = self.execute_query(
            """SELECT training_readiness_score, hrv_last_night_avg, hrv_status,
                      respiration_rate, vo2_max
               FROM garmin_comprehensive_health
               ORDER BY synced_date DESC LIMIT 1""",
        )

        return {
            "has_data": bool(daily),
            "daily_stats": daily[0] if daily else {},
            "comprehensive": comp[0] if comp else {},
            "date": today,
        }

    def get_stats(self) -> Dict[str, int]:
        stats = {}
        tables = ['conversation_log', 'venue_visits', 'life_events', 'daily_activities', 'skill_practice', 'garmin_activities', 'lab_reports', 'lab_markers']
        for table in tables:
            result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = result[0]['count'] if result else 0
        return stats


# =============================================================================
# NATURAL LANGUAGE TO SQL CONVERTER
# =============================================================================
class NLtoSQL:
    def __init__(self, gemini_service):
        self.gemini = gemini_service
        self.schema_context = self._build_schema_context()
    
    def _build_schema_context(self) -> str:
        return """
Available tables (SQLite):

1. conversation_log  every query and Oracle response
   columns: id, timestamp, role, content, intent, day_pillar, officer_type,
            weather_condition, temperature_c, venues_recommended

2. venue_visits  all venues recommended by the Oracle
   columns: id, visit_date, venue_name, venue_address, venue_sector,
            day_pillar, officer_type, weather_condition, temperature_c,
            energy_rating, would_return, activity_type, notes

3. daily_activities  logged activities
   columns: id, activity_date, activity_name, primary_element, duration_minutes,
            meaning_score, energy_score

4. skill_practice  skill tracking
   columns: id, practice_date, skill_name, hours, category

5. life_events  significant events
   columns: id, event_date, event_type, event_description, luck_cycle

6. garmin_activities  activities imported from Garmin (wearable / GPS watch)
   columns: id, activity_date, activity_name, activity_type, primary_element,
            secondary_element, duration_minutes, distance_km, avg_hr, max_hr,
            calories, intensity (light/moderate/high), synced_at
   examples: Pickleball, Pilates, Running, Yoga, Breathwork

7. lab_reports  uploaded PDF health reports (Innoquest, VO2 Master, SPOT-MAS)
   columns: id, collected_at, source, report_type, raw_pdf_name

8. lab_markers  individual lab test results extracted from reports
   columns: id, lab_report_id, marker_code, marker_name, value, unit,
            ref_low, ref_high, flag (optimal/suboptimal/normal/abnormal), collected_at
   examples: LDL_C, LP_A, HBA1C, VIT_D, VO2_MAX, HOMOCYSTEINE

9. garmin_daily_stats  daily health metrics from Garmin
   columns: id, stat_date, steps, sleep_hours, sleep_score, resting_hr,
            body_battery, avg_stress, total_calories, active_calories

SQLite patterns:
- Case-insensitive search: content LIKE '%keyword%' COLLATE NOCASE
- Date filter: activity_date >= date('now', '-7 days')  (for "this week")
- Always include: ORDER BY ... DESC LIMIT 10
- Booleans are INTEGER: 1=TRUE, 0=FALSE
- For movement/activity questions, prefer garmin_activities over daily_activities
"""
    
    def generate_sql(self, question: str, user_context: dict = None) -> Optional[str]:
        # Detect Garmin queries and add contextual data summaries
        garmin_keywords = ["garmin", "activities", "workout", "exercise", "steps", "sleep",
                          "heart rate", "body battery", "stress", "running", "yoga", "pickleball",
                          "calories", "intensity", "how many", "how much", "what activities",
                          "training", "fitness", "moving", "active", "recovery", "about my"]
        is_garmin_query = any(kw in question.lower() for kw in garmin_keywords)

        # HARDCODED SQL TEMPLATES for common Garmin queries (more reliable than LLM generation)
        q_lower = question.lower()
        if is_garmin_query:
            # Check date-specific queries FIRST (these take precedence)
            # THIS WEEK (Monday to Sunday calendar week)
            if any(kw in q_lower for kw in ["this week", "this week's"]):
                return """
                SELECT activity_name, activity_date, duration_minutes, distance_km,
                       calories, avg_hr, max_hr, intensity, primary_element
                FROM garmin_activities
                WHERE activity_date >= date('now', 'weekday 1')
                AND activity_date <= date('now', 'weekday 1', '+6 days')
                ORDER BY activity_date DESC;
                """
            # LAST WEEK (Monday to Sunday previous calendar week)
            elif any(kw in q_lower for kw in ["last week", "last week's"]):
                return """
                SELECT activity_name, activity_date, duration_minutes, distance_km,
                       calories, avg_hr, max_hr, intensity, primary_element
                FROM garmin_activities
                WHERE activity_date >= date('now', 'weekday 1', '-7 days')
                AND activity_date < date('now', 'weekday 1')
                ORDER BY activity_date DESC;
                """
            # Sleep and training question (only if no date-specific query matched)
            elif any(kw in q_lower for kw in ["sleep", "about my", "health"]):
                return """
                SELECT 'Daily Stats' as type, stat_date as date,
                       steps, sleep_hours, body_battery, resting_hr, avg_stress
                FROM garmin_daily_stats
                WHERE stat_date >= date('now', '-7 days')
                ORDER BY stat_date DESC;
                """
            # Activities and movement (default: recent)
            elif any(kw in q_lower for kw in ["activity", "activities", "workout", "moving", "active", "exercise", "how am i doing", "how have i been"]):
                return """
                SELECT activity_name, activity_date, duration_minutes, distance_km,
                       calories, avg_hr, max_hr, intensity, primary_element
                FROM garmin_activities
                WHERE activity_date >= date('now', '-14 days')
                ORDER BY activity_date DESC;
                """
            # Heart rate specific
            elif "heart rate" in q_lower or "hr" in q_lower:
                return """
                SELECT activity_name, activity_date, avg_hr, max_hr, duration_minutes, calories
                FROM garmin_activities
                WHERE avg_hr IS NOT NULL
                AND activity_date >= date('now', '-14 days')
                ORDER BY activity_date DESC;
                """

        garmin_context = ""
        if is_garmin_query and user_context:
            # Build a summary of what's actually in the Garmin tables
            # This helps the LLM understand the data better
            garmin_context = """
NOTE: This query is about Garmin data. Here's what's available:
- garmin_activities: ~10-50 rows per user, contains activity_name (Pickleball, Running, Yoga, etc.),
  activity_date, duration_minutes, distance_km, avg_hr, max_hr, calories, intensity (light/moderate/high),
  primary_element (Fire, Wood, etc.)
- garmin_daily_stats: daily snapshots with steps, sleep_hours, sleep_score, resting_hr,
  body_battery (0-100), avg_stress (0-100), stat_date
- IMPORTANT: When asked about recent data, always filter by date (last 7, 14, or 30 days)
"""

        prompt = f"""Convert this question to a SQLite SELECT query.

Question: {question}

{self.schema_context}
{garmin_context}

Today: {date.today()}

RULES:
1. Return ONLY the SQL query, nothing else
2. SELECT only  no INSERT/UPDATE/DELETE
3. Use LIKE with COLLATE NOCASE for text search
4. Always include LIMIT (default 10)
5. Use date('now') for current date
6. For Garmin queries, order by activity_date or stat_date DESC to show most recent first

Return just the SQL, no markdown, no explanation."""

        result = self.gemini(prompt)
        if not result:
            return None
        sql = result.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[1]
        sql = sql.strip().rstrip(";") + ";"
        if not sql.upper().strip().startswith("SELECT"):
            print(f"Rejected non-SELECT query: {sql}")
            return None
        return sql
    
    def format_results(self, results: List[Dict], question: str, sql: str) -> str:
        if not results:
            # Check if this is a health/activity query
            health_keywords = ["garmin", "activities", "workout", "exercise", "steps", "sleep",
                              "training", "fitness", "moving", "active", "heart rate", "this week",
                              "last week", "activity", "pickleball", "running", "yoga"]
            if any(kw in question.lower() for kw in health_keywords):
                return (
                    "No activities found for the specified time period.\n\n"
                    "This is normal if:\n"
                    "- You haven't synced recent Garmin data yet\n"
                    "- No activities fall within the selected week\n\n"
                    "Try: 'Show me my recent activities' or check the [GARMIN] panel to sync data."
                )
            return (
                "No results found in the database for that query.\n\n"
                "Note: Venue history is only saved after recommendations are made. "
                "Try asking for a venue recommendation first, then check history."
            )

        cleaned = []
        for row in results:
            cleaned_row = {}
            for k, v in row.items():
                cleaned_row[k] = v.isoformat() if isinstance(v, (datetime, date)) else v
            cleaned.append(cleaned_row)

        results_json = json.dumps(cleaned, indent=2, default=str)

        # Detect if this is a Garmin analysis query
        is_garmin = any(kw in question.lower() for kw in ["garmin", "activities", "workout",
                                                            "steps", "sleep", "heart rate", "intensity"])

        analysis_context = ""
        if is_garmin:
            analysis_context = """
This is Garmin/activity data analysis. Provide insights about:
- Activity frequency (how often they work out)
- Activity types and Five Element distribution (Fire, Wood, Water, etc.)
- Intensity patterns (mostly light/moderate/high?)
- Sleep and daily stats trends
- Recovery and body battery patterns (if available)
- Heart rate zones and aerobic fitness hints (if avg_hr data present)"""

        prompt = f"""Answer this question based on the database results.

Question: {question}
Results ({len(results)} rows):
{results_json}
{analysis_context}

Instructions:
- Answer directly and specifically using the data
- Reference actual names, dates, numbers from results
- Be concise (3-5 sentences)
- No greetings, start with substance
- If analyzing activities, highlight patterns and trends
- For health metrics, note any concerning trends or positive progress"""

        return self.gemini(prompt) or "Results retrieved but formatting failed."


# =============================================================================
# SQL NODE FOR LANGGRAPH
# =============================================================================
def make_sql_node(svc, db: OracleDB):
    nl_to_sql = NLtoSQL(svc.gemini)

    def sql_query_node(state: dict) -> dict:
        query = state["query"]
        print("  [sql] Running memory/history query...")

        # Special handling for zone breakdown queries
        zone_query_keywords = ["zone", "duration", "summary", "total", "breakdown", "pickleball"]
        if all(kw in query.lower() for kw in ["zone", "duration"]) or \
           (any(kw in query.lower() for kw in zone_query_keywords) and
            any(kw in query.lower() for kw in ["summary", "total", "breakdown", "time in", "spent in", "how much"])):
            print("  [sql] Detected zone breakdown query - calculating zones...")

            # Extract activity filter (if any)
            activity_filter = ""
            if "pickleball" in query.lower():
                activity_filter = "AND activity_name LIKE '%pickleball%' COLLATE NOCASE"

            # Extract date range
            date_filter = ""
            if "last week" in query.lower():
                date_filter = "AND activity_date >= date('now', 'weekday 1', '-7 days') AND activity_date < date('now', 'weekday 1')"
            elif "this week" in query.lower():
                date_filter = "AND activity_date >= date('now', 'weekday 1') AND activity_date <= date('now', 'weekday 1', '+6 days')"
            else:
                date_filter = "AND activity_date >= date('now', '-7 days')"

            # Query activities
            sql = f"""
                SELECT activity_name, activity_date, avg_hr, duration_minutes
                FROM garmin_activities
                WHERE avg_hr IS NOT NULL
                {activity_filter}
                {date_filter}
                ORDER BY activity_date DESC;
            """

            print(f"  [sql] Zone query: {sql}")
            results = db.execute_query(sql)

            if not results:
                response = f"No {activity_filter.replace('AND activity_name LIKE ', '').replace('%', '')} activities found in the specified period."
                return {"response": response, "history": [{"role": "assistant", "content": response}]}

            # Get personalized zones from lab markers
            from garmin_sync import get_personalized_zones
            zones = get_personalized_zones()
            vt1 = zones.get("vt1_bpm")
            vt2 = zones.get("vt2_bpm")

            if not vt1 or not vt2:
                response = "Cannot calculate zones - no VT1/VT2 threshold data. Please upload a VO2 max test result."
                return {"response": response, "history": [{"role": "assistant", "content": response}]}

            # Calculate zone distribution
            zone1_minutes = 0
            zone2_minutes = 0
            zone3_minutes = 0
            activity_details = []

            for row in results:
                avg_hr = row.get("avg_hr")
                duration = row.get("duration_minutes")
                activity_name = row.get("activity_name")
                activity_date = row.get("activity_date")

                if avg_hr < vt1:
                    zone = 1
                    zone1_minutes += duration
                elif avg_hr <= vt2:
                    zone = 2
                    zone2_minutes += duration
                else:
                    zone = 3
                    zone3_minutes += duration

                activity_details.append({
                    "date": activity_date,
                    "activity": activity_name,
                    "hr": avg_hr,
                    "zone": zone,
                    "duration": duration
                })

            # Format data summary
            total_minutes = zone1_minutes + zone2_minutes + zone3_minutes
            activity_label = activity_filter.replace("AND activity_name LIKE ", "").replace("%", "").strip() if activity_filter else "All activities"

            data_summary = f"""Zone Breakdown ({activity_label})

Your personalized zones (from VO2 max test): Zone 1 (<{vt1} bpm) | Zone 2 ({vt1}-{vt2} bpm) | Zone 3+ (>{vt2} bpm)

Total Duration: {total_minutes} minutes ({total_minutes/60:.1f} hours)

Zone 1 (<{vt1} bpm): {zone1_minutes} minutes ({zone1_minutes/total_minutes*100:.1f}%) - Recovery pace
Zone 2 ({vt1}-{vt2} bpm): {zone2_minutes} minutes ({zone2_minutes/total_minutes*100:.1f}%) - Aerobic base building
Zone 3+ (>{vt2} bpm): {zone3_minutes} minutes ({zone3_minutes/total_minutes*100:.1f}%) - High-intensity work

Activity Breakdown:"""
            for detail in activity_details:
                data_summary += f"\n  {detail['date']} | {detail['activity']:25} | {detail['hr']:3} bpm (Zone {detail['zone']}) | {detail['duration']:3} min"

            # Pass zone data to LLM for actual training insights
            try:
                zone2_pct = zone2_minutes / total_minutes * 100
                zone3_pct = zone3_minutes / total_minutes * 100
                zone1_pct = zone1_minutes / total_minutes * 100
                analysis_prompt = f"""You are a sports science coach analyzing training data for a user whose personalized heart rate zones are:
- Zone 1 (Recovery): <{vt1} bpm
- Zone 2 (Aerobic Base): {vt1}-{vt2} bpm
- Zone 3+ (High Intensity): >{vt2} bpm

Their recent training distribution:
{data_summary}

Provide a concise training analysis (3-5 sentences) covering:
1. Whether this zone distribution is optimal (ideal is ~80% Zone 2, ~20% Zone 3+, minimal Zone 1 for trained athletes — or 80/20 rule)
2. What the current distribution means for their fitness development
3. One specific, actionable recommendation for their next session
4. Any red flags or positives worth highlighting

Be direct and specific. Reference actual percentages and minutes from the data."""

                llm_analysis = svc.deepseek(analysis_prompt)
                answer = data_summary + f"\n\n---\n**Training Analysis:**\n{llm_analysis}"
            except Exception:
                answer = data_summary

            print(f"\n{'='*65}\nZONE BREAKDOWN\n{'='*65}\n{answer}\n{'-'*65}")
            return {"response": answer, "history": [{"role": "assistant", "content": answer}]}

        sql = nl_to_sql.generate_sql(query, user_context=state)
        
        if not sql:
            response = (
                "I couldn't formulate the right database query. "
                "Try phrasing like: 'Which venues have been recommended?' "
                "or 'Show my recent conversations.'"
            )
            return {"response": response, "history": [{"role": "assistant", "content": response}]}
        
        print(f"  [sql] Query: {sql}")
        results = db.execute_query(sql)
        
        if results is None:
            response = "Database query failed. The query may be malformed."
            return {"response": response, "history": [{"role": "assistant", "content": response}]}
        
        print(f"  [sql] {len(results)} results returned")
        answer = nl_to_sql.format_results(results, query, sql)
        
        print(f"\n{'='*65}\nMEMORY RECALL\n{'='*65}\n{answer}\n{'-'*65}")
        return {"response": answer, "history": [{"role": "assistant", "content": answer}]}
    
    return sql_query_node


# =============================================================================
# CONVERSATION LOGGER WRAPPER
# =============================================================================
def wrap_with_logging(compiled_graph, db: OracleDB):
    class LoggedGraph:
        def __init__(self, graph, database):
            self.graph = graph
            self.db = database
        
        def invoke(self, initial_state: dict):
            # Log user message
            self.db.log_conversation(
                role="user",
                content=initial_state["query"],
                intent=None,
                weather=initial_state.get("weather")
            )
            
            # Execute graph
            result = self.graph.invoke(initial_state)
            
            # Log assistant response with full context
            meta    = result.get("meta") or {}
            weather = result.get("weather")
            intent  = result.get("intent")
            
            # Extract venue names from response if it was a recommendation
            venues_list = None
            response_text = result.get("response", "")
            if intent == "recommend" and "" in response_text:
                # Parse venue name from markdown heading ###  VenueName
                import re
                venue_matches = re.findall(r"###\s*\s*(.+)", response_text)
                if venue_matches:
                    venues_list = [{"name": v.strip()} for v in venue_matches]

            self.db.log_conversation(
                role="assistant",
                content=response_text,
                intent=intent,
                day_pillar=meta.get("day_pillar"),
                stem=meta.get("stem"),
                branch=meta.get("branch"),
                officer_type=meta.get("officer_name"),
                auspicious_sector=meta.get("auspicious_sector"),
                weather=weather,
                venues_recommended=venues_list
            )
            
            return result
    
    return LoggedGraph(compiled_graph, db)


if __name__ == "__main__":
    print("Testing Strategic Oracle SQLite Integration...\n")
    db = OracleDB("test_oracle.db")
    print("[OK] Database connected")
    
    db.log_conversation(role="user", content="Where should I go for coffee?", intent="recommend", day_pillar="")
    db.log_conversation(role="assistant", content="Try Tiong Bahru Bakery.", intent="recommend", day_pillar="")
    db.log_venue_visit(venue_name="Tiong Bahru Bakery", venue_sector="SE", day_pillar="", officer_type="Open")
    
    stats = db.get_stats()
    print("Stats:", stats)
    venues = db.get_all_venues()
    print("Venues:", venues)
    
    import os
    os.remove("test_oracle.db")
    print("[OK] All tests passed!")