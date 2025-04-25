# alert_engine.py

import datetime

class AlertEngine:
    def __init__(self, zone_manager, logger=None):
        self.zone_manager = zone_manager
        self.logger = logger

    def check_alerts(self, targets):
        alerts = []
        for tid, lat, lon in targets:
            zones_hit = self.zone_manager.is_target_in_zone(lat, lon)
            for zone_name in zones_hit:
                msg = f"[ALERT] Target ID {tid} entered zone '{zone_name}' at {datetime.datetime.now()}"
                alerts.append(msg)
                if self.logger:
                    self.logger(msg)
        return alerts
