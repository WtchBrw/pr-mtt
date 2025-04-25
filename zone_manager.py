# zone_manager.py

import json
import os

class ZoneManager:
    def __init__(self, config_file='zones_config.json'):
        self.config_file = config_file
        self.zones = []
        self.load_zones()

    def load_zones(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.zones = json.load(f)
        else:
            self.zones = []

    def save_zones(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.zones, f, indent=4)

    def add_zone(self, name, center_lat, center_lon, radius_meters):
        zone = {
            'name': name,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'radius_meters': radius_meters
        }
        self.zones.append(zone)
        self.save_zones()

    def remove_zone(self, name):
        self.zones = [z for z in self.zones if z['name'] != name]
        self.save_zones()

    def get_zones(self):
        return self.zones

    def is_target_in_zone(self, target_lat, target_lon):
        triggered = []
        for zone in self.zones:
            for i in range (0, len(self.zones[zone])):
                distance = self.haversine(self.zones[zone][i]["center_lon"], self.zones[zone][i]["center_lat"], target_lon, target_lat)
                if distance <= self.zones[zone][i]["radius_meters"]:
                    triggered.append(self.zones[zone][i]["name"])
        return triggered

    def haversine(self, lon1, lat1, lon2, lat2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
