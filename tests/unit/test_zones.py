"""
Unit tests for zone management (core/zones.py)
"""

import pytest
import numpy as np
from core.zones import Zone, ZoneManager


class TestZone:
    """Test Zone class"""

    def test_zone_initialization(self):
        """Test zone initialization"""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        zone = Zone("test_zone", points, zone_type="entrance")

        assert zone.name == "test_zone"
        assert zone.zone_type == "entrance"
        assert len(zone.polygon) == 4

    def test_point_inside_zone(self):
        """Test point inside zone detection"""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        zone = Zone("test_zone", points)

        assert zone.contains_point(50, 50) is True
        assert zone.contains_point(150, 150) is False

    def test_bbox_intersects_zone(self):
        """Test bounding box intersection with zone"""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        zone = Zone("test_zone", points)

        # Bbox inside zone
        assert zone.intersects_bbox(25, 25, 75, 75) is True

        # Bbox outside zone
        assert zone.intersects_bbox(200, 200, 300, 300) is False

        # Bbox partially intersecting
        assert zone.intersects_bbox(75, 75, 150, 150) is True


class TestZoneManager:
    """Test ZoneManager class"""

    @pytest.fixture
    def zone_config(self):
        """Zone configuration fixture"""
        return {
            "entrance": {
                "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "type": "entrance"
            },
            "checkout": {
                "points": [[150, 150], [250, 150], [250, 250], [150, 250]],
                "type": "checkout"
            }
        }

    def test_zone_manager_initialization(self, zone_config):
        """Test ZoneManager initialization"""
        manager = ZoneManager(zone_config)

        assert len(manager.zones) == 2
        assert "entrance" in manager.zones
        assert "checkout" in manager.zones

    def test_get_zone_for_point(self, zone_config):
        """Test getting zone for a point"""
        manager = ZoneManager(zone_config)

        zone = manager.get_zone_for_point(50, 50)
        assert zone is not None
        assert zone.name == "entrance"

        zone = manager.get_zone_for_point(200, 200)
        assert zone is not None
        assert zone.name == "checkout"

        zone = manager.get_zone_for_point(500, 500)
        assert zone is None

    def test_get_zone_for_bbox(self, zone_config):
        """Test getting zone for a bounding box"""
        manager = ZoneManager(zone_config)

        zones = manager.get_zones_for_bbox(25, 25, 75, 75)
        assert len(zones) == 1
        assert zones[0].name == "entrance"

        zones = manager.get_zones_for_bbox(75, 75, 175, 175)
        # Should intersect both zones
        assert len(zones) >= 1
