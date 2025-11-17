"""Unit tests for geometry utilities."""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import (
    min_shift_to_resolve_overlap,
    bbox_from_mask,
    compute_overlap_pixels,
    draw_rounded_rectangle_mask
)


class TestGeometry(unittest.TestCase):
    """Test geometry utility functions."""
    
    def test_min_shift_no_overlap(self):
        """Test min_shift when boxes don't overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 100, 100)
        
        dx_top, dy_top, dx_bottom, dy_bottom = min_shift_to_resolve_overlap(bbox1, bbox2)
        
        self.assertEqual(dx_top, 0.0)
        self.assertEqual(dy_top, 0.0)
        self.assertEqual(dx_bottom, 0.0)
        self.assertEqual(dy_bottom, 0.0)
    
    def test_min_shift_horizontal_overlap(self):
        """Test min_shift for horizontal overlap."""
        # bbox1: (0, 0, 100, 100)
        # bbox2: (50, 0, 100, 100) - overlaps by 50px horizontally
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 0, 100, 100)
        
        dx_top, dy_top, dx_bottom, dy_bottom = min_shift_to_resolve_overlap(bbox1, bbox2)
        
        # Should suggest shifting right by 50px or left by 50px
        self.assertGreater(abs(dx_top), 0)
        self.assertEqual(dy_top, 0.0)
    
    def test_min_shift_vertical_overlap(self):
        """Test min_shift for vertical overlap."""
        # bbox1: (0, 0, 100, 100)
        # bbox2: (0, 50, 100, 100) - overlaps by 50px vertically
        bbox1 = (0, 0, 100, 100)
        bbox2 = (0, 50, 100, 100)
        
        dx_top, dy_top, dx_bottom, dy_bottom = min_shift_to_resolve_overlap(bbox1, bbox2)
        
        # Should suggest shifting down by 50px or up by 50px
        self.assertEqual(dx_top, 0.0)
        self.assertGreater(abs(dy_top), 0)
    
    def test_min_shift_complete_overlap(self):
        """Test min_shift when one box completely contains another."""
        # bbox1: (0, 0, 200, 200)
        # bbox2: (50, 50, 100, 100) - completely inside bbox1
        bbox1 = (0, 0, 200, 200)
        bbox2 = (50, 50, 100, 100)
        
        dx_top, dy_top, dx_bottom, dy_bottom = min_shift_to_resolve_overlap(bbox1, bbox2)
        
        # Should suggest a shift
        self.assertGreater(abs(dx_top) + abs(dy_top) + abs(dx_bottom) + abs(dy_bottom), 0)
    
    def test_bbox_from_mask(self):
        """Test bbox extraction from mask."""
        # Create a mask with a rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255
        
        x, y, w, h = bbox_from_mask(mask)
        
        self.assertEqual(x, 30)
        self.assertEqual(y, 20)
        self.assertEqual(w, 40)
        self.assertEqual(h, 60)
    
    def test_bbox_from_empty_mask(self):
        """Test bbox extraction from empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        x, y, w, h = bbox_from_mask(mask)
        
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(w, 0)
        self.assertEqual(h, 0)
    
    def test_compute_overlap_pixels(self):
        """Test overlap pixel computation."""
        # Create two overlapping masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:60, 0:60] = 255
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[40:100, 40:100] = 255
        
        overlap = compute_overlap_pixels(mask1, mask2)
        
        # Overlap region: (40:60, 40:60) = 20x20 = 400 pixels
        self.assertEqual(overlap, 400)
    
    def test_compute_overlap_no_overlap(self):
        """Test overlap computation when masks don't overlap."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:50, 0:50] = 255
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[60:100, 60:100] = 255
        
        overlap = compute_overlap_pixels(mask1, mask2)
        
        self.assertEqual(overlap, 0)
    
    def test_draw_rounded_rectangle_mask(self):
        """Test drawing rounded rectangle on mask."""
        mask = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        
        result = draw_rounded_rectangle_mask(mask, 10, 10, 50, 50, 10, color)
        
        # Check that some pixels were drawn
        self.assertGreater(np.sum(result), 0)
        
        # Check that corners are rounded (not all pixels in corner are filled)
        # Top-left corner should have some transparent pixels
        corner_sum = np.sum(result[10:15, 10:15])
        self.assertLess(corner_sum, 255 * 3 * 25)  # Not completely filled
    
    def test_draw_rectangle_no_radius(self):
        """Test drawing rectangle without border radius."""
        mask = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        
        result = draw_rounded_rectangle_mask(mask, 10, 10, 50, 50, 0, color)
        
        # Should draw a simple rectangle
        self.assertGreater(np.sum(result), 0)


if __name__ == '__main__':
    unittest.main()

