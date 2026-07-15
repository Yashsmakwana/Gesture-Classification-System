# test_preprocessing.py
"""
Unit Tests for Keypoint Preprocessing Normalization
Verifies translation and scale invariance of coordinate preprocessing.
Execute using:
    python D:/Projects/Gesture-Classification-System/tests/test_preprocessing.py
"""
import unittest
import copy

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Convert to relative coordinates (origin at wrist - landmark 0)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 2. Flatten list
    flattened = []
    for lp in temp_landmark_list:
        flattened.extend(lp)

    # 3. Normalization (dividing by max absolute value)
    max_value = max(list(map(abs, flattened)))
    if max_value == 0:
        return flattened

    return [val / max_value for val in flattened]


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # A mock list of 21 landmarks representing hand skeleton (x, y)
        self.mock_hand = [
            [100.0, 150.0], [120.0, 140.0], [140.0, 130.0], [150.0, 120.0], [160.0, 110.0], # Wrist + Thumb
            [110.0, 100.0], [115.0, 90.0], [118.0, 85.0], [120.0, 80.0],               # Index
            [100.0, 95.0], [102.0, 85.0], [103.0, 80.0], [104.0, 75.0],                 # Middle
            [90.0, 100.0], [88.0, 90.0], [87.0, 85.0], [86.0, 80.0],                    # Ring
            [80.0, 110.0], [75.0, 105.0], [72.0, 100.0], [70.0, 95.0]                   # Pinky
        ]

    def test_translation_invariance(self):
        """Verifies that shifting the hand position doesn't alter normalized features."""
        original_processed = pre_process_landmark(self.mock_hand)
        
        # Translate the hand coordinates by shifting X and Y by offsets
        offset_x, offset_y = 50.0, -80.0
        translated_hand = [[pt[0] + offset_x, pt[1] + offset_y] for pt in self.mock_hand]
        translated_processed = pre_process_landmark(translated_hand)
        
        # Assert each element is equal (within double precision tolerance)
        for val1, val2 in zip(original_processed, translated_processed):
            self.assertAlmostEqual(val1, val2, places=6)

    def test_scale_invariance(self):
        """Verifies that scaling the hand size doesn't alter normalized features."""
        original_processed = pre_process_landmark(self.mock_hand)
        
        # Scale the hand coordinates by a factor of 2.5 (keeping float representation)
        scale_factor = 2.5
        scaled_hand = [[pt[0] * scale_factor, pt[1] * scale_factor] for pt in self.mock_hand]
        scaled_processed = pre_process_landmark(scaled_hand)
        
        # Assert each element is equal (within double precision tolerance)
        for val1, val2 in zip(original_processed, scaled_processed):
            self.assertAlmostEqual(val1, val2, places=6)

if __name__ == "__main__":
    unittest.main()
