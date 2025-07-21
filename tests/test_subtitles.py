import json
import os
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from pysubs2 import SSAEvent, SSAFile

# Assuming the module is imported as subtitle_tool
from subtitle_tool.subtitles import (
    SubtitleEvent,
    SubtitleValidationException,
    equalize_subtitles,
    events_to_subtitles,
    merge_subtitle_events,
    save_to_json,
    subtitles_to_dict,
    subtitles_to_events,
    validate_subtitles,
)


class TestSubtitleEvent(unittest.TestCase):
    """Test the SubtitleEvent model"""

    def test_subtitle_event_creation(self):
        """Test creating a SubtitleEvent with valid data"""
        event = SubtitleEvent(start=1000, end=2000, text="Hello world")
        self.assertEqual(event.start, 1000)
        self.assertEqual(event.end, 2000)
        self.assertEqual(event.text, "Hello world")

    def test_subtitle_event_validation(self):
        """Test SubtitleEvent field validation"""
        # Test with missing fields - should raise ValidationError
        with self.assertRaises(ValidationError):
            SubtitleEvent(start=1000)  # type: ignore # missing end and text


class TestSubtitlesToEvents(unittest.TestCase):
    """Test subtitles_to_events function"""

    def test_empty_subtitle_file(self):
        """Test with empty subtitle file"""
        subtitle = SSAFile()
        result = subtitles_to_events(subtitle)
        self.assertEqual(result, [])

    def test_single_subtitle_event(self):
        """Test with single subtitle event"""
        subtitle = SSAFile()
        subtitle.events = [SSAEvent(start=1000, end=2000, text="Hello")]

        result = subtitles_to_events(subtitle)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 1000)
        self.assertEqual(result[0].end, 2000)
        self.assertEqual(result[0].text, "Hello")
        self.assertIsInstance(result[0], SubtitleEvent)

    def test_multiple_subtitle_events(self):
        """Test with multiple subtitle events"""
        subtitle = SSAFile()
        subtitle.events = [
            SSAEvent(start=1000, end=2000, text="First"),
            SSAEvent(start=3000, end=4000, text="Second"),
            SSAEvent(start=5000, end=6000, text="Third"),
        ]

        result = subtitles_to_events(subtitle)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].text, "First")
        self.assertEqual(result[1].text, "Second")
        self.assertEqual(result[2].text, "Third")


class TestSubtitlesToDict(unittest.TestCase):
    """Test subtitles_to_dict function"""

    def test_empty_subtitle_file(self):
        """Test with empty subtitle file"""
        subtitle = SSAFile()
        result = subtitles_to_dict(subtitle)
        self.assertEqual(result, [])

    def test_single_subtitle_event(self):
        """Test with single subtitle event"""
        subtitle = SSAFile()
        subtitle.events = [SSAEvent(start=1000, end=2000, text="Hello")]

        result = subtitles_to_dict(subtitle)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {"start": 1000, "end": 2000, "text": "Hello"})
        self.assertIsInstance(result[0], dict)

    def test_multiple_subtitle_events(self):
        """Test with multiple subtitle events"""
        subtitle = SSAFile()
        subtitle.events = [
            SSAEvent(start=1000, end=2000, text="First"),
            SSAEvent(start=3000, end=4000, text="Second"),
        ]

        result = subtitles_to_dict(subtitle)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"start": 1000, "end": 2000, "text": "First"})
        self.assertEqual(result[1], {"start": 3000, "end": 4000, "text": "Second"})


class TestEventsToSubtitles(unittest.TestCase):
    """Test events_to_subtitles function"""

    def test_empty_events_list(self):
        """Test with empty events list"""
        result = events_to_subtitles([])
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 0)

    def test_single_event(self):
        """Test with single subtitle event"""
        events = [SubtitleEvent(start=1000, end=2000, text="Hello")]

        result = events_to_subtitles(events)

        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].start, 1000)
        self.assertEqual(result.events[0].end, 2000)
        self.assertEqual(result.events[0].text, "Hello")
        self.assertIsInstance(result.events[0], SSAEvent)

    def test_multiple_events(self):
        """Test with multiple subtitle events"""
        events = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
        ]

        result = events_to_subtitles(events)

        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.events[0].text, "First")
        self.assertEqual(result.events[1].text, "Second")


class TestValidateSubtitles(unittest.TestCase):
    """Test validate_subtitles function"""

    def test_valid_subtitles(self):
        """Test with valid, non-overlapping subtitles"""
        subtitles = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
            SubtitleEvent(start=5000, end=6000, text="Third"),
        ]
        duration = 10.0  # 10 seconds

        # Should not raise any exception
        validate_subtitles(subtitles, duration)

    def test_subtitle_exceeds_duration(self):
        """Test when last subtitle exceeds video duration"""
        subtitles = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=12000, text="Second"),  # 12 seconds
        ]
        duration = 10.0  # 10 seconds

        with self.assertRaises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        self.assertIn("Subtitle ends at 12000", str(exc_info.exception))
        self.assertIn("while audio segment ends at 10000.0", str(exc_info.exception))

    def test_subtitle_start_after_end(self):
        """Test when subtitle start time is after end time"""
        subtitles = [SubtitleEvent(start=2000, end=1000, text="Invalid")]  # start > end
        duration = 10.0

        with self.assertRaises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        self.assertIn("starts at 2000", str(exc_info.exception))
        self.assertIn("but ends at 1000", str(exc_info.exception))

    def test_overlapping_subtitles(self):
        """Test when subtitles overlap"""
        subtitles = [
            SubtitleEvent(start=1000, end=3000, text="First"),
            SubtitleEvent(start=2000, end=4000, text="Second"),  # overlaps with first
        ]
        duration = 10.0

        with self.assertRaises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        self.assertIn("starts at 2000", str(exc_info.exception))
        self.assertIn("previous subtitle finishes at 3000", str(exc_info.exception))

    def test_adjacent_subtitles(self):
        """Test subtitles that are adjacent (end of one = start of next)"""
        subtitles = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=2000, end=3000, text="Second"),  # adjacent
        ]
        duration = 10.0

        # Should not raise exception
        validate_subtitles(subtitles, duration)

    def test_single_subtitle(self):
        """Test with single subtitle"""
        subtitles = [SubtitleEvent(start=1000, end=2000, text="Only one")]
        duration = 10.0

        # Should not raise exception
        validate_subtitles(subtitles, duration)

    def test_no_subtitle(self):
        """Test with no subtitle generated"""
        subtitles = []
        duration = 10.0

        # Should not raise exception
        validate_subtitles(subtitles, duration)


class TestSaveToJson(unittest.TestCase):
    """Test save_to_json function"""

    def test_save_empty_subtitles(self):
        """Test saving empty subtitles list"""
        subtitles = []

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            save_to_json(subtitles, temp_path)

            # Read back and verify
            with open(temp_path) as f:
                content = json.load(f)

            self.assertEqual(content, [])
        finally:
            os.unlink(temp_path)

    def test_save_single_subtitle(self):
        """Test saving single subtitle"""
        subtitles = [SubtitleEvent(start=1000, end=2000, text="Hello")]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            save_to_json(subtitles, temp_path)

            # Read back and verify
            with open(temp_path) as f:
                content = json.load(f)

            self.assertEqual(len(content), 1)
            self.assertEqual(content[0], {"start": 1000, "end": 2000, "text": "Hello"})
        finally:
            os.unlink(temp_path)

    def test_save_multiple_subtitles(self):
        """Test saving multiple subtitles"""
        subtitles = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            save_to_json(subtitles, temp_path)

            # Read back and verify
            with open(temp_path) as f:
                content = json.load(f)

            self.assertEqual(len(content), 2)
            self.assertEqual(content[0], {"start": 1000, "end": 2000, "text": "First"})
            self.assertEqual(content[1], {"start": 3000, "end": 4000, "text": "Second"})
        finally:
            os.unlink(temp_path)


class TestMergeSubtitleEvents(unittest.TestCase):
    """Test merge_subtitle_events function"""

    def test_merge_empty_groups(self):
        """Test merging empty subtitle groups"""
        subtitle_groups = []
        segment_durations = []

        with self.assertRaises(ValueError) as cm:
            merge_subtitle_events(subtitle_groups, segment_durations)
        self.assertIn("segment_durations cannot be empty", str(cm.exception))

    def test_merge_more_subtitle_groups_than_durations(self):
        """Test merging when there are more subtitle_groups than segment_durations"""
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Group 1")],
            [SubtitleEvent(start=1000, end=2000, text="Group 2")],
        ]
        segment_durations = [3.0 * 1000]  # One duration less than groups

        with self.assertRaises(ValueError) as cm:
            merge_subtitle_events(subtitle_groups, segment_durations)
        self.assertIn(
            "Number of subtitle groups (2) must match number of segment durations (1)",
            str(cm.exception),
        )

    def test_merge_subtitle_groups_but_no_durations(self):
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Group 1")],
            [SubtitleEvent(start=1000, end=2000, text="Group 2")],
        ]
        segment_durations = []

        with self.assertRaises(ValueError) as cm:
            merge_subtitle_events(subtitle_groups, segment_durations)
        self.assertIn(
            "segment_durations cannot be empty",
            str(cm.exception),
        )

    def test_merge_durations_but_no_subtitle_groups(self):
        subtitle_groups = []
        segment_durations = [3 * 0.5]

        with self.assertRaises(ValueError) as cm:
            merge_subtitle_events(subtitle_groups, segment_durations)
        self.assertIn(
            "subtitle_groups cannot be empty",
            str(cm.exception),
        )

    def test_merge_zero_length_segment_durations(self):
        """Test merging when segment_durations is of zero length"""
        subtitle_groups = [[SubtitleEvent(start=1000, end=2000, text="Group 1")]]
        segment_durations = []

        with self.assertRaises(ValueError) as cm:
            merge_subtitle_events(subtitle_groups, segment_durations)
        self.assertIn("segment_durations cannot be empty", str(cm.exception))

    def test_merge_single_group(self):
        """Test merging single group (no time adjustment needed)"""
        subtitle_groups = [
            [
                SubtitleEvent(start=1000, end=2000, text="First"),
                SubtitleEvent(start=3000, end=4000, text="Second"),
            ]
        ]
        segment_durations = [5.0]

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].start, 1000)  # No time shift
        self.assertEqual(result[0].end, 2000)
        self.assertEqual(result[1].start, 3000)
        self.assertEqual(result[1].end, 4000)

    def test_merge_two_groups(self):
        """Test merging two subtitle groups with time adjustment"""
        subtitle_groups = [
            [
                SubtitleEvent(start=1000, end=2000, text="First segment - first"),
                SubtitleEvent(start=3000, end=4000, text="First segment - second"),
            ],
            [
                SubtitleEvent(start=1000, end=2000, text="Second segment - first"),
                SubtitleEvent(start=3000, end=4000, text="Second segment - second"),
            ],
        ]
        segment_durations = [
            5.0 * 1000,
            5.0 * 1000,
        ]  # Durations are in milliseconds, 5s each.

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        self.assertEqual(len(result), 4)

        # First group should remain unchanged
        self.assertEqual(result[0].start, 1000)
        self.assertEqual(result[0].end, 2000)
        self.assertEqual(result[1].start, 3000)
        self.assertEqual(result[1].end, 4000)

        # Second group should be shifted by 5 seconds (5000ms)
        self.assertEqual(result[2].start, 6000)  # 1000 + 5000
        self.assertEqual(result[2].end, 7000)  # 2000 + 5000
        self.assertEqual(result[3].start, 8000)  # 3000 + 5000
        self.assertEqual(result[3].end, 9000)  # 4000 + 5000

    def test_merge_three_groups(self):
        """Test merging three subtitle groups"""
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Group 1")],
            [SubtitleEvent(start=1000, end=2000, text="Group 2")],
            [SubtitleEvent(start=1000, end=2000, text="Group 3")],
        ]
        segment_durations = [3.0 * 1000, 4.0 * 1000, 5.0 * 1000]

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].start, 1000)  # No shift
        self.assertEqual(result[1].start, 4000)  # Shifted by 3000ms
        self.assertEqual(result[2].start, 8000)  # Shifted by 3000 + 4000 = 7000ms

    def test_merge_with_validation_error(self):
        """Test that merge raises validation error for invalid result"""
        # Create subtitles that will exceed total duration after merging
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=15000, text="Too long")]  # 15 seconds
        ]
        segment_durations = [10.0]  # Only 10 seconds total

        with self.assertRaises(SubtitleValidationException):
            merge_subtitle_events(subtitle_groups, segment_durations)

    def test_merge_preserves_text_content(self):
        """Test that merging preserves all text content"""
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Hello")],
            [SubtitleEvent(start=1000, end=2000, text="World")],
        ]
        segment_durations = [3.0 * 1000, 3.0 * 1000]

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Hello")
        self.assertEqual(result[1].text, "World")

    def test_merge_clean_newlines_true(self):
        """Test that newlines are cleaned when clean_newlines is True"""
        subtitle_groups = [
            [
                SubtitleEvent(start=1000, end=2000, text="Line1\\NLine2"),
                SubtitleEvent(start=3000, end=4000, text="Line3\\nLine4"),
                SubtitleEvent(start=5000, end=6000, text="Line5\nLine6"),
            ]
        ]
        segment_durations = [7.0 * 1000]

        result = merge_subtitle_events(
            subtitle_groups, segment_durations, clean_newlines=True
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].text, "Line1 Line2")
        self.assertEqual(result[1].text, "Line3 Line4")
        self.assertEqual(result[2].text, "Line5 Line6")

    def test_merge_clean_newlines_false(self):
        """Test that newlines are preserved when clean_newlines is False"""
        subtitle_groups = [
            [
                SubtitleEvent(start=1000, end=2000, text="Line1\\NLine2"),
                SubtitleEvent(start=3000, end=4000, text="Line3\\nLine4"),
                SubtitleEvent(start=5000, end=6000, text="Line5\nLine6"),
            ]
        ]
        segment_durations = [7.0 * 1000]

        result = merge_subtitle_events(
            subtitle_groups, segment_durations, clean_newlines=False
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].text, "Line1\\NLine2")
        self.assertEqual(result[1].text, "Line3\\nLine4")
        self.assertEqual(result[2].text, "Line5\nLine6")


class TestSubtitleValidationException(unittest.TestCase):
    """Test SubtitleValidationException"""

    def test_exception_creation(self):
        """Test creating SubtitleValidationException"""
        msg = "Test error message"
        exc = SubtitleValidationException(msg)
        self.assertEqual(str(exc), msg)
        self.assertIsInstance(exc, Exception)


# Integration tests
class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""

    def test_round_trip_conversion(self):
        """Test converting SSAFile -> SubtitleEvent -> SSAFile"""
        # Create original SSAFile
        original = SSAFile()
        original.events = [
            SSAEvent(start=1000, end=2000, text="First"),
            SSAEvent(start=3000, end=4000, text="Second"),
        ]

        # Convert to events and back
        events = subtitles_to_events(original)
        converted_back = events_to_subtitles(events)

        # Verify conversion preserved data
        self.assertEqual(len(converted_back.events), len(original.events))
        for orig, conv in zip(original.events, converted_back.events, strict=False):
            self.assertEqual(orig.start, conv.start)
            self.assertEqual(orig.end, conv.end)
            self.assertEqual(orig.text, conv.text)

    def test_merge_and_validate_workflow(self):
        """Test complete workflow of merging and validating"""
        # Create subtitle groups
        group1 = [SubtitleEvent(start=1000, end=2000, text="Part 1")]
        group2 = [SubtitleEvent(start=1000, end=2000, text="Part 2")]

        subtitle_groups = [group1, group2]
        segment_durations = [3.0 * 1000, 3.0 * 1000]

        # Merge and validate
        merged = merge_subtitle_events(subtitle_groups, segment_durations)
        validate_subtitles(merged, sum(segment_durations))

        # Should complete without errors
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1].start, 4000)  # Shifted by 3000ms


class TestEqualizeSubtitles(unittest.TestCase):
    """Test equalize_subtitles function"""

    @patch("srt_equalizer.srt_equalizer.equalize_srt_file")
    @patch("pysubs2.SSAFile.load")
    @patch("pysubs2.SSAFile.save")
    def test_equalize_subtitles_default_params(
        self, mock_save, mock_load, mock_equalize_srt_file
    ):
        """Test with default line_length and method (halving)"""
        subtitle = SSAFile()
        subtitle.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long sentence that definitely needs "
                + "to be split into multiple lines for better readability on screen.",
            )
        )

        # Configure mock_load to return a modified SSAFile
        mock_loaded_ssa = SSAFile()
        mock_loaded_ssa.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long sentence that definitely\\Nneeds to "
                + "be split into multiple lines for better\\Nreadability on screen.",
            )
        )
        mock_load.return_value = mock_loaded_ssa

        result = equalize_subtitles(subtitle)

        # Assertions
        mock_save.assert_called_once()
        mock_equalize_srt_file.assert_called_once_with(
            mock_save.call_args[0][0],  # src_path
            mock_load.call_args[0][0],  # dst_path
            target_chars=42,
            method="halving",
        )
        mock_load.assert_called_once()
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(
            result.events[0].text,
            "This is a very long sentence that definitely\\Nneeds "
            + "to be split into multiple lines for better\\Nreadability on screen.",
        )

    @patch("srt_equalizer.srt_equalizer.equalize_srt_file")
    @patch("pysubs2.SSAFile.load")
    @patch("pysubs2.SSAFile.save")
    def test_equalize_subtitles_custom_line_length(
        self, mock_save, mock_load, mock_equalize_srt_file
    ):
        """Test with a custom line_length"""
        subtitle = SSAFile()
        subtitle.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long sentence that definitely "
                + "needs to be split into multiple lines for better "
                + "readability on screen.",
            )
        )

        custom_line_length = 20
        mock_loaded_ssa = SSAFile()
        mock_loaded_ssa.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long\\Nsentence that definitely\\Nneeds "
                + "to be split into\\Nmultiple lines for\\Nbetter "
                + "readability on\\Nscreen.",
            )
        )
        mock_load.return_value = mock_loaded_ssa

        result = equalize_subtitles(subtitle, line_length=custom_line_length)

        mock_save.assert_called_once()
        mock_equalize_srt_file.assert_called_once_with(
            mock_save.call_args[0][0],
            mock_load.call_args[0][0],
            target_chars=custom_line_length,
            method="halving",
        )
        mock_load.assert_called_once()
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(
            result.events[0].text,
            "This is a very long\\Nsentence that definitely\\Nneeds "
            + "to be split into\\Nmultiple lines for\\Nbetter "
            + "readability on\\Nscreen.",
        )

    @patch("srt_equalizer.srt_equalizer.equalize_srt_file")
    @patch("pysubs2.SSAFile.load")
    @patch("pysubs2.SSAFile.save")
    def test_equalize_subtitles_greedy_method(
        self, mock_save, mock_load, mock_equalize_srt_file
    ):
        """Test with the "greedy" method"""
        subtitle = SSAFile()
        subtitle.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long sentence that definitely needs"
                + "to be split into multiple lines for better readability on screen.",
            )
        )

        mock_loaded_ssa = SSAFile()
        mock_loaded_ssa.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a very long sentence that definitely\\Nneeds "
                + "to be split into multiple lines for better\\Nreadability on screen.",
            )
        )
        mock_load.return_value = mock_loaded_ssa

        result = equalize_subtitles(subtitle, method="greedy")

        mock_save.assert_called_once()
        mock_equalize_srt_file.assert_called_once_with(
            mock_save.call_args[0][0],
            mock_load.call_args[0][0],
            target_chars=42,
            method="greedy",
        )
        mock_load.assert_called_once()
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(
            result.events[0].text,
            "This is a very long sentence that definitely\\Nneeds "
            + "to be split into multiple lines for better\\Nreadability on screen.",
        )

    @patch("srt_equalizer.srt_equalizer.equalize_srt_file")
    @patch("pysubs2.SSAFile.load")
    @patch("pysubs2.SSAFile.save")
    def test_equalize_subtitles_punctuation_method(
        self, mock_save, mock_load, mock_equalize_srt_file
    ):
        """Test with the "punctuation" method"""
        subtitle = SSAFile()
        subtitle.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a sentence. This is another sentence! And a third one?",
            )
        )

        mock_loaded_ssa = SSAFile()
        mock_loaded_ssa.events.append(
            SSAEvent(
                start=0,
                end=5000,
                text="This is a sentence.\\NThis is another "
                + "sentence!\\NAnd a third one?",
            )
        )
        mock_load.return_value = mock_loaded_ssa

        result = equalize_subtitles(subtitle, method="punctuation")

        mock_save.assert_called_once()
        mock_equalize_srt_file.assert_called_once_with(
            mock_save.call_args[0][0],
            mock_load.call_args[0][0],
            target_chars=42,
            method="punctuation",
        )
        mock_load.assert_called_once()
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(
            result.events[0].text,
            "This is a sentence.\\NThis is another sentence!\\NAnd a third one?",
        )

    @patch("srt_equalizer.srt_equalizer.equalize_srt_file")
    @patch("pysubs2.SSAFile.load")
    @patch("pysubs2.SSAFile.save")
    def test_equalize_subtitles_empty_input(
        self, mock_save, mock_load, mock_equalize_srt_file
    ):
        """Test with an empty SSAFile input"""
        subtitle = SSAFile()

        mock_loaded_ssa = SSAFile()  # Empty SSAFile
        mock_load.return_value = mock_loaded_ssa

        result = equalize_subtitles(subtitle)

        mock_save.assert_called_once()
        mock_equalize_srt_file.assert_called_once_with(
            mock_save.call_args[0][0],
            mock_load.call_args[0][0],
            target_chars=42,
            method="halving",
        )
        mock_load.assert_called_once()
        self.assertIsInstance(result, SSAFile)
        self.assertEqual(len(result.events), 0)


if __name__ == "__main__":
    unittest.main()
