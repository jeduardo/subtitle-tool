import json
import pytest
import tempfile
import os

from pydantic import ValidationError
from pysubs2 import SSAFile, SSAEvent

# Assuming the module is imported as subtitle_tool
from subtitle_tool.subtitles import (
    SubtitleEvent,
    SubtitleValidationException,
    subtitles_to_events,
    subtitles_to_dict,
    events_to_subtitles,
    validate_subtitles,
    save_to_json,
    merge_subtitle_events,
)


class TestSubtitleEvent:
    """Test the SubtitleEvent model"""

    def test_subtitle_event_creation(self):
        """Test creating a SubtitleEvent with valid data"""
        event = SubtitleEvent(start=1000, end=2000, text="Hello world")
        assert event.start == 1000
        assert event.end == 2000
        assert event.text == "Hello world"

    def test_subtitle_event_validation(self):
        """Test SubtitleEvent field validation"""
        # Test with missing fields - should raise ValidationError
        with pytest.raises(ValidationError):
            SubtitleEvent(start=1000)  # type: ignore # missing end and text


class TestSubtitlesToEvents:
    """Test subtitles_to_events function"""

    def test_empty_subtitle_file(self):
        """Test with empty subtitle file"""
        subtitle = SSAFile()
        result = subtitles_to_events(subtitle)
        assert result == []

    def test_single_subtitle_event(self):
        """Test with single subtitle event"""
        subtitle = SSAFile()
        subtitle.events = [SSAEvent(start=1000, end=2000, text="Hello")]

        result = subtitles_to_events(subtitle)

        assert len(result) == 1
        assert result[0].start == 1000
        assert result[0].end == 2000
        assert result[0].text == "Hello"
        assert isinstance(result[0], SubtitleEvent)

    def test_multiple_subtitle_events(self):
        """Test with multiple subtitle events"""
        subtitle = SSAFile()
        subtitle.events = [
            SSAEvent(start=1000, end=2000, text="First"),
            SSAEvent(start=3000, end=4000, text="Second"),
            SSAEvent(start=5000, end=6000, text="Third"),
        ]

        result = subtitles_to_events(subtitle)

        assert len(result) == 3
        assert result[0].text == "First"
        assert result[1].text == "Second"
        assert result[2].text == "Third"


class TestSubtitlesToDict:
    """Test subtitles_to_dict function"""

    def test_empty_subtitle_file(self):
        """Test with empty subtitle file"""
        subtitle = SSAFile()
        result = subtitles_to_dict(subtitle)
        assert result == []

    def test_single_subtitle_event(self):
        """Test with single subtitle event"""
        subtitle = SSAFile()
        subtitle.events = [SSAEvent(start=1000, end=2000, text="Hello")]

        result = subtitles_to_dict(subtitle)

        assert len(result) == 1
        assert result[0] == {"start": 1000, "end": 2000, "text": "Hello"}
        assert isinstance(result[0], dict)

    def test_multiple_subtitle_events(self):
        """Test with multiple subtitle events"""
        subtitle = SSAFile()
        subtitle.events = [
            SSAEvent(start=1000, end=2000, text="First"),
            SSAEvent(start=3000, end=4000, text="Second"),
        ]

        result = subtitles_to_dict(subtitle)

        assert len(result) == 2
        assert result[0] == {"start": 1000, "end": 2000, "text": "First"}
        assert result[1] == {"start": 3000, "end": 4000, "text": "Second"}


class TestEventsToSubtitles:
    """Test events_to_subtitles function"""

    def test_empty_events_list(self):
        """Test with empty events list"""
        result = events_to_subtitles([])
        assert isinstance(result, SSAFile)
        assert len(result.events) == 0

    def test_single_event(self):
        """Test with single subtitle event"""
        events = [SubtitleEvent(start=1000, end=2000, text="Hello")]

        result = events_to_subtitles(events)

        assert isinstance(result, SSAFile)
        assert len(result.events) == 1
        assert result.events[0].start == 1000
        assert result.events[0].end == 2000
        assert result.events[0].text == "Hello"
        assert isinstance(result.events[0], SSAEvent)

    def test_multiple_events(self):
        """Test with multiple subtitle events"""
        events = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
        ]

        result = events_to_subtitles(events)

        assert len(result.events) == 2
        assert result.events[0].text == "First"
        assert result.events[1].text == "Second"


class TestValidateSubtitles:
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

        with pytest.raises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        assert "Subtitle ends at 12000" in str(exc_info.value)
        assert "while audio segment ends at 10000.0" in str(exc_info.value)

    def test_subtitle_start_after_end(self):
        """Test when subtitle start time is after end time"""
        subtitles = [SubtitleEvent(start=2000, end=1000, text="Invalid")]  # start > end
        duration = 10.0

        with pytest.raises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        assert "starts at 2000" in str(exc_info.value)
        assert "but ends at 1000" in str(exc_info.value)

    def test_overlapping_subtitles(self):
        """Test when subtitles overlap"""
        subtitles = [
            SubtitleEvent(start=1000, end=3000, text="First"),
            SubtitleEvent(start=2000, end=4000, text="Second"),  # overlaps with first
        ]
        duration = 10.0

        with pytest.raises(SubtitleValidationException) as exc_info:
            validate_subtitles(subtitles, duration)

        assert "starts at 2000" in str(exc_info.value)
        assert "previous subtitle finishes at 3000" in str(exc_info.value)

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


class TestSaveToJson:
    """Test save_to_json function"""

    def test_save_empty_subtitles(self):
        """Test saving empty subtitles list"""
        subtitles = []

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            save_to_json(subtitles, temp_path)

            # Read back and verify
            with open(temp_path, "r") as f:
                content = json.load(f)

            assert content == []
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
            with open(temp_path, "r") as f:
                content = json.load(f)

            assert len(content) == 1
            assert content[0] == {"start": 1000, "end": 2000, "text": "Hello"}
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
            with open(temp_path, "r") as f:
                content = json.load(f)

            assert len(content) == 2
            assert content[0] == {"start": 1000, "end": 2000, "text": "First"}
            assert content[1] == {"start": 3000, "end": 4000, "text": "Second"}
        finally:
            os.unlink(temp_path)


class TestMergeSubtitleEvents:
    """Test merge_subtitle_events function"""

    def test_merge_empty_groups(self):
        """Test merging empty subtitle groups"""
        subtitle_groups = []
        segment_durations = []

        with pytest.raises(Exception):  # Pydantic ValidationError
            merge_subtitle_events(subtitle_groups, segment_durations)

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

        assert len(result) == 2
        assert result[0].start == 1000  # No time shift
        assert result[0].end == 2000
        assert result[1].start == 3000
        assert result[1].end == 4000

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

        assert len(result) == 4

        # First group should remain unchanged
        assert result[0].start == 1000
        assert result[0].end == 2000
        assert result[1].start == 3000
        assert result[1].end == 4000

        # Second group should be shifted by 5 seconds (5000ms)
        assert result[2].start == 6000  # 1000 + 5000
        assert result[2].end == 7000  # 2000 + 5000
        assert result[3].start == 8000  # 3000 + 5000
        assert result[3].end == 9000  # 4000 + 5000

    def test_merge_three_groups(self):
        """Test merging three subtitle groups"""
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Group 1")],
            [SubtitleEvent(start=1000, end=2000, text="Group 2")],
            [SubtitleEvent(start=1000, end=2000, text="Group 3")],
        ]
        segment_durations = [3.0 * 1000, 4.0 * 1000, 5.0 * 1000]

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        assert len(result) == 3
        assert result[0].start == 1000  # No shift
        assert result[1].start == 4000  # Shifted by 3000ms
        assert result[2].start == 8000  # Shifted by 3000 + 4000 = 7000ms

    def test_merge_with_validation_error(self):
        """Test that merge raises validation error for invalid result"""
        # Create subtitles that will exceed total duration after merging
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=15000, text="Too long")]  # 15 seconds
        ]
        segment_durations = [10.0]  # Only 10 seconds total

        with pytest.raises(SubtitleValidationException):
            merge_subtitle_events(subtitle_groups, segment_durations)

    def test_merge_preserves_text_content(self):
        """Test that merging preserves all text content"""
        subtitle_groups = [
            [SubtitleEvent(start=1000, end=2000, text="Hello")],
            [SubtitleEvent(start=1000, end=2000, text="World")],
        ]
        segment_durations = [3.0 * 1000, 3.0 * 1000]

        result = merge_subtitle_events(subtitle_groups, segment_durations)

        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"


class TestSubtitleValidationException:
    """Test SubtitleValidationException"""

    def test_exception_creation(self):
        """Test creating SubtitleValidationException"""
        msg = "Test error message"
        exc = SubtitleValidationException(msg)
        assert str(exc) == msg
        assert isinstance(exc, Exception)


# Integration tests
class TestIntegration:
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
        assert len(converted_back.events) == len(original.events)
        for orig, conv in zip(original.events, converted_back.events):
            assert orig.start == conv.start
            assert orig.end == conv.end
            assert orig.text == conv.text

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
        assert len(merged) == 2
        assert merged[1].start == 4000  # Shifted by 3000ms


if __name__ == "__main__":
    pytest.main([__file__])
