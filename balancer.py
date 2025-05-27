import pysubs2
import argparse
import re
from typing import List


class SRTBalancer:
    def __init__(self, max_words_per_screen=8, min_duration_ms=1000):
        """
        Initialize the SRT balancer.

        Args:
            max_words_per_screen (int): Maximum words to display per subtitle
            min_duration_ms (int): Minimum duration for each subtitle in milliseconds
        """
        self.max_words_per_screen = max_words_per_screen
        self.min_duration_ms = min_duration_ms

    def count_words(self, text: str) -> int:
        """Count words in subtitle text, ignoring HTML tags."""
        # Remove HTML tags and formatting
        clean_text = re.sub(r"<[^>]+>", "", text)
        # Split by whitespace and filter empty strings
        words = [word for word in clean_text.split() if word.strip()]
        return len(words)

    def split_text_smartly(self, text: str, max_words: int) -> List[str]:
        """Split text into chunks with smart line breaking."""
        lines = text.split("\\N")  # pysubs2 uses \\N for line breaks
        chunks = []
        current_chunk = []
        current_word_count = 0

        for line in lines:
            line_words = len([w for w in line.split() if w.strip()])

            # If adding this line would exceed the limit, start a new chunk
            if current_word_count + line_words > max_words and current_chunk:
                chunks.append("\\N".join(current_chunk))
                current_chunk = [line]
                current_word_count = line_words
            else:
                current_chunk.append(line)
                current_word_count += line_words

        # Add the last chunk
        if current_chunk:
            chunks.append("\\N".join(current_chunk))

        # If chunks are still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if self.count_words(chunk) <= max_words:
                final_chunks.append(chunk)
            else:
                # Split by words
                words = chunk.replace("\\N", " ").split()
                temp_chunk = []

                for word in words:
                    if len(temp_chunk) + 1 > max_words and temp_chunk:
                        final_chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                    else:
                        temp_chunk.append(word)

                if temp_chunk:
                    final_chunks.append(" ".join(temp_chunk))

        return final_chunks

    def balance_subtitles(self, subs: pysubs2.SSAFile) -> pysubs2.SSAFile:
        """Balance subtitles by splitting long entries."""
        new_subs = pysubs2.SSAFile()

        for line in subs:
            word_count = self.count_words(line.text)

            if word_count <= self.max_words_per_screen:
                # Keep as is
                new_subs.append(line)
            else:
                # Split the subtitle
                text_chunks = self.split_text_smartly(
                    line.text, self.max_words_per_screen
                )

                if len(text_chunks) == 1:
                    # Couldn't split effectively, keep as is
                    new_subs.append(line)
                else:
                    # Calculate timing for each chunk
                    total_duration = line.end - line.start
                    chunk_word_counts = [
                        self.count_words(chunk) for chunk in text_chunks
                    ]
                    total_words = sum(chunk_word_counts)

                    current_start = line.start

                    for i, (chunk_text, chunk_words) in enumerate(
                        zip(text_chunks, chunk_word_counts)
                    ):
                        # Calculate duration proportional to word count
                        if i == len(text_chunks) - 1:
                            # Last chunk gets remaining time
                            chunk_end = line.end
                        else:
                            duration_ratio = chunk_words / total_words
                            chunk_duration = max(
                                int(total_duration * duration_ratio),
                                self.min_duration_ms,
                            )
                            chunk_end = current_start + chunk_duration

                        # Ensure minimum duration
                        if chunk_end - current_start < self.min_duration_ms:
                            chunk_end = current_start + self.min_duration_ms

                        # Create new subtitle line
                        new_line = pysubs2.SSAEvent(
                            start=current_start, end=chunk_end, text=chunk_text
                        )
                        new_subs.append(new_line)

                        current_start = chunk_end

        return new_subs

    def process_file(self, input_path: str, output_path: str = None) -> None:
        """Process an SRT file and save balanced subtitles."""
        # Load subtitles
        subs = pysubs2.load(input_path)

        # Balance them
        balanced_subs = self.balance_subtitles(subs)

        # Save to output file
        if not output_path:
            base_name = input_path.rsplit(".", 1)[0]
            output_path = f"{base_name}_balanced.srt"

        balanced_subs.save(output_path)
        print(f"Balanced subtitles saved to: {output_path}")

    def analyze_file(self, input_path: str) -> None:
        """Analyze an SRT file and show statistics."""
        subs = pysubs2.load(input_path)

        total_entries = len(subs)
        long_entries = []
        word_counts = []

        for i, line in enumerate(subs):
            word_count = self.count_words(line.text)
            word_counts.append(word_count)
            if word_count > self.max_words_per_screen:
                preview = line.text.replace("\\N", " ")[:50] + "..."
                long_entries.append((i + 1, word_count, preview))

        if not word_counts:
            print("No subtitles found in file.")
            return

        avg_words = sum(word_counts) / len(word_counts)
        max_words = max(word_counts)

        print(f"\n📊 Subtitle Analysis")
        print(f"━" * 50)
        print(f"Total subtitles: {total_entries}")
        print(f"Average words per subtitle: {avg_words:.1f}")
        print(f"Maximum words in a subtitle: {max_words}")
        print(
            f"Subtitles exceeding {self.max_words_per_screen} words: {len(long_entries)}"
        )

        if long_entries:
            print(f"\n🔍 Long subtitles that will be split:")
            for idx, words, preview in long_entries[:10]:  # Show first 10
                print(f'  #{idx}: {words} words - "{preview}"')
            if len(long_entries) > 10:
                print(f"  ... and {len(long_entries) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Balance SRT subtitles by splitting long entries"
    )
    parser.add_argument("input", help="Input SRT file path")
    parser.add_argument(
        "-o", "--output", help="Output SRT file path (default: input_balanced.srt)"
    )
    parser.add_argument(
        "-w",
        "--max-words",
        type=int,
        default=8,
        help="Maximum words per subtitle (default: 8)",
    )
    parser.add_argument(
        "-d",
        "--min-duration",
        type=int,
        default=1000,
        help="Minimum duration per subtitle in ms (default: 1000)",
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        help="Analyze the file without processing",
    )

    args = parser.parse_args()

    balancer = SRTBalancer(
        max_words_per_screen=args.max_words, min_duration_ms=args.min_duration
    )

    try:
        if args.analyze:
            balancer.analyze_file(args.input)
        else:
            print(f"Processing: {args.input}")
            print(f"Max words per subtitle: {args.max_words}")
            print(f"Min duration per subtitle: {args.min_duration}ms")

            # Analyze first
            balancer.analyze_file(args.input)

            # Process
            balancer.process_file(args.input, args.output)
            print(f"✅ Processing complete!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
