import json
import os
import time
import shutil  # For getting terminal size
import textwrap  # For text wrapping

labels = ["LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]

evaluation_instructions = """Evaluation Options:

5 (Perfect): Correct/almost correct
4 (Good): Mostly correct
3 (Ok): Notable issues
2 (Poor): Major issues
1 (Wrong): Incorrect
"""


class SegmentEvaluator:
    def __init__(self, input_file, old_file, output_dir):
        self.input_file = input_file
        self.old_file = old_file
        self.output_dir = output_dir
        self.previous_evaluations = {}
        self.terminal_width = shutil.get_terminal_size().columns
        self.split_width = (self.terminal_width - 3) // 2  # -3 for the separator

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_segments(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def load_previous_evaluations(self, evaluator_id):
        prev_file = os.path.join(
            self.output_dir, f"evaluations_prev_{evaluator_id}.jsonl"
        )
        if not os.path.exists(prev_file):
            return []

        evaluations = []
        with open(prev_file, "r", encoding="utf-8") as f:
            for line in f:
                eval_data = json.loads(line)
                evaluations.append(
                    {
                        "label_score": eval_data["label_score"],
                        "segment_score": eval_data["segment_score"],
                    }
                )
        return evaluations

    def get_last_evaluated_position(self, evaluator_id):
        output_file = os.path.join(self.output_dir, f"evaluations_{evaluator_id}.jsonl")
        if not os.path.exists(output_file):
            return 0
        with open(output_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")
        time.sleep(0.2)

    def wrap_text(self, text, width):
        """Wrap text to specified width and return as list of lines"""
        return textwrap.wrap(text, width=width, replace_whitespace=False)

    def format_segment_text(self, text, width):
        """Format segment text to fit within specified width"""
        # First wrap the text
        lines = textwrap.wrap(text, width=width, replace_whitespace=False)
        # Join with explicit newlines
        return "\n".join(lines) if lines else ""

    def create_split_line(self, left_text, right_text, header=False):
        """Create a split line with left and right content"""
        # Split both texts into lines first
        left_parts = left_text.split("\n")
        right_parts = right_text.split("\n")

        # Then wrap each line to fit the width
        left_lines = []
        for part in left_parts:
            left_lines.extend(textwrap.wrap(part, width=self.split_width) or [""])

        right_lines = []
        for part in right_parts:
            right_lines.extend(textwrap.wrap(part, width=self.split_width) or [""])

        # Ensure both sides have same number of lines
        max_lines = max(len(left_lines), len(right_lines))
        left_lines.extend([""] * (max_lines - len(left_lines)))
        right_lines.extend([""] * (max_lines - len(right_lines)))

        separator = "║" if header else "│"

        # Create the split lines with proper padding
        result = []
        for left, right in zip(left_lines, right_lines):
            left_padded = left.ljust(self.split_width)
            right_padded = right.ljust(self.split_width)
            result.append(f"{left_padded} {separator} {right_padded}")

        return result

    def display_split_segments(
        self, current_segment, old_segment, current_pos, total_segments
    ):
        """Display current and old segments side by side"""
        self.clear_screen()

        # Header
        print("=" * self.terminal_width)
        header_lines = self.create_split_line(
            f"Current Segmentation ({current_pos}/{total_segments})",
            "Previous Segmentation",
            header=True,
        )
        print("\n".join(header_lines))

        # Display manual label for the text
        print("=" * self.terminal_width)
        label_line = self.create_split_line(
            f"Manual label: {current_segment['label']}",
            f"Manual label: {old_segment['label']}",
            header=True,
        )
        print("\n".join(label_line))
        print("=" * self.terminal_width)

        # Display segment pairs
        max_segments = max(
            len(current_segment["segments"]), len(old_segment["segments"])
        )

        for i in range(max_segments):
            current_seg = (
                current_segment["segments"][i]
                if i < len(current_segment["segments"])
                else None
            )
            old_seg = (
                old_segment["segments"][i] if i < len(old_segment["segments"]) else None
            )

            # Prepare the display strings for both sides
            if current_seg:
                # Just take the last probability array and check threshold
                current_probs = current_seg["probs"][-1]  # Get last probability array
                current_label = " ".join(
                    [labels[j] for j, prob in enumerate(current_probs) if prob > 0.7]
                )
                current_text = (
                    f"Segment {i+1} [{current_label}]:\n{current_seg['text']}"
                )
            else:
                current_text = f"[No more segments]"

            if old_seg:
                old_probs = old_seg["probs"][-1]  # Get last probability array
                old_label = " ".join(
                    [labels[j] for j, prob in enumerate(old_probs) if prob > 0.7]
                )
                old_text = f"Segment {i+1} [{old_label}]:\n{old_seg['text']}"
            else:
                old_text = f"[No more segments]"

            # Create and display the split lines
            split_lines = self.create_split_line(current_text, old_text)
            print("\n".join(split_lines))
            print("-" * self.terminal_width)

        print("\n" + evaluation_instructions)

    def get_single_evaluation(self, evaluation_type, position):
        prev_score = None
        if position < len(self.previous_evaluations):
            prev_score = self.previous_evaluations[position][f"{evaluation_type}_score"]
            prev_display = f" (previous: {prev_score})"
        else:
            prev_display = ""

        while True:
            prompt = f"Your evaluation for {evaluation_type.upper()}{prev_display} (1-5 or q): "
            choice = input(prompt).strip().lower()

            if choice == "q":
                return None
            try:
                score = int(choice)
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass

            print("\nInvalid input. Please enter a single digit (1-5), or q.")

    def get_sequential_evaluation(self, position):
        label_score = self.get_single_evaluation("label", position)
        if label_score is None:
            return None

        segment_score = self.get_single_evaluation("segment", position)
        if segment_score is None:
            return None

        self.clear_screen()
        return (label_score, segment_score)

    def save_evaluation(self, evaluator_id, evaluation):
        output_file = os.path.join(self.output_dir, f"evaluations_{evaluator_id}.jsonl")
        evaluation_record = {
            "label_score": evaluation[0],
            "segment_score": evaluation[1],
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(evaluation_record) + "\n")

    def run_evaluation(self):
        self.clear_screen()
        print("Welcome to the Segmentation Evaluator")
        evaluator_id = input("Name: ").strip()

        # Load all necessary data
        self.previous_evaluations = self.load_previous_evaluations(evaluator_id)
        current_segments = self.load_segments(self.input_file)
        old_segments = self.load_segments(self.old_file)

        start_position = self.get_last_evaluated_position(evaluator_id)

        if start_position >= len(current_segments):
            self.clear_screen()
            print("\nYou have completed all available segments!")
            return

        self.clear_screen()
        print(
            f"\nStarting from position {start_position + 1} of {len(current_segments)}"
        )
        input("Press Enter to continue...")

        try:
            for i, (current_segment, old_segment) in enumerate(
                zip(current_segments[start_position:], old_segments[start_position:]),
                start_position,
            ):
                self.display_split_segments(
                    current_segment, old_segment, i + 1, len(current_segments)
                )
                evaluation = self.get_sequential_evaluation(i)

                if evaluation is None:  # User quit
                    self.clear_screen()
                    break

                self.save_evaluation(evaluator_id, evaluation)

        except KeyboardInterrupt:
            self.clear_screen()
            print("\nEvaluation interrupted. Progress has been saved.")
            return

        self.clear_screen()
        print("\nThank you for completing the evaluation!")
        print(f"Your evaluations have been saved in: {self.output_dir}")


def main():
    evaluator = SegmentEvaluator(
        input_file="sample.jsonl",
        old_file="prev_sample.jsonl",
        output_dir=".",
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
