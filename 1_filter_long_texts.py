import os
from transformers import AutoTokenizer
import csv
import sys

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

# Create output directory if it doesn't exist
os.makedirs("en_core_filtered", exist_ok=True)


def clean_registers(register_string):
    """Keep only uppercase register labels."""
    registers = register_string.split()
    uppercase_registers = [reg for reg in registers if reg.isupper()]
    return " ".join(uppercase_registers)


def process_file(input_path, output_path):
    """Process a single TSV file."""
    # Read input file and write filtered content
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            if len(row) < 2:  # Skip malformed rows
                continue

            registers = row[0]
            text = row[1]

            # Calculate token length
            length = len(tokenizer.encode(text))

            # Only keep if length <= 8192
            if length <= 8192:
                # Clean registers and write row
                cleaned_registers = clean_registers(registers)
                writer.writerow([cleaned_registers] + row[1:])


# Process all files
files = ["train.tsv", "dev.tsv", "test.tsv"]

for filename in files:
    input_path = os.path.join("en_core", filename)
    output_path = os.path.join("en_core_filtered", filename)
    print(f"Processing {filename}...")
    process_file(input_path, output_path)
    print(f"Completed {filename}")

print("All files processed. Filtered data saved in en_core_filtered/")
