#!/usr/bin/env python3
# filepath: /home/foresti/mdlm/clean_and_save_aligned.py
import argparse
import json
import os
import re

def extract_model_id(path):
    """Extract model ID (like M7) from a path"""
    # Use regex to find MX pattern in the path
    match = re.search(r'/M(\d+)/', path)
    if match:
        return f"M{match.group(1)}"
    return None


def main():
    parser = argparse.ArgumentParser(description='Clean and standardize summary file schema')
    parser.add_argument('--input_file', type=str, required=True, help='File with summaries')
    parser.add_argument('--output_file', type=str, help='Optional output file path')

    args = parser.parse_args()
    
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"{base_name}_cleaned.jsonl"
    
    print(f"Reading from {args.input_file}")
    print(f"Writing to {args.output_file}")
    
    # Process the file line by line for JSONL files
    processed_items = 0
    fixed_items = 0
    
    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the JSON object
                entry = json.loads(line.strip())
                processed_items += 1
                
                # Check if "decoded" field is missing
                if "decoded" not in entry:
                    # Look for fields that might contain "decoded" in their name
                    decoded_field = None
                    for key in entry.keys():
                        if "decoded" in key.lower():
                            decoded_field = key
                            break
                    
                    # If we found a field with "decoded" in the name, rename it
                    if decoded_field:
                        entry["decoded"] = entry[decoded_field]
                        del entry[decoded_field]
                        fixed_items += 1
                
                # Write the cleaned entry
                outfile.write(json.dumps(entry) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON at line {line_num}: {e}")
                # Write the original line if parsing fails
                outfile.write(line)
    
    print(f"Processed {processed_items} items, fixed schema for {fixed_items} items")
    print(f"Cleaned data saved to {args.output_file}")

if __name__ == '__main__':
    main()