#!/usr/bin/env python3
"""
Check specific lines in a file for whitespace issues.
"""
import sys


def check_line(filepath, line_number):
    """
    Check a specific line in a file for whitespace issues.

    Args:
        filepath: Path to the file
        line_number: Line number to check (1-based)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if line_number <= len(lines):
        line = lines[line_number - 1]
        line_repr = repr(line)

        print(f"Line {line_number}: {line_repr}")

        if line.strip() == '':
            if line != '\n':
                print("  - Blank line contains whitespace")
                # Show ASCII codes for each character
                chars = []
                for i, c in enumerate(line):
                    chars.append(f"{ord(c):02x}")
                print(f"  - ASCII codes: {' '.join(chars)}")
            else:
                print("  - Clean blank line (only contains newline)")
        else:
            print("  - Not a blank line")
    else:
        print(f"Line {line_number} is beyond the end of file")


def main():
    """Main function to check specific lines."""
    # Check if file parameter was provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'utils/env_manager.py'

    print(f"Checking specific lines in {filepath}...")

    if filepath == 'fix_whitespace.py':
        # For fix_whitespace.py, check the lines that were previously fixed
        problem_lines = [17, 22, 29, 34, 47, 54, 79]
    else:
        # Lines mentioned in diagnostics for env_manager.py
        problem_lines = [27, 60, 64, 74, 78, 92, 97, 110, 115, 
                         129, 134, 150, 154, 175, 184]

    for line_num in problem_lines:
        check_line(filepath, line_num)
        print()


if __name__ == "__main__":
    main()
