#!/usr/bin/env python3
"""
Whitespace Diagnostics Tool

This script examines files for whitespace issues with detailed diagnostics,
showing exactly what characters are present in each line.
"""

import os
import sys


def diagnose_file(filepath):
    """
    Diagnose whitespace issues in a file with detailed output.

    Args:
        filepath: Path to the file to diagnose
    """
    print(f"Diagnosing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    issues_found = 0

    for i, line in enumerate(lines, 1):
        # Check blank lines that might contain whitespace
        if line.strip() == '' and line != '\n':
            issues_found += 1
            chars = []
            for c in line:
                if c == ' ':
                    chars.append('·')  # Middle dot for space
                elif c == '\t':
                    chars.append('→')  # Arrow for tab
                elif c == '\r':
                    chars.append('↵')  # Carriage return
                elif c == '\n':
                    chars.append('↓')  # Newline
                else:
                    chars.append(c)

            char_repr = ''.join(chars)
            print(f"Line {i}: Blank line contains whitespace: {char_repr}")

    # Check for missing final newline
    if lines and not lines[-1].endswith('\n'):
        issues_found += 1
        print(f"Line {len(lines)}: No newline at end of file")

    if issues_found == 0:
        print(f"No whitespace issues found in {filepath}")
    else:
        print(f"Found {issues_found} whitespace issues in {filepath}")


def main():
    """Main function to process command line arguments."""
    if len(sys.argv) > 1:
        # Process files specified on command line
        for filepath in sys.argv[1:]:
            if os.path.isfile(filepath):
                diagnose_file(filepath)
            else:
                print(f"Error: {filepath} is not a valid file")
    else:
        # Check the files mentioned in diagnostics
        diagnose_file('utils/env_manager.py')


if __name__ == "__main__":
    main()
