#!/usr/bin/env python3
"""
Whitespace Fixer

This script fixes common whitespace issues in Python files:
1. Removes whitespace (spaces, tabs) from blank lines
2. Ensures files end with a newline character
"""

import os
import sys


def fix_file(filepath):
    """
    Fix whitespace issues in a file.

    Args:
        filepath: Path to the file to fix
    """
    print(f"Processing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Fix whitespace issues
    fixed_lines = []
    changes_made = False
    issues_found = 0

    for i, line in enumerate(lines, 1):
        # Check for blank lines with whitespace
        if line.strip() == '':
            if line != '\n':
                print(f"Line {i}: Blank line contains whitespace")
                fixed_lines.append('\n')
                changes_made = True
                issues_found += 1
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    # Ensure the file ends with a newline
    if fixed_lines and not fixed_lines[-1].endswith('\n'):
        print("File does not end with newline")
        fixed_lines[-1] += '\n'
        changes_made = True
        issues_found += 1

    # Write changes if needed
    if changes_made:
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.writelines(fixed_lines)
            print(f"Fixed {issues_found} whitespace issues in {filepath}")
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
    else:
        print(f"No whitespace issues found in {filepath}")


def main():
    """Main function to process command line arguments."""
    if len(sys.argv) > 1:
        # Process files specified on command line
        for filepath in sys.argv[1:]:
            if os.path.isfile(filepath):
                fix_file(filepath)
            else:
                print(f"Error: {filepath} is not a valid file")
    else:
        # No arguments provided, fix some common files with issues
        fix_file('utils/env_manager.py')

        # Check for other files with issues mentioned in diagnostics
        print("\nChecking for other files with known issues...")
        if os.path.isfile('fix_whitespace.py'):
            fix_file('fix_whitespace.py')


if __name__ == "__main__":
    main()
