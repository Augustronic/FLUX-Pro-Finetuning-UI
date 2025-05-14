#!/usr/bin/env python3
"""
Project Whitespace Fixer

This script scans through a project directory and fixes common whitespace 
issues:
1. Removes whitespace from blank lines
2. Ensures files end with a newline character
3. Shows statistics about fixes applied

Usage:
  python fix_all_whitespace.py [directory]

If no directory is specified, it will scan the current directory.
"""

import os
import sys
from pathlib import Path


def should_process_file(filepath):
    """
    Determine if a file should be processed based on its extension.

    Args:
        filepath: Path to the file

    Returns:
        bool: True if the file should be processed
    """
    extensions = {
        '.py', '.md', '.txt', '.json', '.js', 
        '.html', '.css', '.yml', '.yaml'
    }
    return (filepath.suffix.lower() in extensions and 
            not filepath.name.startswith('.'))


def fix_file(filepath, dry_run=False):
    """
    Fix whitespace issues in a file.

    Args:
        filepath: Path to the file to fix
        dry_run: If True, only report issues without fixing

    Returns:
        tuple: (blank_line_fixes, end_newline_fixes) - count of fixes applied
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0

    # Track fixes
    blank_line_fixes = 0
    end_newline_fix = 0

    # Fix trailing whitespace on blank lines
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if line.strip() == '':
            if line != '\n':
                blank_line_fixes += 1
                fixed_lines.append('\n')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    # Ensure the file ends with a newline
    if fixed_lines and not fixed_lines[-1].endswith('\n'):
        fixed_lines[-1] += '\n'
        end_newline_fix = 1

    # Write back if changes were made and not in dry run mode
    if (blank_line_fixes > 0 or end_newline_fix > 0) and not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.writelines(fixed_lines)
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
            return 0, 0

    return blank_line_fixes, end_newline_fix


def scan_directory(directory, dry_run=False):
    """
    Scan a directory recursively for files to fix.

    Args:
        directory: Directory path to scan
        dry_run: If True, only report issues without fixing
    """
    total_files = 0
    files_with_issues = 0
    total_blank_line_fixes = 0
    total_end_newline_fixes = 0

    action = "Scanning" if dry_run else "Fixing"
    print(f"{action} whitespace issues in {directory}...")

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = Path(root) / filename

            if should_process_file(filepath):
                total_files += 1
                blank_fixes, end_fixes = fix_file(filepath, dry_run)

                if blank_fixes > 0 or end_fixes > 0:
                    files_with_issues += 1
                    action = "Would fix" if dry_run else "Fixed"
                    print(f"{action} in {filepath}:")
                    if blank_fixes > 0:
                        print(f"  - {blank_fixes} blank line(s) with "
                              f"whitespace")
                    if end_fixes > 0:
                        print("  - Added missing final newline")

                total_blank_line_fixes += blank_fixes
                total_end_newline_fixes += end_fixes

    # Print summary
    print("\nSummary:")
    print(f"Scanned {total_files} files")
    print(f"Found {files_with_issues} files with whitespace issues")
    print(f"  - {total_blank_line_fixes} blank lines with whitespace")
    print(f"  - {total_end_newline_fixes} files missing final newline")

    has_issues = (total_blank_line_fixes > 0 or total_end_newline_fixes > 0)

    if not dry_run and has_issues:
        print("\nAll whitespace issues have been fixed.")
    elif dry_run and has_issues:
        print("\nRun without --dry-run to fix these issues.")
    else:
        print("\nNo whitespace issues found.")


def main():
    """Main function to process command line arguments."""
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")

    directory = "."  # Default to current directory
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        directory = sys.argv[1]

    scan_directory(directory, dry_run)


if __name__ == "__main__":
    main()
