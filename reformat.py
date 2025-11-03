import os
import re
import sys

_ROMAN_PATTERN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_BRACKET_PATTERN = re.compile(r"\[.*?\]", re.DOTALL)
# New pattern to find years 1800, 1900, or 2000+
# \b ensures we match whole numbers (e.g., not '12024')
_YEAR_PATTERN = re.compile(r"\b((18|19)\d{2}|2\d{3})\b")


def _replace_newlines_with_token(text: str) -> str:
    """
    Replaces 3 or more consecutive newlines with '<|endoftext|>' and a single newline.
    """
    # Find 3 or more consecutive newlines and replace them
    return re.sub(r"\n{3,}", "<|endoftext|>\n", text)

def _remove_excessive_newlines(text: str) -> str:
    """
    Replaces 2 or more consecutive newlines with a single newline.
    """
    return re.sub(r"\n{2,}", "\n", text)

def _process_text(text: str) -> str:
    # Remove text within square brackets
    without_brackets = _BRACKET_PATTERN.sub("", text)
    
    # Filter out lines
    filtered = [
        line
        for line in without_brackets.splitlines(keepends=True)
        # Keep line if:
        # 1. It's a blank line (just whitespace)
        # OR
        # 2. It does NOT fully match the Roman numeral pattern
        #    AND it does NOT contain a year (1800+)
        if (line.strip() == "" or
            (not _ROMAN_PATTERN.fullmatch(line.strip()) and
             not _YEAR_PATTERN.search(line)))
    ]
    
    # Re-join the text
    compacted = "".join(filtered)
    
    # Use the new function to replace consecutive newlines with the token
    return _remove_excessive_newlines(_replace_newlines_with_token(compacted))


def main(path: str) -> None:
    if not os.path.isfile(path):
        print(f"Error: No such file: {path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
            
        processed_content = _process_text(content)
        
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(processed_content)
            
        print(f"Successfully processed file: {path}")

    except IOError as e:
        print(f"Error reading or writing to file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
        
    # Example usage:
    # Create a dummy test.txt if it doesn't exist for testing
    file_path = 'test.txt'
    if not os.path.exists("test.txt"):
        print("Creating dummy test.txt for demonstration.")
        # Updated dummy content to test all conditions
        dummy_content = (
            "This is the first line.\n"
            "\n"
            "This is the second.\n"
            "This line was published in 1995 and will be removed.\n"
            "This line mentions 1776 and will be kept.\n"
            "\n"
            "\n"
            "This is the third.\n"
            "[This text will be removed]\n"
            "IV\n"
            "This line is after the Roman numeral.\n"
            "This line is from 2024 and will be deleted.\n"
            "\n\n\n\n"
            "The end."
        )
        try:
            with open("test.txt", "w", encoding="utf-8") as f:
                f.write(dummy_content)
        except IOError as e:
            print(f"Failed to create test.txt: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"--- Processing {file_path} ---")
    # Print content *before* processing
    print("--- Content BEFORE ---")
    with open(file_path, "r", encoding="utf-8") as f:
        print(f.read())
    
    main(file_path)

    # Print content *after* processing
    print("\n--- Content AFTER ---")
    with open(file_path, "r", encoding="utf-8") as f:
        print(f.read())