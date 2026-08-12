from pathlib import Path
import pandas as pd
import argparse
import json
import re



r'''
    Original: — matches the literal text Original:
    \s* — skips any whitespace (spaces, newlines) right after it
    (.*?) — a lazy/non-greedy capture group that grabs everything up to the next part of the pattern, capturing as little as possible
    \s* — skips trailing whitespace before the next marker
    Dummy: — matches the literal text Dummy:, which marks the end of the capture

    re.DOTALL makes . match newlines too, so the captured content (.*?) can span multiple lines.

    Purpose: extract the text that appears between Original: and the next Dummy: — i.e., the "original" value, trimmed of surrounding whitespace.
r'''
ORIGINAL_TO_DUMMY_PATTERN = re.compile(
    r"Original:\s*(.*?)\s*Dummy:",
    re.DOTALL,
)

r'''
    Dummy: — matches the literal text Dummy:
    \s* — skips whitespace after it
    (.*?) — lazy capture group for the "dummy" value, spanning multiple lines thanks to DOTALL
    \s* — trims trailing whitespace before the boundary
    (?=\nOriginal:|\Z) — a lookahead (zero-width, not consumed) that stops the capture right before either:
    \nOriginal: — the start of the next Original: block on a new line, or
    \Z — the absolute end of the string
r'''
DUMMY_TO_ORIGINAL_PATTERN = re.compile(
    r"Dummy:\s*(.*?)\s*(?=\nOriginal:|\Z)",
    re.DOTALL,
)


def extract_sections(text: str) -> tuple[list[str], list[str]]:
    original_to_dummy = [match.strip() for match in ORIGINAL_TO_DUMMY_PATTERN.findall(text)]
    dummy_to_original = [match.strip() for match in DUMMY_TO_ORIGINAL_PATTERN.findall(text)]
    return original_to_dummy, dummy_to_original


def extract_from_file(log_path: str | Path) -> tuple[list[str], list[str]]:
    content = Path(log_path).read_text(encoding="utf-8")
    return extract_sections(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 'Original -> Dummy' and 'Dummy -> Original' sections into two arrays.",
    )
    parser.add_argument(
        "--log",
        default="log.txt",
        help="Path to the log file (default: log.txt)",
    )
    args = parser.parse_args()
    original_to_dummy, dummy_to_original = extract_from_file(args.log)
    res = json.dumps(
    {
        "original_to_dummy": original_to_dummy,
        "dummy_to_original": dummy_to_original,
    }, ensure_ascii=False, indent=2)
    return res 


def extract_options(row: pd.Series) -> str:
    puzzle = row['Puzzle']
    mx_options = int(row['Grid'].split('x')[0]) - 1
    lines = puzzle.split('\n')
    new_lines = []
    for line in lines:
        if line != '':
            new_lines.append(line)
    lines = new_lines[::-1]
    return '\n'.join(lines[:mx_options])


def impute(res):
    df = pd.read_csv('data/grid-puzzles-ordered.csv')
    data = json.loads(res)
    new_rows = []
    sz = len(data['original_to_dummy'])
    for iter, row in df.iterrows():
        if iter >= sz:
            break
        new_rows.append({
            'Grid': row['Grid'],
            'Difficulty': row['Difficulty'],
            'Story': row['Story'],
            'Clues': row['Clues'],
            'Table': row['Table'],
            'Options': row['Options'],
            'Puzzle': row['Puzzle']
        })
        puzzle = 'Story:\n' + data['dummy_to_original'][iter]
        clue_match = re.search(
            r"Clues:\s*(.*?)\s*Solve the grid puzzle",
            puzzle,
            flags=re.DOTALL
        )
        story_match = re.search(
            r"Story:\s*(.*?)\s*Clues:",
            puzzle,
            flags=re.DOTALL
        )
        new_rows.append({
            'Grid': row['Grid'],
            'Difficulty': row['Difficulty'],
            'Story': story_match.group(1).strip() if story_match else None,
            'Clues': clue_match.group(1).strip() if clue_match else None,
            'Options': extract_options(row),
            'Puzzle': puzzle,
        })
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv('data/grid-puzzles-ordered-new.csv', index=False)
    
if __name__ == "__main__":
    res = main()
    impute(res)
    
