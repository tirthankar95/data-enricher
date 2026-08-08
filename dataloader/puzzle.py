from dataloader.dataloader_factory import DataLoader, register
from pathlib import Path
import pandas as pd
import re
@register("grid-puzzle")
class PuzzleDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.csv_path = Path(__file__).parent.parent / 'data' / 'grid-puzzles.csv'
        self.df = pd.read_csv(self.csv_path)
    
    def iterate(self):
        limit = self.df.shape[0]
        for _, row in self.df.iterrows():
            '''
            Skip rows where the 'Story' column is empty; 
            because these are synthetically generated via LLMs
            '''
            if _ >= limit:
                break
            if row['Story'] == "": 
                continue
            yield {
                "metadata": {
                    "Grid": row['Grid'],
                    "Difficulty": row['Difficulty'],
                    "Story": row['Story'],
                    "Clues": row['Clues'],
                    "Table": row['Table'],
                    "Options": row['Options']
                },
                "data": row['Puzzle']
            }
    
    def impute(self, metadata: dict, new_puzzle: str) -> pd.DataFrame:
        puzzle = 'Story:\n' + new_puzzle
        story_match = re.search(
            r"Story:\s*(.*?)\s*Clues:",
            puzzle,
            flags=re.DOTALL
        )
        clue_match = re.search(
            r"Clues:\s*(.*?)\s*Solve the grid puzzle",
            puzzle,
            flags=re.DOTALL
        )
        new_row = pd.DataFrame({
            "Grid": [metadata['Grid']],
            "Difficulty": [metadata['Difficulty']],
            "Story": [story_match.group(1).strip() if story_match else None],
            "Clues": [clue_match.group(1).strip() if clue_match else None],
            "Table": [""],
            "Options": [""],
            "Puzzle": [new_puzzle]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.csv_path, index=False)
        return self.df