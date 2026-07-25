from dataloader.dataloader_factory import DataLoader, register
from pathlib import Path
import pandas as pd

@register("grid-puzzle")
class PuzzleDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.csv_path = Path(__file__).parent.parent / 'data' / 'grid-puzzles.csv'
        self.df = pd.read_csv(self.csv_path)
    
    def iterate(self):
        for _, row in self.df.iterrows():
            '''
            Skip rows where the 'Story' column is empty; 
            because these are synthetically generated via LLMs
            '''
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
        new_row = pd.DataFrame({
            "Grid": [metadata['Grid']],
            "Difficulty": [metadata['Difficulty']],
            "Story": [""],
            "Clues": [""],
            "Table": [""],
            "Options": [""],
            "Puzzle": [new_puzzle]
        })
        new_df = pd.concat([self.df, new_row], ignore_index=True)
        new_df.to_csv(self.csv_path, index=False)