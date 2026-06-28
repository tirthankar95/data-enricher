from dataloader.dataloader_factory import DataLoader, register
from pathlib import Path
import pandas as pd
import time

@register("grid-puzzle")
class PuzzleDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.csv_path = Path(__file__).parent.parent / 'data' / 'grid-puzzles.csv'
        self.df = pd.read_csv(self.csv_path)
    
    def load_data(self) -> pd.DataFrame:
        df_shuffle = self.df.sample(frac = 1, random_state = int(time.time()))
        self.grid = df_shuffle.iloc[0]['Grid']
        self.difficulty = df_shuffle.iloc[0]['Difficulty']
        return df_shuffle.iloc[0]['Puzzle']
    
    def impute(self, new_puzzle: str) -> pd.DataFrame:
        new_row = pd.DataFrame({
            "Grid": [self.grid],
            "Difficulty": [self.difficulty],
            "Story": [""],
            "Clues": [""],
            "Table": [""],
            "Options": [""],
            "Puzzle": [new_puzzle]
        })
        new_df = pd.concat([self.df, new_row], ignore_index=True)
        new_df.to_csv(self.csv_path, index=False)