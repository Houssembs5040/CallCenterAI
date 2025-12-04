"""Data loading and preprocessing"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """Load and preprocess ticket data"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()

    def load_data(self, test_size: float = 0.2, random_state: int = 42):
        """Load and split data"""
        print(f"📊 Loading data from {self.data_path}...")

        # Load CSV
        df = pd.read_csv(self.data_path)

        # Show available columns
        print(f"📋 Available columns: {df.columns.tolist()}")

        # Detect text and label columns automatically
        text_col = None
        label_col = None

        # Common column names for text
        text_candidates = [
            "ticket_text",
            "Document",
            "text",
            "description",
            "ticket",
            "message",
            "content",
            "Text",
            "Description",
        ]
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break

        # Common column names for labels
        label_candidates = [
            "category",
            "Topic_group",
            "label",
            "type",
            "topic",
            "Category",
            "Label",
            "Type",
        ]
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break

        # If not found automatically, use first text column and last column as label
        if text_col is None:
            text_col = df.columns[0]
            print(f"⚠️ Using first column as text: '{text_col}'")

        if label_col is None:
            label_col = df.columns[-1]
            print(f"⚠️ Using last column as label: '{label_col}'")

        print(f"✅ Text column: '{text_col}'")
        print(f"✅ Label column: '{label_col}'")

        # Clean data
        df = df.dropna(subset=[text_col, label_col])
        df[text_col] = df[text_col].astype(str)

        # Rename columns to standard names
        df = df.rename(columns={text_col: "Document", label_col: "Topic_group"})

        print(f"✅ Loaded {len(df)} tickets")
        print(f"📂 Categories: {df['Topic_group'].unique()}")

        # Encode labels
        df["label"] = self.label_encoder.fit_transform(df["Topic_group"])

        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["label"]
        )

        print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")

        return train_df, test_df, self.label_encoder

    def save_label_encoder(self, save_path: str):
        """Save label encoder to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        print(f"💾 Label encoder saved to {save_path}")

    @staticmethod
    def load_label_encoder(load_path: str):
        """Load label encoder from disk"""
        with open(load_path, "rb") as f:
            return pickle.load(f)
