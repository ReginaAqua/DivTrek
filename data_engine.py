import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import warnings
warnings.filterwarnings('ignore')

class DestinationDatabase:
    def __init__(self, data_path="data/destinations.csv", embedding_model='all-MiniLM-L6-v2'):
        self.df = pd.read_csv(data_path, encoding='latin-1')
        
        self.seasonality_df = None
        if os.path.exists("data/real_seasonality.csv"):
            self.seasonality_df = pd.read_csv("data/real_seasonality.csv").set_index('Month')

        self.feature_matrix = None
        self.similarity_matrix = None
        
        print(f"Loading NLP Model '{embedding_model}'...")
        self.embedder = SentenceTransformer(embedding_model)
        self.text_embeddings = None
        
        self._process_data()

    def _clean_tourists(self, val):
        if pd.isna(val): return 0.0
        val_str = str(val).lower()
        num_str = re.sub(r'[^0-9.]', '', val_str)
        try:
            num = float(num_str) if num_str else 0.0
            if 'million' in val_str or 'm' in val_str: num *= 1000000
            elif 'k' in val_str: num *= 1000
            return num
        except ValueError:
            return 0.0

    def _map_price(self, cost_string):
        cost_mapping = {'Free': 20.0, 'Low': 45.0, 'Medium': 85.0, 'Medium-high': 130.0, 'High': 200.0, 'Extremely high': 350.0, 'Varies': 100.0}
        return cost_mapping.get(str(cost_string).strip(), 100.0)

    def _process_data(self):
        self.df.fillna({'Region': 'Europe', 'Category': 'Unknown', 'Cost of Living': 'Medium', 'Description': ''}, inplace=True)
        self.df['Clean_Traffic'] = self.df['Approximate Annual Tourists'].apply(self._clean_tourists)
        self.df['Daily_Budget_USD'] = self.df['Cost of Living'].apply(self._map_price)
        
        if 'Area_sq_km' not in self.df.columns:
            self.df['Area_sq_km'] = 100.0 
        else:
            self.df['Area_sq_km'] = pd.to_numeric(self.df['Area_sq_km'], errors='coerce').fillna(100.0)
        
        self.df['Combined_Text'] = "A " + self.df['Category'] + " destination in " + self.df['Region'] + ". " + self.df['Description']
        
        print("Generating dense NLP embeddings...")
        self.text_embeddings = self.embedder.encode(self.df['Combined_Text'].tolist(), show_progress_bar=False)
        self.similarity_matrix = cosine_similarity(self.text_embeddings)

    def get_data(self):
        return self.df, self.similarity_matrix

    def get_seasonal_multiplier(self, month, region):
        if self.seasonality_df is None or not month: return 1.0
        try:
            if region in self.seasonality_df.columns: return self.seasonality_df.at[month, region]
            else: return self.seasonality_df.at[month, 'Europe']
        except Exception: return 1.0

    def get_critique_relevance(self, destination_name):
        try:
            idx = self.df[self.df['Destination'] == destination_name].index[0]
            target_vec = self.text_embeddings[idx].reshape(1, -1)
            return cosine_similarity(target_vec, self.text_embeddings).flatten()
        except IndexError:
            return np.full(len(self.df), 0.5)

    def get_semantic_relevance(self, tags, persona="Standard"):
        if not tags or tags == ['']: return np.full(len(self.df), 0.5)
        
        query = " ".join(tags).lower()
        
        # EXTENDED PERSONA LOGIC
        if persona == "Backpacker": query += " cheap budget hostel adventure young"
        elif persona == "Luxury": query += " luxury expensive resort premium spa"
        elif persona == "Family": query += " family kids safe relaxing calm"
        elif persona == "Digital Nomad": query += " wifi cafe coworking city long-term"
        elif persona == "Eco-Tourist": query += " sustainable nature green eco-friendly wildlife"
        elif persona == "Cultural Explorer": query += " history museum traditional local art"

        price_boost = np.zeros(len(self.df))
        for i, row in self.df.iterrows():
            price = row['Daily_Budget_USD']
            if persona == "Backpacker" and price <= 60.0: price_boost[i] += 0.2
            if persona == "Digital Nomad" and price <= 80.0: price_boost[i] += 0.1
            if persona == "Luxury" and price >= 180.0: price_boost[i] += 0.2

        query_vec = self.embedder.encode([query])
        semantic_scores = cosine_similarity(query_vec, self.text_embeddings).flatten()
        final_scores = semantic_scores + price_boost
        
        if final_scores.max() > final_scores.min():
            final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
        else:
            final_scores = np.full(len(self.df), 0.5)
            
        return final_scores + np.random.uniform(0.0, 0.0001, len(final_scores))