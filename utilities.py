import os
import pandas as pd

class Create_User:
    def __init__(self, username, dataframe):
        self.user_name = username.replace(' ', '_')
        self.dataframe = dataframe
        self.create_user_folder()
    
    def create_user_folder(self):
        os.makedirs('User', exist_ok=True)
        os.makedirs(f'User/{self.user_name}', exist_ok=True)
        self.dataframe.to_csv(f'User/{self.user_name}/{self.user_name}.csv', index=False)

def check_model_training_status(user_name):
    user_dir = f"User/{user_name}"
    if os.path.exists(user_dir) and 'model.pth' in os.listdir(user_dir):
        return True
    else: 
        return False

class cache_memory:
    def __init__(self, user_name):
        self.user_name = user_name
        self.loaded_dataframe = None
    
    def create_cache(self):
        df = pd.DataFrame(columns=['Title', 'Upload_Date', 'Hype_Score', 'Min', 'Max'])
        user_dir = f"User/{self.user_name}"
        os.makedirs(user_dir, exist_ok=True)
        df.to_csv(f'{user_dir}/{self.user_name}_cache.csv', index=False)

    def load_cache(self):
        user_dir = f"User/{self.user_name}"
        cache_file = f'{user_dir}/{self.user_name}_cache.csv'
        self.loaded_dataframe = pd.read_csv(cache_file)

    def dump_data(self, movie_name, date, trend_score, min_view, max_view):
        user_dir = f"User/{self.user_name}"
        cache_file = f'{user_dir}/{self.user_name}_cache.csv'
        
        df = pd.DataFrame({
            'Title': [movie_name],
            'Upload_Date': [date],
            'Hype_Score': [trend_score],
            'Min': [min_view],
            'Max': [max_view]
        })

        # Append to existing cache or create new
        if os.path.exists(cache_file):
            existing_df = pd.read_csv(cache_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(cache_file, index=False)

    def check_for_cache(self):
        user_dir = f"User/{self.user_name}"
        cache_file = f'{user_dir}/{self.user_name}_cache.csv'
        
        if os.path.exists(cache_file):
            self.load_cache()
        else: 
            self.create_cache()
            self.load_cache()