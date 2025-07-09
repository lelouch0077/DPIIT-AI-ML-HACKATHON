import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
df = pd.read_csv(r"C:\Users\Raihan\OneDrive\Desktop\DPIIT_HACKATHON\image_labels.csv")

# Split into 80% train / 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to new files
train_df.to_csv(r"C:\Users\Raihan\OneDrive\Desktop\DPIIT_HACKATHON\train.csv", index=False, encoding='utf-8-sig')
val_df.to_csv(r"C:\Users\Raihan\OneDrive\Desktop\DPIIT_HACKATHON\val.csv", index=False, encoding='utf-8-sig')

print("✅ Split complete: train.csv and val.csv saved.")
