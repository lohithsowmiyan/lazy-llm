import pandas as pd

# Load the CSV file
file_path = 'data/misc/auto93.csv'
df = pd.read_csv(file_path)

df = df.drop(columns = ['HpX'])



# Save back to the same location
df.to_csv(file_path, index=False)
print(f"File saved successfully with swapped columns.")
