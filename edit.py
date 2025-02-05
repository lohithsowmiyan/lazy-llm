import pandas as pd

# Load the CSV file
file_path = 'data/tao_og/exastencils.csv'
df = pd.read_csv(file_path, delimiter = ';')

df = df.drop(columns = ['root'])



# Save back to the same location
df.to_csv(file_path, index=False)
print(f"File saved successfully with swapped columns.")
