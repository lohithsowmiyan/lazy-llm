import pandas as pd

# Load the CSV file
file_path = 'data/hpo/healthCloseIsses12mths0011-easy.csv'
df = pd.read_csv(file_path)

# Swap the first three columns
Y = df[['PRED40+','MRE-','ACC+']]
X = df.drop(columns = ['PRED40+','MRE-','ACC+'])

X['PRED40+'] = Y['PRED40+']
X['MRE-'] = Y['MRE-']
X['ACC+'] = Y['ACC+']

# Save back to the same location
X.to_csv(file_path, index=False)
print(f"File saved successfully with swapped columns.")
