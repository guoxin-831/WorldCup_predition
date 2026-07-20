from task1.loader import WorldCupLoader

loader = WorldCupLoader()

df = loader.load()

print(df.head())