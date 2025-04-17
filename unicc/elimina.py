import pandas as pd

# Legge un file CSV (non Excel)
df = pd.read_csv("bigearthnet-train.csv")


# Mantieni solo righe con indice pari (0, 2, 4, ...)
df_senza_righe_dispari = df.iloc[::2]
print(df_senza_righe_dispari.size)

# Salva in un nuovo CSV (o sovrascrivi quello esistente)
df_senza_righe_dispari.to_csv("bigearthnet-train.csv", index=False)
