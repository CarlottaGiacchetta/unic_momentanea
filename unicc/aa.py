import itertools
import string

def is_flowing(s):
    return all(s[i] >= s[i - 1] for i in range(1, len(s)))

def is_receding(s):
    return all(s[i] <= s[i - 1] for i in range(1, len(s)))

def is_turbulent(s):
    return not is_flowing(s) and not is_receding(s)

unique_turbulent = set()

for i in range(1, 6):  # lunghezze da 1 a 5 inclusi
    all_combinations = itertools.product(string.ascii_lowercase, repeat=i)
    for combo in all_combinations:
        if is_turbulent(combo):
            unique_turbulent.add(combo)

print(f"Numero di stringhe UNICHE turbulent (lunghezza 1–5): {len(unique_turbulent)}")
