# оценка схожести двух строк
def calc_jaccard(str1:str, str2:str):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))