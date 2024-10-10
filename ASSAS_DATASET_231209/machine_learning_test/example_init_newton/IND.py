def extract_index(vector):
    index = {"Steam Partial-Pressure": [], "Hydrogen Partial-Pressure": [], "Void Fraction": [], "Void Fraction SWALLEN": [],
             "Gas Temperature": [], "Liquid Temperature": [], "Surface Temperature 1": [], "Surface Temperature 2": []}

    for i in range(0, 390, 5):
        index["Steam Partial-Pressure"].append(i)
        index["Hydrogen Partial-Pressure"].append(i+1)
        index["Void Fraction"].append(i+2)
        index["Gas Temperature"].append(i+3)
        index["Liquid Temperature"].append(i+4)

    for i in range(390, 402, 6):
        index["Steam Partial-Pressure"].append(i)
        index["Hydrogen Partial-Pressure"].append(i+1)
        index["Void Fraction"].append(i+2)
        index["Gas Temperature"].append(i+3)
        index["Liquid Temperature"].append(i+4)
        index["Void Fraction SWALLEN"].append(i+5)

    for i in range(402, 807, 5):
        index["Steam Partial-Pressure"].append(i)
        index["Hydrogen Partial-Pressure"].append(i+1)
        index["Void Fraction"].append(i+2)
        index["Gas Temperature"].append(i+3)
        index["Liquid Temperature"].append(i+4)

    for i in range(807, 837, 6):
        index["Steam Partial-Pressure"].append(i)
        index["Hydrogen Partial-Pressure"].append(i+1)
        index["Void Fraction"].append(i+2)
        index["Gas Temperature"].append(i+3)
        index["Liquid Temperature"].append(i+4)
        index["Void Fraction SWALLEN"].append(i+5)

    for i in range(837, 1177, 5):
        index["Steam Partial-Pressure"].append(i)
        index["Hydrogen Partial-Pressure"].append(i+1)
        index["Void Fraction"].append(i+2)
        index["Gas Temperature"].append(i+3)
        index["Liquid Temperature"].append(i+4)

    for i in range(1799, 2233, 2):
        index["Surface Temperature 1"].append(i)
        index["Surface Temperature 2"].append(i+1)
    return index

def outliers(vec, index):
    for i in index["Steam Partial-Pressure"]:
        if vec[i] < 0:
            vec[i] = 0.0
        elif vec[i] > 22100000:
            vec[i] = 22100000.0

    for i in index["Hydrogen Partial-Pressure"]:
        if vec[i] < 0:
            vec[i] = 0.0
        elif vec[i] > 22100000:
            vec[i] = 22100000.0

    for i in index["Void Fraction"]:
        if vec[i] < 0.00001:
            vec[i] = 0.00001
        elif vec[i] > 0.999999:
            vec[i] = 0.999999

    for i in index["Void Fraction SWALLEN"]:
        if vec[i] < 0.00001:
            vec[i] = 0.00001
        elif vec[i] > 0.999999:
            vec[i] = 0.999999

    for i in index["Gas Temperature"]:
        if vec[i] < 273.15:
            vec[i] = 273.15
        elif vec[i] > 5273:
            vec[i] = 5273

    for i in index["Surface Temperature 1"]:
        if vec[i] < 273.15:
            vec[i] = 273.15
        elif vec[i] > 5273:
            vec[i] = 5273

    for i in index["Surface Temperature 2"]:
        if vec[i] < 273.15:
            vec[i] = 273.15
        elif vec[i] > 5273:
            vec[i] = 5273

    for i in index["Liquid Temperature"]:
        if vec[i] < 273.15:
            vec[i] = 273.15
        elif vec[i] > 646.85:
            vec[i] = 646.85
    return vec
