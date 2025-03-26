import itertools
import numpy as np
import pandas as pd


def load_decision_system(filename):
    df = pd.read_csv(filename, delimiter=" ")
    attributes = df.columns[:-1]
    decision = df.columns[-1]
    return df, attributes, decision


def compute_indiscernibility_matrix(df, attributes, decision):
    n = len(df)
    matrix = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(i + 1, n):
            differing_attrs = [attr for attr in attributes if df[attr].iloc[i] != df[attr].iloc[j]]
            if df[decision].iloc[i] != df[decision].iloc[j]:
                matrix[i, j] = differing_attrs
    return matrix


def extract_exhaustive_rules(matrix, df, attributes, decision):
    rules = []
    n = len(df)

    for i in range(n):
        possible_rules = []
        for j in range(n):
            if matrix[i, j] is not None:
                possible_rules.append(set(matrix[i, j]))

        min_rules = set.intersection(*possible_rules) if possible_rules else set()

        for attr in min_rules:
            rules.append(f"({attr} = {df[attr].iloc[i]}) => ({decision} = {df[decision].iloc[i]})")

    return rules


def main():
    filename = "dane_wejsciowe.txt"
    df, attributes, decision = load_decision_system(filename)

    indiscernibility_matrix = compute_indiscernibility_matrix(df, attributes, decision)
    rules = extract_exhaustive_rules(indiscernibility_matrix, df, attributes, decision)

    print("Wygenerowane reguły decyzyjne:")
    for rule in rules:
        print(rule)


if __name__ == "__main__":
    main()
