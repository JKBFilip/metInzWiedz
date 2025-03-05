import numpy as np

def load_data(filename):
    with open(filename, "r") as f:
        data = [list(map(int, line.split())) for line in f.readlines()]
    return np.array(data)



def find_first_order_rules(data):
    rules = []
    covered = set()
    num_objects, num_attributes = data.shape[0], data.shape[1] - 1

    for i in range(num_objects):
        if i in covered:
            continue

        for j in range(num_attributes):
            col_values = data[:, j]
            decision_values = data[:, -1]
            mask = (col_values == col_values[i])
            matching_decisions = np.unique(decision_values[mask])
            if len(matching_decisions) == 1:
                rule = f"o{i + 1} (a{j + 1}={col_values[i]}) => d={matching_decisions[0]}"
                rules.append(rule)
                covered.update(set(np.where(mask)[0]))
                break
        else:
            rules.append(f"o{i + 1} brak")

    return rules, covered


def find_second_order_rules(data, covered):
    rules = []
    num_objects, num_attributes = data.shape[0], data.shape[1] - 1  # Last column is decision

    for i in range(num_objects):
        if i in covered:
            continue

        for j in range(num_attributes - 1):
            for k in range(j + 1, num_attributes):
                mask = (data[:, j] == data[i, j]) & (data[:, k] == data[i, k])
                matching_decisions = np.unique(data[mask, -1])

                if len(matching_decisions) == 1:  # Unique decision for this attribute pair
                    rule = f"o{i + 1} (a{j + 1}={data[i, j]}) ^ (a{k + 1}={data[i, k]}) => d={matching_decisions[0]}"
                    rules.append(rule)
                    covered.add(i)
                    break
            if i in covered:
                break

    return rules


def main():
    filename = "values.txt"
    data = load_data(filename)
    first_order_rules, covered = find_first_order_rules(data)
    second_order_rules = find_second_order_rules(data, covered)

    print("Rząd I:")
    for rule in first_order_rules:
        print(rule)

    print("\nRząd II:")
    for rule in second_order_rules:
        print(rule)


if __name__ == "__main__":
    main()
