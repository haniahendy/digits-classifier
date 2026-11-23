from digits.train import evaluate_all, worst_confusions

print(evaluate_all())
print("most confused (count, true, predicted):", worst_confusions())
