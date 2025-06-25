input_file = "data/expert_scores_uproszczone.csv"
output_file = "data/expert_scores_uproszczone_po_konwersji.csv"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(",", ".")
content = content.replace(";", ",")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Zamiana zakończona. Wynik zapisano w", output_file)