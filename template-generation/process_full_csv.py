import csv

full_csv_path = './full_csv.csv'
data_path = './data.csv'
processed_csv = [('text', 'headlines')]

with open(full_csv_path, 'r+') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        text = f'{row[0]} # {row[1]} # {row[2]} # {row[3]}'
        headlines = row[4]
        processed_csv.append((text, headlines))

with open(data_path, 'w+') as f:
    writer = csv.writer(f)
    for example in processed_csv:
        writer.writerow(example)
