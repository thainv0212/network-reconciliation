with open('result.txt', 'r') as f:
    txt_data = f.read()

line_splitted_data = txt_data.split('\n')
arr = []
for line in line_splitted_data:
    splits = line.split('\t')
    arr.append([int(splits[0]), int(splits[1])])

print(len([a for a in arr if a[0] == a[1]]))
print(len(arr))