import json

data = None
with open('annotations.json', 'r') as js_file:
    data = json.load(js_file)

print(data)

with open('annotations2.json', 'w') as js_file:
    data = json.dumps(data, indent=4)
    js_file.write(data)
