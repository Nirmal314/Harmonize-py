import json

with open('json/songs_with_categories.json', 'r') as file:
    original = json.load(file)

# ! 68.75%
# with open('json/songs_with_predicted_categories.json', 'r') as file:
#     predicted = json.load(file)
    
with open('json/final.json', 'r') as file:
    predicted = json.load(file)

count = 0

for i in range(len(original)):
    
    print(original[i]['name'])
    print(f"original: {original[i]['category']}")
    print(f"predicted: {predicted[i]['category']}")
    print()

    if original[i]['category'] != predicted[i]['category']:
        count = count + 1

print(f"{100 - count/len(original) * 100}%")