import json
import math

# Load the JSON file
with open("reddit_meme_data.json", "r") as f:
    data = json.load(f)

# Extract upvotes from the nested structure
entries = data["_default"]
ups_list = [entry["ups"] for entry in entries.values()]

# Define bucket function
def get_bucket(upvotes):
    if upvotes <= 2000:
        return "0-2000"
    elif upvotes <= 20000:
        return "2001-20000"
    else:
        return "20001-999999"

# Count occurrences
bucket_counts = {}

for ups in ups_list:
    bucket = get_bucket(ups)
    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

# Sort buckets by numeric lower-bound for printing
def bucket_sort_key(b):
    low = int(b.split("-")[0])
    return low

sorted_buckets = sorted(bucket_counts.items(), key=lambda x: bucket_sort_key(x[0]))

# Print result
print("Upvote Distribution:\n")
for bucket, count in sorted_buckets:
    print(f"{bucket}: {count}")
