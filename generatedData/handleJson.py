import json
from collections import defaultdict

# Mapping of field IDs to strings
field_map = {
    1: "Education",
    2: "Innovation",
    3: "Culture",
    4: "Events",
    5: "Competitions",
    6: "Coworking/Marker Spaces",
    7: "Platforms",
    8: "Media",
    9: "IP/TTOS & KTOS",
    10: "Science/Tecknology Parks",
    11: "R&D",
    12: "Policy",
    13: "Mentoring",
    14: "Training",
    15: "NGOs",
    16: "Special Abilities",
    17: "Women",
    18: "Incubators",
    19: "Accelerators",
    20: "Business Online",
    21: "Teach Advice",
    22: "Marketing",
    23: "Hiring",
    24: "Accounting",
    25: "Legal Information",
    26: "Preseed",
    27: "Seed",
    28: "Business Angles",
    29: "Private Equity",
    30: "Venture Capital",
    31: "Crowd Funding",
    32: "Grants",
    33: "Donors"
}

def merge_companies(input_path, output_path):
    # Load input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use a dict to merge companies by name
    merged = {}
    for item in data:
        name = item['name']
        field_id = item['fieldId']
        if name not in merged:
            merged[name] = item.copy()
            merged[name]['fieldId'] = [field_map[field_id]]  # Convert to string array
        else:
            # Avoid duplicate field names
            mapped_field = field_map[field_id]
            if mapped_field not in merged[name]['fieldId']:
                merged[name]['fieldId'].append(mapped_field)

    # Convert merged dict back to list
    merged_list = list(merged.values())

    # Save output JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} entries into {len(merged_list)} unique companies.")

# Example usage
input_file = "data.json"
output_file = "cleaned_data.json"
merge_companies(input_file, output_file)
