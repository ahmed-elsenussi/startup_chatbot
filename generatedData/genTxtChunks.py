import json

def prepare_statements(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Combine name, description, and fields into a single statement
        fields_str = ", ".join(item['fieldId'])
        statement = f"{item['name']} is associated with the fields: {fields_str}. {item['description']}"
        # Add a new key for the prepared text
        item['prepared_text'] = statement

    # Save the updated data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Prepared statements for {len(data)} companies.")

# Example usage
input_file = "cleaned_data.json"
output_file = "prepared_data.json"
prepare_statements(input_file, output_file)
