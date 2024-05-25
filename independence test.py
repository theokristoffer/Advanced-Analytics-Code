import pandas as pd
import json
from scipy.stats import chi2_contingency

community_mined = r"C:\Users\Theo\Documents\Advanced Analytics\Graph Files\Community_Mining.json"

# cm_data = pd.read_json(community_mined)

data_list=[]

# Open the JSON file and read its contents
with open(community_mined, "r", encoding="utf-8") as file:
    for line in file:
        # Parse the JSON data in the line
        try:
            json_data = json.loads(line)
            # Extract relevant data and append to the list
            node_data = json_data['node']['properties']
            node_data['community_id'] = json_data['community_id']
            data_list.append(node_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)
print(df[1:100])
print(df.columns)
df.to_json(r'C:\\Users\\Theo\Documents\\Advanced Analytics\\graph analysis\\formatted_communities.json')

frequencies = df['community_id'].value_counts()
print(frequencies)


contingency_table = pd.crosstab(df['community_id'], df['lang'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

print("Contingency Table:")
print(contingency_table)

# Print the chi-square test result
print("\nChi-Square Test of Independence:")
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_value}")

integer_freq = df['community_id'].value_counts()

top_10_values = df['community_id'].value_counts().head(10).index

# Step 2 and 3: Iterate through the top values, find the most common entry in the language column, and calculate the ratio
ratios = []
for value in top_10_values:
    # Filter the DataFrame for the current value
    filtered_df = df[df['community_id'] == value]
    # Find the most common entry in the language column
    most_common_language = filtered_df['lang'].mode().iloc[0]
    # Calculate the frequency of each language for the current value
    language_freq = filtered_df['lang'].value_counts()
    # Calculate the ratio of entries for each language
    ratio = language_freq / language_freq.sum()
    # Append the results to the ratios list
    ratios.append({'value': value, 'most_common_language': most_common_language, 'ratios': ratio})

# Convert the results to a DataFrame
result_df = pd.DataFrame(ratios)

result_df.to_json(r'C:\\Users\\Theo\Documents\\Advanced Analytics\\graph analysis\\ratios_csv.json')

# Print the result
print(result_df)