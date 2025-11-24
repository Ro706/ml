import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns

# Load the dataset
df = pd.read_csv('eda/dataset/crime_aginest_women.csv')

# Clean column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()

# List of crime columns to sum
crime_columns = [
    'murder_with_rape_gang_rape', 'dowry_deaths', 'abetment_to_suicide_of_women',
    'miscarriage', 'acid_attack', 'attempt_to_acid_attack',
    'cruelty_by_husband_or_his_relatives', 'kidnapping_and_abduction',
    'kidnapping_abduction_in_order_to_murder', 'kidnapping_for_ransom',
    'kidnp_and_abductn_of_women_above_18_for_marrg',
    'kidnp_and_abductn_of_girls_below_18_for_marrg', 'procuration_of_minor_girls',
    'importation_of_girls_from_foreign', 'kidnapping_and_abduction_of_women_others',
    'human_trafficking', 'selling_of_minor_girls', 'buying_of_minor_girls',
    'rape_women_above_18', 'rape_girls_below_18', 'attempt_to_commit_rape_above_18',
    'attempt_to_commit_rape_girls_below_18', 'assault_on_womenabove_18',
    'assault_on_women_below_18', 'insult_to_the_modesty_of_women_above_18',
    'insult_to_the_modesty_of_women_below_18', 'dowry_prohibition',
    'procuring_inducing_children_for_the_sake_of_prostitution',
    'detaining_a_prsn_in_premises_where_prost_is_carried',
    'prostitution_in_or_in_the_vicinity_of_public_places',
    'seducing_or_soliciting_for_purpose_of_prostitution',
    'other_sections_under_itp_act', 'protection_of_women_from_domestic_violence',
    'publshng_or_transmitting_of_sexually_explicit_mtrl',
    'other_women_centric_cyber_crimes', 'child_rape', 'sexual_assault_of_children',
    'child_sexual_harassment', 'use_of_child_for_pornography', 'offences_of_pocso_act',
    'pocso_act_unnatural_offences', 'indecent_representation_of_women'
]

for col in crime_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].mean(), inplace=True)

df['total_crimes'] = df[crime_columns].sum(axis=1)

# Define features and target
features = [
    'dowry_deaths',
    'acid_attack',
    'cruelty_by_husband_or_his_relatives',
    'kidnapping_and_abduction',
    'rape_women_above_18',
    'rape_girls_below_18',
]
target = 'total_crimes'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2 score:", r2_score(y_test, y_pred))

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-')
plt.grid(True)
plt.savefig('eda/plots/actual_vs_predicted.png')
print("Plot saved to eda/plots/actual_vs_predicted.png")
