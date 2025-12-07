# Crime Against Women in India: Analysis and Forecasting

## Introduction

This project aims to analyze the trends of crimes against women in India using a dataset of reported cases. The project involves exploratory data analysis (EDA) to understand the distribution and trends of different crimes, and time series forecasting to predict future crime rates.

## Dataset

The dataset used in this project is `dataset/crime_against_women.csv`, which contains data on various crimes against women reported in different states and districts of India from 2017 to 2022.

### `crime_against_women.csv` Column Descriptions:

This dataset captures various incidents related to crimes against women across different states and districts in India. The columns are:

-   **`id`**: Unique identifier for each record.
-   **`year`**: The year the crime data was recorded.
-   **`state_name`**: The name of the state where the crime occurred.
-   **`state_code`**: A numerical code representing the state.
-   **`district_name`**: The name of the district where the crime occurred.
-   **`district_code`**: A numerical code representing the district.
-   **`registration_circles`**: Specific police registration circles or areas within districts.
-   **`murder_with_rape_gang_rape`**: Number of cases of murder associated with rape or gang rape.
-   **`dowry_deaths`**: Number of deaths related to dowry disputes.
-   **`abetment_to_suicide_of_women`**: Number of cases where women were abetted to commit suicide.
-   **`miscarriage`**: Number of cases related to miscarriage (context needs further domain knowledge for precise interpretation, likely related to forced miscarriage or related crimes).
-   **`acid_attack`**: Number of acid attack incidents.
-   **`attempt_to_acid_attack`**: Number of attempted acid attack incidents.
-   **`cruelty_by_husband_or_his_relatives`**: Cases of cruelty inflicted upon women by their husbands or relatives.
-   **`kidnapping_and_abduction`**: Total cases of kidnapping and abduction.
-   **`kidnapping_abduction_in_order_to_murder`**: Kidnapping/abduction with intent to murder.
-   **`kidnapping_for_ransom`**: Kidnapping for ransom.
-   **`kidnp_and_abductn_of_women_above_18_for_marrg`**: Kidnapping/abduction of women above 18 for marriage.
-   **`kidnp_and_abductn_of_girls_below_18_for_marrg`**: Kidnapping/abduction of girls below 18 for marriage.
-   **`procuration_of_minor_girls`**: Cases of procuring minor girls (likely for illegal activities).
-   **`importation_of_girls_from_foreign`**: Cases of importing girls from foreign countries (likely for exploitation).
-   **`kidnapping_and_abduction_of_women_others`**: Other cases of kidnapping and abduction of women.
-   **`human_trafficking`**: Cases of human trafficking.
-   **`selling_of_minor_girls`**: Cases involving the selling of minor girls.
-   **`buying_of_minor_girls`**: Cases involving the buying of minor girls.
-   **`rape_women_above_18`**: Number of rape cases where the victim is above 18 years old.
-   **`rape_girls_below_18`**: Number of rape cases where the victim is below 18 years old.
-   **`attempt_to_commit_rape_above_18`**: Attempted rape cases where the victim is above 18 years old.
-   **`attempt_to_commit_rape_girls_below_18`**: Attempted rape cases where the victim is below 18 years old.
-   **`assault_on_women_above_18`**: Cases of assault on women above 18.
-   **`assault_on_women_below_18`**: Cases of assault on women below 18.
-   **`insult_to_the_modesty_of_women_above_18`**: Cases of insulting the modesty of women above 18.
-   **`insult_to_the_modesty_of_women_below_18`**: Cases of insulting the modesty of women below 18.
-   **`dowry_prohibition`**: Cases registered under the Dowry Prohibition Act.
-   **`procuring_inducing_children_for_the_sake_of_prostitution`**: Procuring or inducing children for prostitution.
-   **`detaining_a_prsn_in_premises_where_prost_is_carried`**: Detaining a person in premises where prostitution is carried out.
-   **`prostitution_in_or_in_the_vicinity_of_public_places`**: Prostitution in or near public places.
-   **`seducing_or_soliciting_for_purpose_of_prostitution`**: Seducing or soliciting for prostitution.
-   **`other_sections_under_itp_act`**: Cases under other sections of the Immoral Traffic (Prevention) Act.
-   **`protection_of_women_from_domestic_violence`**: Cases under the Protection of Women from Domestic Violence Act.
-   **`publshng_or_transmitting_of_sexually_explicit_mtrl`**: Publishing or transmitting sexually explicit material.
-   **`other_women_centric_cyber_crimes`**: Other cybercrimes targeting women.
-   **`child_rape`**: Cases of child rape.
-   **`sexual_assault_of_children`**: Cases of sexual assault of children.
-   **`child_sexual_harassment`**: Cases of child sexual harassment.
-   **`use_of_child_for_pornography`**: Cases involving the use of children for pornography.
-   **`offences_of_pocso_act`**: Cases under the Protection of Children from Sexual Offences (POCSO) Act.
-   **`pocso_act_unnatural_offences`**: Unnatural offenses under the POCSO Act.
-   **`indecent_representation_of_women`**: Cases related to indecent representation of women.

Most crime-related columns (`murder_with_rape_gang_rape` onwards) contain numerical counts, and may include `0.0` or empty strings (`""`) for years where data is not available.


## Exploratory Data Analysis

The EDA is performed in the `eda/eda.py` script. The script performs the following steps:
1.  Loads the dataset and displays summary statistics.
2.  Checks for missing values.
3.  Generates and saves the following plots in the `eda/plots` directory:
    *   `crime_distribution.png`: A bar plot showing the distribution of different crimes.
    *   `total_crimes_over_years.png`: A line plot showing the trend of total crimes over the years.
    *   `severe_crimes_over_years.png`: A line plot showing the trend of some of the most severe crimes over the years.

## Time Series Forecasting

The `eda/model.py` script performs time series forecasting to predict the total number of crimes for the next 5 years. The script uses Holt's linear trend model for forecasting. The forecast plot is saved as `eda/plots/forecast.png`.

## Regression Analysis

The `eda/eda-2.py` script builds a linear regression model to predict the total number of crimes based on a set of selected crime features. The script also generates a scatter plot of actual vs. predicted values, which is saved as `eda/plots/actual_vs_predicted.png`.

## How to Run

1.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```
2.  **Run the EDA script:**
    ```
    python eda/eda.py
    ```
3.  **Run the time series forecasting script:**
    ```
    python eda/model.py
    ```
4.  **Run the regression analysis script:**
    ```
    python eda/eda-2.py
    ```

## Other Files

The `sklearn` directory and the files `MLtask.md`, `mlREADME.md`, and the original `README.md` contain generic machine learning examples and are not part of this project.