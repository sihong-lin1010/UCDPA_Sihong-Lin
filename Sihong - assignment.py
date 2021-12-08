import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get data from csv file
data = pd.read_csv("owid-covid-data.csv", parse_dates = ['date'])

# take a first look to understand data
print(data.head())
print(data.info())
print(data.shape)
print(data.isnull().sum())
print([column for column in data])

# remove unwanted columns
data = data.drop(['stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older',
                  'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
                  'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
                  'life_expectancy', 'human_development_index', 'excess_mortality_cumulative_absolute',
                  'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million'],
                 axis=1)

# A list of countries selected
location_select = ["Bahrain", "Bolivia", "Brazil", "Canada", "Chile", "China", "Colombia", "Finland",
                   "Hungary", "India", "Indonesia", "Japan", "Kenya", "Latvia", "Malaysia", "Mexico",
                   "Mongolia", "Myanmar", "Nepal", "Netherlands", "Oman", "Philippines", "Portugal",
                   "Qatar", "Russia", "Seychelles", "Singapore", "South Korea", "Sweden", "Spain",
                   "Thailand", "Turkey", "Vietnam"]
# *************************************************************************************************************
# 1. Analyse vaccination progress based on a list of countries selected
# fill the values using forward fill method
data["people_vaccinated_per_hundred"] = data["people_vaccinated_per_hundred"].fillna(method='ffill',axis=0,limit=100)
data["people_fully_vaccinated_per_hundred"] = data["people_fully_vaccinated_per_hundred"].fillna(method='ffill',axis=0, limit=100)
# extract selected countries' most recent vaccination data
lis = []
for i in location_select:
    mid = data[data["location"] == i]
    mid = mid.sort_values("date")
    lis.append([mid.iloc[-1:]["people_vaccinated_per_hundred"].tolist()[0], mid.iloc[-1:]["people_fully_vaccinated_per_hundred"].tolist()[0]])
df_1 = pd.DataFrame(lis)
df_1.columns = "people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred"

# Visualize the selected countries' vaccination progress
plt.figure(figsize=(10,6),dpi=200)
plt.barh(df_1.index,df_1.iloc[:,0],label="percentage of people vaccinated")
plt.barh(df_1.index,df_1.iloc[:,1],label="percentage of people fully vaccinated")
plt.legend(loc=2, bbox_to_anchor=(0.1, 1.07),framealpha=0,ncol=2)
for k,i in enumerate(df_1.iloc[:,0]):
    plt.text(i+0.2,k-0.4,round(i,2),size=8)
plt.show()

# *************************************************************************************************************
# 2. visualize new cases per million for five continents
# select 5 continents data: iso_code=["OWID_AFR", "OWID_ASI", "OWID_EUR", "OWID_NAM", "OWID_SAM"]
df_2_1 = data[data["iso_code"].isin(["OWID_AFR", "OWID_ASI", "OWID_EUR", "OWID_NAM", "OWID_SAM"])]

# fill 0 for missing values in column ['new_cases_smoothed_per_million']
df_2_1['new_cases_smoothed_per_million'].fillna(0, inplace=True)

# extract data of 'new_cases_smoothed_per_million' for Africa and group the data by week
df_2_2 = df_2_1[df_2_1["iso_code"] == "OWID_AFR"].set_index("date")[["new_cases_smoothed_per_million"]].resample("w").sum()
# extract data of 'new_cases_smoothed_per_million' for the rest of 4 continents and group the data by week
for i in ["OWID_ASI", "OWID_EUR", "OWID_NAM", "OWID_SAM"]:
    mid_df = df_2_1[df_2_1["iso_code"] == i].set_index("date")[["new_cases_smoothed_per_million"]].resample("w").sum()
    # merge Africa's data with other 4 continents' data
    df_2_2 = pd.merge(df_2_2,mid_df,how="left", on=df_2_2.index)
    df_2_2 = df_2_2.set_index("key_0")
df_2_2.columns = ["Africa", "Asia", "Europe", "North America", "South America"]

# Visualize data of 'new_cases_smoothed_per_million' for five continents
plt.figure(figsize=(10,6),dpi=200)
for index, row in df_2_2.T.iterrows():
    plt.plot(row,label=index)
plt.xlabel("date")
plt.ylabel("new_cases_smoothed_per_million")
plt.legend()
plt.show()

# *************************************************************************************************************
# 3. visualize mortality risk for 5 continents
df_2_1["case_fatality_rate"] = df_2_1["total_deaths"]/df_2_1["total_cases"]*100
# calculate the mortality risk for Africa on a weekly basis
df_2_2 = df_2_1[df_2_1["iso_code"] == "OWID_AFR"].set_index("date")[["case_fatality_rate"]].resample("w").sum()
for i in ["OWID_ASI", "OWID_EUR", "OWID_NAM", "OWID_SAM"]:
    mid_df = df_2_1[df_2_1["iso_code"] == i].set_index("date")[["case_fatality_rate"]].resample("w").sum()
    # merge Africa's data with other 4 continents' data
    df_2_2 = pd.merge(df_2_2,mid_df,how="left", on=df_2_2.index)
    df_2_2 = df_2_2.set_index("key_0")
df_2_2.columns = ["Africa", "Asia", "Europe", "North America", "South America"]

# Visualize data of 'mortality risk rate' for five continents
plt.figure(figsize=(10,6),dpi=200)
for index, row in df_2_2.T.iterrows():
    plt.plot(row,label=index)
plt.xlabel("date")
plt.ylabel("mortality risk rate (%)")
plt.legend()
plt.show()

# *************************************************************************************************************
# 4. explore correlation among new_cases, new_deaths, percentage of people vaccinated, percentage of people fully vaccinated
data.fillna(0, inplace=True)
data_hot = data.iloc[:,[12,14,41,42]]
# calculate the correlation coefficient
corr = data_hot.corr()
plt.figure(figsize=(6,6),dpi=200)
# visualize the correlation heatmap
sns.heatmap(corr, cmap='GnBu_r', square=True, annot=True)
plt.show()