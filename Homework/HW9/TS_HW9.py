from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt

# q 1
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print(df.describe())

# q 2
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

# q 3
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# q 4
print(df.dtypes)
features = df.columns
for feature in features:
    if df[feature].dtype == 'object':
        df[feature].fillna(value=df[f'{feature}'].mode(), inplace=True)
    elif df[feature].dtype == 'int64' or df[feature].dtype == 'float64' or df[feature].dtype == 'numeric':
        df[feature].fillna(value=df[f'{feature}'].mode(), inplace=True)
print(df.isna().sum())

# q 5 - 8
durations = df['tenure']
event_observed = df['Churn']
ax = plt.subplot(111)
kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed,label='Customer Retention')
kmf.plot_survival_function(ax=ax)
plt.show()

