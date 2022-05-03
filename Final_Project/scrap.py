import pandas as pd

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Time-Series-Analysis-and-Moldeing\Final_Project\AAPL.csv')
df['Date']=pd.to_datetime(df['Date'])

print(min(df['Date']))