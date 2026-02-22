import pandas as pd
#read_csv , series ,DataFrame , values , head , tail can be use py pandas
csv_file="employee_salary.csv"
df= pd.read_csv(csv_file)
df[["Monthly_Salary", "Department"]]
print(df)
df.head() # it will print first 5 rows of dataframe
df.tail() # it will print last 5 rows of dataframe
print(df.values)