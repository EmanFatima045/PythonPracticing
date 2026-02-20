import numpy as np
import os

print("Working Directory:", os.getcwd())

# Load CSV correctly (force as string)
data = np.genfromtxt(
    "employee_salary.csv",
    delimiter=",",
    skip_header=1,
    dtype=str
)

print("Shape of Data:", data.shape)

# If still 1D, stop program
if len(data.shape) == 1:
    print("ERROR: File not loaded correctly. Check delimiter or CSV format.")
    exit()

# Extract numeric columns
# 0 = EmployeeID
# 3 = Experience_Years
# 5 = Age
# 8 = Monthly_Salary
numeric_data = data[:, [0, 3, 5, 8]].astype(float)

print("\nNumeric Data:")
print(numeric_data)

# Remove missing values
clean_data = numeric_data[~np.isnan(numeric_data).any(axis=1)]

# Remove duplicates
unique_data = np.unique(clean_data, axis=0)

# Sort employees by Monthly Salary (ascending)
sorted_data = unique_data[unique_data[:, 3].argsort()]

print("\nSorted by Salary:")
print(sorted_data)

# Save cleaned & sorted data
np.savetxt(
    "cleaned_employee_salary.csv",
    sorted_data,   # saving sorted data (not normalized)
    delimiter=",",
    header="EmployeeID,Experience_Years,Age,Monthly_Salary",
    comments="",
    fmt="%.2f"
)
print("\nCleaned and Sorted file saved successfully!")

