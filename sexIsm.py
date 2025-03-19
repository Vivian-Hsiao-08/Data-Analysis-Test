import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = "data/salary_data.csv"
df = pd.read_csv(file_path)

# Ensure relevant columns are valid
df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Drop NaN values
df = df.dropna(subset=["Years of Experience", "Salary", "Gender"])

# Separate data by gender
male_data = df[df["Gender"] == "Male"]
female_data = df[df["Gender"] == "Female"]

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(male_data["Years of Experience"], male_data["Salary"], color='b', alpha=0.5, label="Male")
plt.scatter(female_data["Years of Experience"], female_data["Salary"], color='r', alpha=0.5, label="Female")

# Fit best-fit lines and plot them with different colors
if len(male_data) > 1:  # Ensure there's enough data
    male_m, male_b = np.polyfit(male_data["Years of Experience"], male_data["Salary"], 1)
    plt.plot(male_data["Years of Experience"], male_m * male_data["Years of Experience"] + male_b, 
             color='darkblue', linestyle='dashed', linewidth=2.5, label=f"Male Best Fit: y={male_m:.2f}x+{male_b:.2f}")
    
if len(female_data) > 1:  # Ensure there's enough data
    female_m, female_b = np.polyfit(female_data["Years of Experience"], female_data["Salary"], 1)
    plt.plot(female_data["Years of Experience"], female_m * female_data["Years of Experience"] + female_b, 
             color='darkred', linestyle='dashed', linewidth=2.5, label=f"Female Best Fit: y={female_m:.2f}x+{female_b:.2f}")

# Labels and title
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Comparison by Gender with Best Fit Lines")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()