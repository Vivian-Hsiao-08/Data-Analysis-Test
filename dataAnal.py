import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = "data/salary_data.csv"
df = pd.read_csv(file_path)

# Ensure the relevant columns are numeric
df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Remove NaN or infinite values
df = df.dropna(subset=["Years of Experience", "Salary"])
df = df[(df["Years of Experience"].apply(np.isfinite)) & (df["Salary"].apply(np.isfinite))]

# Extract cleaned variables
x = df["Years of Experience"]
y = df["Salary"]

# Ensure there's enough data for regression
if len(x) > 1:
    # Compute best-fit line using np.linalg.lstsq
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, color='b', label="Data Points")

    # Plot the line of best fit
    plt.plot(x, m*x + b, color='r', label="Best Fit Line")

    # Labels and title
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Experience vs. Salary Scatter Plot with Best Fit Line")
    plt.legend()
    plt.grid(True)

    # Display the equation of the line on the plot
    equation = f"y = {m:.2f}x + {b:.2f}"
    plt.text(min(x), max(y), equation, fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.5))

    # Show the plot
    plt.show()
else:
    print("Not enough valid data points to fit a regression line.")