"""
Step 1: Data Simulation
We'll create a dataset simulating the hull integrity variables: 
- corrosion rate
- hull thickness measurements
- watertight door status

Step 2: Data Analysis
We'll conduct a basic analysis to identify sections of the hull that might be at higher risk, 
using simulated data.

Step 3: Risk Assessment
Based on the analysis, we'll highlight areas that require closer inspection or maintenance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Data Simulation
np.random.seed(0)
data_size = 1000
corrosion_rate = np.random.rand(data_size) * 10  # Simulating corrosion rate
hull_thickness = np.random.rand(data_size) * 100  # Simulating hull thickness measurements
watertight_door_status = np.random.randint(0, 2, data_size)  # 0 for closed, 1 for issues detected

# Creating a DataFrame
hull_df = pd.DataFrame({
    'corrosion_rate': corrosion_rate,
    'hull_thickness': hull_thickness,
    'watertight_door_status': watertight_door_status
})

# Calculating risk indicator
hull_df['risk_indicator'] = hull_df['corrosion_rate'] / hull_df['hull_thickness']

# Identifying high-risk areas
high_risk_threshold = np.median(hull_df['risk_indicator'])
hull_df['risk_level'] = ['High' if x > high_risk_threshold else 'Low' for x in hull_df['risk_indicator']]

# Plotting
plt.figure(figsize=(10, 6))
for risk_level, color in [('Low', 'blue'), ('High', 'red')]:
    subset = hull_df[hull_df['risk_level'] == risk_level]
    plt.scatter(subset['hull_thickness'], subset['corrosion_rate'], c=color, label=f"{risk_level} Risk")

plt.title('Hull Integrity Analysis')
plt.xlabel('Hull Thickness')
plt.ylabel('Corrosion Rate')
plt.legend()
plt.grid(True)
plt.show()

# The scatter plot visualizes the hull integrity analysis. 
# Points represent individual measurements, with the color indicating the risk level: 
# blue for low risk and red for high risk. The plot correlates hull thickness with corrosion rate, 
# providing a clear visual cue to identify areas that may be at a higher risk and thus require closer 
# inspection or maintenance.
