import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('notebooks/eth_predicted_next24hours.csv')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['hour_ahead'], df['predicted_close'], marker='o', linestyle='-', linewidth=2)

# Customize the plot
plt.title('ETH Price Predictions for Next 24 Hours', fontsize=14, pad=15)
plt.xlabel('Hours Ahead', fontsize=12)
plt.ylabel('Predicted Price (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Format y-axis to show full numbers
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

# Add some padding to the y-axis limits
y_min = df['predicted_close'].min() * 0.999
y_max = df['predicted_close'].max() * 1.001
plt.ylim(y_min, y_max)

# Rotate x-axis labels for better readability
plt.xticks(df['hour_ahead'], rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('eth_predictions.png', dpi=300, bbox_inches='tight')
plt.close() 