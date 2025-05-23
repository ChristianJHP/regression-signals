import matplotlib.pyplot as plt

# Data
symbols = ['ARB', 'RUNE', 'SOL', 'ADA', 'AVAX', 'ETH', 'RNDR', 'INJ', 'FET', 'MATIC']
percentage_errors = [9.0, 0.9, 5.7, 2.7, -4.4, -2.5, -7.8, -12.4, -9.5, -14.1]

# Create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(symbols, percentage_errors, color=['green' if x >= 0 else 'red' for x in percentage_errors])
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Percentage Error of Predicted Prices on May 22, 2025')
plt.xlabel('Cryptocurrency')
plt.ylabel('Percentage Error (%)')

# Annotate bars with percentage error
for bar, error in zip(bars, percentage_errors):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{error:.1f}%', va='bottom' if yval >= 0 else 'top', ha='center')

plt.tight_layout()
plt.show()