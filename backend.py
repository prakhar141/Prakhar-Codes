import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Simple Graph Example")

# Generate dummy data
data = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100))
})

# Plot with matplotlib
fig, ax = plt.subplots()
ax.plot(data['x'], data['y'], label='Sine Wave')
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Sine Wave Plot")
ax.legend()

st.pyplot(fig)
