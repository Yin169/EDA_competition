import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go 

file = "final_sdf.csv"
df = pd.read_csv(file)
df.head()
fig = go.Figure(data=go.Isosurface(
    x=df.x,
    y=df.y,
    z=df.z,
    value=df.value,
    surface_count=1,
    # slices= {
    #     'x': {'show': True, 'locations': [df.x.max()/2]},
    #     'y': {'show': True, 'locations': [df.y.max()/2]},
    #     'z': {'show': True, 'locations': [df.z.max()/2]}
    # },
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale='Viridis',
    # opacity=0.6
))

# Add axis labels and title
fig.update_layout(
    title='3D Volumetric Slice Visualization',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)
fig.show()