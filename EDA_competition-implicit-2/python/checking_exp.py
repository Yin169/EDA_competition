import pandas as pd
import matplotlib.pyplot as plt

path = 'checking.csv'
df = pd.read_csv(path)
df = df[df["x"] <= 0]
df = df[df["material"] != "default"].reset_index(drop=True)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')

for grp_name, grp_idx in df.groupby('material').groups.items():
    y = df.iloc[grp_idx,1]
    x = df.iloc[grp_idx,0]
    z = df.iloc[grp_idx,2]
    ax.scatter(x, y, z, label=grp_name)

ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
plt.show()