import pandas as pd
import matplotlib.pyplot as plt

path = 'data/initial_struct_vertex_material.csv'
df = pd.read_csv(path)
df = df[df["X"] >=0].reset_index(drop=True)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')

for grp_name, grp_idx in df.groupby('Material').groups.items():
    y = df.iloc[grp_idx,2]
    x = df.iloc[grp_idx,1]
    z = df.iloc[grp_idx,3]
    ax.scatter(x, y, z, label=grp_name)

print(df.groupby(['Material'])['Z'].apply(lambda x: x.max() - x.min()).reset_index())

ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
plt.show()