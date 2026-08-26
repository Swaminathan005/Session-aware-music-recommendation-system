import plotly.express as px
import pandas as pd

plot_data = pd.read_csv("song_embedding_2d.csv")

fig = px.scatter(
    plot_data,
    x="x",
    y="y",
    color="artist",  # or a cluster/genre column
    hover_name="name",
    hover_data=["artist","tags"],
    render_mode="webgl",
    title="Song embedding space"
)

fig.update_traces(marker={"size": 10, "opacity": 0.65})

fig.update_layout(
    xaxis_title="UMAP dimension 1",
    yaxis_title="UMAP dimension 2",
    legend_title="Artist",
)

fig.show()