import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

promoter = pd.read_csv('files/promoter_regions.tsv', sep='\t', header = None)

promoter_target = promoter.iloc[:,-1:]

no_promoter = pd.read_csv('files/non_promoter_regions.tsv', sep='\t', header = None)
no_promoter_target = no_promoter.iloc[:,-1:]

promoter_target["label"] = "promoter"

no_promoter_target["label"] = "no_promoter"

data = pd.concat([promoter_target, no_promoter_target],ignore_index=True)

import plotly.express as px
#df = px.data.tips()
fig = px.box(data, x="label", y=3, labels={
                     "label": " ",
                     "3": "Target value"
                 })
fig.update_traces(marker_color='darkblue')
#fig.update_xaxes(tickfont_size=20)
#fig.update_yaxes(tickfont_size=20)
#fig.update_yaxes(font=dict(size=20))

#fig.layout.yaxis.title.font(size=20)

fig.layout.template = 'plotly_white'

fig.show()

fig.write_image("Promoters.png")