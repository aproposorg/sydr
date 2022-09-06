import panel as pn

import hvplot.pandas

from bokeh.sampledata.autompg import autompg_clean

pn.extension(sizing_mode="stretch_width")

quant = [None, 'mpg', 'cyl']
cat = [None, 'origin']
combined = quant+cat[1:]

x = pn.widgets.Select(name='x', value='mpg', options=combined)
y = pn.widgets.Select(name='y', value='cyl', options=combined)
color = pn.widgets.Select(name='color', options=combined)
facet = pn.widgets.Select(name='facet', options=cat)

@pn.depends(x, y, color, facet)
def plot(x, y, color, facet):
    cmap = 'Category10' if color in cat else 'viridis'
    return autompg_clean.hvplot.scatter(
        x, y, color=color or 'green', by=facet, subplots=True, padding=0.1,
        cmap=cmap, responsive=True, min_height=500, size=100
    )

settings = pn.Row(pn.WidgetBox(x, y, color, facet))
panel = pn.Column(
    '### Auto MPG Explorer', 
    settings,
    plot,
    width_policy='max'
)

panel.save('test.html', embed=True)
