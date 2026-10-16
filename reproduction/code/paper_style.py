"""Render figures at their printed column width using the manuscript font."""
from pathlib import Path
import hashlib
import json
import os
import textwrap
import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib.text import Text

FONT_DIRECTORY = Path(__file__).with_name('fonts')
for font in FONT_DIRECTORY.glob('*.otf'):
    font_manager.fontManager.addfont(str(font))
matplotlib.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': 'Linux Libertine O',
                            'mathtext.it': 'Linux Libertine O:italic', 'mathtext.bf': 'Linux Libertine O:bold'})


def numerical_artists(figure):
    arrays = []
    for axes in figure.axes:
        for line in axes.lines:
            arrays.append(repr(line.get_xydata().tolist()))
        for collection in axes.collections:
            arrays.append(repr(np.asarray(collection.get_offsets()).tolist()))
            if collection.get_array() is not None:
                arrays.append(repr(np.asarray(collection.get_array()).tolist()))
        for patch in axes.patches:
            if all(hasattr(patch, method) for method in ['get_x', 'get_y', 'get_width', 'get_height']):
                arrays.append(repr([patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()]))
        arrays.append(repr([axes.get_xlim(), axes.get_ylim()]))
    return hashlib.sha256('\n'.join(arrays).encode()).hexdigest()


def save_figure(figure, filename, **options):
    before = numerical_artists(figure)
    width, height = figure.get_size_inches()
    full_width = width > 4.0
    target_width = 6.975 if full_width else 3.395
    rows = max((axes.get_subplotspec().get_gridspec().nrows for axes in figure.axes if axes.get_subplotspec() is not None), default=1)
    if full_width:
        target_height = 6.15 if rows == 4 else 4.85 if rows == 2 else 5.1 if height > 5 else 2.85
    else:
        target_height = max(2.1, height)
    figure.set_size_inches(target_width, target_height)
    for text in figure.findobj(Text):
        text.set_fontfamily('Linux Libertine O')
        text.set_math_fontfamily('custom')
        text.set_fontsize(8)
    for axes in figure.axes:
        for label in [axes.xaxis.label, axes.yaxis.label]:
            label.set_fontsize(8.5)
            original = label.get_text()
            if len(original) > 45 and '\n' not in original:
                label.set_text(textwrap.fill(original, width=38, break_long_words=False))
        for title in [axes.title, axes._left_title, axes._right_title]:
            title.set_fontsize(8.5)
            original = title.get_text().replace('; ', '\n')
            if len(original) > 48 and '\n' not in original:
                original = textwrap.fill(original, width=43, break_long_words=False)
            title.set_text(original)
        for label in axes.get_xticklabels() + axes.get_yticklabels():
            label.set_fontsize(8)
        legend = axes.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(8)
    figure.set_layout_engine('tight', pad=0.75, w_pad=1.8, h_pad=1.5)
    figure.canvas.draw()
    after = numerical_artists(figure)
    if before != after:
        raise ValueError('Figure styling changed plotted values or limits')
    options.pop('bbox_inches', None)
    options.pop('pad_inches', None)
    if Path(filename).suffix.lower() == '.png':
        options['metadata'] = {**(options.get('metadata') or {}), 'Software': None}
    figure.savefig(filename, **options)
    if os.environ.get('FIGURE_STYLE_RECEIPT'):
        record = {'file': str(filename), 'width_inches': target_width, 'height_inches': target_height,
                  'data_before_sha256': before, 'data_after_sha256': after, 'font': 'Linux Libertine O',
                  'tick_and_legend_pt': 8, 'label_and_title_pt': 8.5}
        with Path(os.environ['FIGURE_STYLE_RECEIPT']).open('a') as stream:
            stream.write(json.dumps(record) + '\n')
