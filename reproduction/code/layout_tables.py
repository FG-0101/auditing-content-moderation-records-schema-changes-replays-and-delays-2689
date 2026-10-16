"""Reflow table headings and columns without changing data cells."""
from pathlib import Path
import re
import textwrap


def balanced(text, start):
    depth = 0
    for end in range(start, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return end
    raise ValueError('Unbalanced column specification')


def headings(text):
    replacements = {
        'Separate numerator/count box': r'\shortstack{Separate numerator/\\count box}',
        '95\\% percentile interval': r'\shortstack{95\% percentile\\interval}',
        'Statement jump ($p$)': r'\shortstack{Statement\\jump ($p$)}',
        'Platform jump ($p$)': r'\shortstack{Platform\\jump ($p$)}',
        'Local linear jump': r'\shortstack{Local linear\\jump}',
        'Difference in means': r'\shortstack{Difference\\in means}',
        'Statements per day': r'\shortstack{Statements\\per day}',
        'Share of keyword mentions (\\%)': r'\shortstack{Share of keyword\\mentions (\%)}',
        'Keyword without an aggregate column': r'\shortstack[l]{Keyword without an\\aggregate column}',
        'Share of the count of 30 September 2026 known': r'\shortstack[l]{Share of the count of\\30 September 2026 known}',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def reflow(text, name='', key=''):
    if name == 'tab_rd.tex' and key == 'P06' and 'Statement weighted' in text and 'Platform weighted' in text:
        rows = []
        for line in text.splitlines():
            if line.endswith('\\\\') and ' & ' in line and not any(word in line for word in ['multicolumn', 'Outcome']):
                cells = line[:-2].split(' & ')
                if len(cells) == 11:
                    rows.append(cells)
        if rows:
            blocks = []
            for panel, label, indices in [('A', 'Statement weighted', [0,1,2,3,4,9,10]), ('B', 'Platform weighted', [0,5,6,7,8,9,10])]:
                blocks.append(r'\begin{tabular}{p{5.0cm}rrrrrr}\toprule' + '\n' +
                              r'\multicolumn{7}{l}{' + panel + '. ' + label + r'}\\' + '\n' +
                              r'Outcome & Before & After & Jump (s.e.) & $p$ & Up & Down\\\midrule' + '\n' +
                              '\n'.join(' & '.join(cells[i] for i in indices) + r'\\' for cells in rows) +
                              '\n' + r'\bottomrule\end{tabular}')
            text = '\n\\par\\medskip\n'.join(blocks) + '\n'
    specs = {
        ('P01','tab_dup.tex'): 'p{2.8cm}lrrrrrrrr',
        ('P01','tab_rules.tex'): 'p{8.5cm}rrr',
        ('P01','tab_dates.tex'): 'p{5.5cm}r',
        ('P01','tab_timeliness.tex'): 'lrrrr',
        ('P01','tab_capacity.tex'): 'p{6.6cm}rrrr',
        ('P01','tab_enforcement.tex'): 'p{6.3cm}rrrr',
        ('P01','tab_summary.tex'): r'p{0.40\linewidth}p{0.13\linewidth}p{0.40\linewidth}',
        ('P06','tab_hidden_kw.tex'): 'p{5.5cm}p{5.5cm}rr',
        ('P06','tab_hidden_cat.tex'): 'p{8.0cm}rrr',
        ('P06','tab_bandwidth.tex'): 'p{5.0cm}rrrrr',
        ('P06','tab_crosswalk.tex'): 'p{2.5cm}rrrrp{5.4cm}',
        ('P10','tab_platforms.tex'): 'p{3.7cm}rrrrrrr',
        ('P10','tab_vintage.tex'): 'p{8.3cm}rrrrr',
        ('P10','tab_spikes.tex'): 'p{3.3cm}lrrrr',
        ('P10','tab_rd.tex'): 'p{3.0cm}p{2.2cm}rrrrrrr',
        ('P10','tab_rdplat.tex'): 'p{3.3cm}p{2.6cm}rrrrrr',
        ('P10','tab_robust.tex'): 'p{5.1cm}p{3.0cm}rrr',
        ('P10','tab_nowcast.tex'): 'p{5.4cm}rrrrrr',
    }
    # The three panel notes are paragraphs, outside the tabular width calculation.
    text = text.replace('keywords; visible', 'keywords. Visible').replace('March 2025; March/May', 'March 2025. March/May')
    starts = list(re.finditer(r'\\begin\{tabular\}\{', text))
    for match in reversed(starts):
        start = match.end()-1
        end = balanced(text, start)
        spec = specs.get((key,name), text[start+1:end])
        text = text[:match.start()] + r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' + spec + '@{}}' + text[end+1:]
    text = text.replace(r'\end{tabular}', r'\end{tabular*}')
    # Wrap only column headings. Numeric rows retain their exact spelling.
    for m in reversed(list(re.finditer(r'\\begin\{tabular\*\}.*?\\midrule', text, re.S))):
        original = m.group()
        changed = headings(original)
        lines = changed.splitlines()
        for i, line in enumerate(lines):
            if ' & ' in line and 'multicolumn' not in line and 'cmidrule' not in line:
                cells = line.split(' & ')
                for j, cell in enumerate(cells):
                    suffix = ''
                    if cell.endswith(r'\\\midrule'):
                        cell, suffix = cell[:-10], r'\\\midrule'
                    elif cell.endswith(r'\\'):
                        cell, suffix = cell[:-2], r'\\'
                    if len(cell) > 16 and '\\' not in cell and '$' not in cell:
                        cell = r'\shortstack{' + r'\\'.join(textwrap.wrap(cell, 15, break_long_words=False, break_on_hyphens=False)) + '}'
                    cells[j] = cell + suffix
                lines[i] = ' & '.join(cells)
        text = text[:m.start()] + '\n'.join(lines) + text[m.end():]
    return text


def format_generated_tables(key, folder=Path('tex')):
    for file in folder.glob('tab_*.tex'):
        file.write_text(reflow(file.read_text(), file.name, key))


if __name__ == '__main__':
    import sys
    format_generated_tables(sys.argv[1])
