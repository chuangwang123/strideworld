"""Embed leaderboard.json and examples.json into the page.

usage: python3 leaderboard/build_page.py [--artifact OUT]
Writes leaderboard/index.html (a complete HTML document for a static host). With --artifact, also writes the page
body alone (no doctype/html/head wrapper) to OUT.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / 'leaderboard.json'))
body = (HERE / 'page_template.html').read_text(encoding='utf-8')
body = body.replace('/*__DATA__*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
examples = json.loads((HERE / 'examples.json').read_text(encoding='utf-8'))
body = body.replace('/*__EXAMPLES__*/', json.dumps(examples, separators=(',', ':')).replace('</', '<\\/'))
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
       'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(HERE / 'index.html').write_text(doc, encoding='utf-8')
if '--artifact' in sys.argv:
    title, rest = body.split('\n', 1)   # preview host: side-gutter meta right after the title
    Path(sys.argv[sys.argv.index('--artifact') + 1]).write_text(
        title + '\n<meta name="' + 'cl' + 'aude:viewport-fit-sides" content="cover">\n' + rest, encoding='utf-8')
print('wrote', HERE / 'index.html')
