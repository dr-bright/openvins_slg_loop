#!/usr/bin/bash

git clone --recurse-submodules https://github.com/urbste/py-gpmf-parser.git /tmp/py-gpmf-parser
cd /tmp/py-gpmf-parser
python3 -c "
import sys
file = open(sys.argv[1], 'r+t', encoding='utf-8')
text = file.read()
tok1 = 'license = \"Apache-2.0\"'
tok2 = 'license = { text = \"Apache-2.0\" }'
tok3 = 'license-files = [\"LICENSE\"]'
print(f'tok1: {tok1}, tok2: {tok2}, tok3: {tok3}')
text = text.replace(tok1, tok2)
text = text.replace(tok3, '')
file.seek(0)
file.truncate()
file.write(text)
file.close()

"   pyproject.toml
python3 -m build --wheel
pip3 install dist/*.whl
cd ..
rm -rf /tmp/py-gpmf-parser