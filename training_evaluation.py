"""Execute the authoritative Phase 2 notebook code without duplicating it."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    os.chdir(ROOT)
    nb = json.loads((ROOT / 'Movie_Recommendation_Phase2.ipynb').read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
