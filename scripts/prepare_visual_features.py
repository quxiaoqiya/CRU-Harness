"""Local numeric screenshot features. Run in the isolated Pillow environment.

No thumbnails/captions are saved or sent; inference keeps original PNG bytes.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import hashlib,json,sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'work/longmemeval_v2_visual_holdout_20261103'


def feature(item):
 relative,expected=item;raw=(W/'data'/relative).read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if sha!=expected:raise ValueError('Original image changed')
 with Image.open(BytesIO(raw)) as image:
  gray=image.convert('L')
  h=list(gray.resize((9,8),Image.Resampling.LANCZOS).getdata())
  v=list(gray.resize((8,9),Image.Resampling.LANCZOS).getdata())
  horizontal=sum(int(h[y*9+x]>h[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
  vertical=sum(int(v[y*8+x]>v[(y+1)*8+x])<<(y*8+x) for y in range(8) for x in range(8))
  colors=[c for px in image.convert('RGB').resize((8,8),Image.Resampling.BOX).getdata() for c in px]
 return {'path':relative,'image_sha256':sha,'horizontal':horizontal,'vertical':vertical,'colors':colors}


def main():
 assets=json.loads((W/'assets_manifest.json').read_text());items=sorted({**assets['history_images'],**assets['query_images']}.items())
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(feature,items))
 out=W/'visual_features.json';out.write_text(json.dumps({'algorithm':'dHash-horizontal64+vertical64+RGB8x8','pillow':'11.3.0','features':rows},separators=(',',':'))+'\n')
 print(json.dumps({'numeric_feature_images':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'model_calls':0,'original_images_changed':False}))


if __name__=='__main__':main()
