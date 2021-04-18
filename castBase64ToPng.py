import io, base64
import json
from PIL import Image
name = "shishko_y_v"
count = 0
with open(f"./datasets/{name}.txt", "r") as data:
    item = data.readline()
    while item:
        tmp = item[0:-2]
        x = json.loads(tmp)
        coded_str = x['photo']
        im = Image.open(io.BytesIO(base64.b64decode(coded_str.split(',')[1])))
        im.save(f"./uploads/images/{str(count)+name}.jpg")
        with open(f"./uploads/results/{str(count)+name}.txt", "w") as res:
            # res.write(f"x={x['x']},\n y={x['y']},\n radius={x['radius']}")
            res.write(json.dumps({'x': x['x'], 'y': x['y'], 'radius': x['radius']}))
        item = data.readline()
        count+=1
