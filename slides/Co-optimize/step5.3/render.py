"""Text-free process envelope pictures using the existing CPU renderer."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

import stage5_common as C


def box_corners(bounds):
    low, high = np.asarray(bounds)
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1]) for z in (low[2], high[2])])


def box_piece(bounds, camera, size):
    vertices = box_corners(bounds)
    quads = [[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5]]
    parts = [C.H.outline_piece(vertices[q], '#465966', camera, size, .75) for q in quads]
    return dict(triangles=np.concatenate([p['triangles'] for p in parts]),
        normals=np.concatenate([p['normals'] for p in parts]),
        colors=np.concatenate([p['colors'] for p in parts]), bias=4e-6, unlit=True)


def envelope(renderer, data, path, size=1200):
    camera = C.V.camera_for(box_corners(data['bounds']))
    parts, seen = [], set()
    for row in data['rows']:
        state = row['fixture_transform']
        key = np.round(state, 10).tobytes()
        if key not in seen:
            parts.append(C.V.CPU.placed(C.V.piece(C.posed(data['fixture'], state), C.V.BLUE, .18)))
            seen.add(key)
        parts.append(C.V.CPU.placed(C.V.piece(C.posed(data['body'], row['object_transform']), C.V.GRAY, .10)))
    parts.append(C.V.CPU.placed(box_piece(data['bounds'], camera, size)))
    image = renderer.render(parts, camera, size)
    image.save(path)
    return camera


def reference(renderer, data, path, fixture=None, size=900):
    state = data['states'][0]
    fixture = data['fixture'] if fixture is None else fixture
    polygons = [C.F.transform_points(C.F.transform_points(r['hull'], np.linalg.inv(r['fixture_transform'])), state)
                for r in data['rows']]
    posed = C.posed(fixture, state)
    camera = C.V.camera_for(np.vstack([posed.vertices, *polygons]))
    parts = [C.V.CPU.placed(C.V.piece(posed, C.V.BLUE))]
    for polygon, row in zip(polygons, data['rows']):
        parts.append(C.V.CPU.placed(C.H.outline_piece(polygon, row['color'], camera, size)))
    image = renderer.render(parts, camera, size)
    image.save(path)
    return image


def polygon_image(polygons, points=(), line=None, size=900):
    """An exact orthographic diagnostic; colors are described in the HTML."""
    arrays = [np.asarray(p) for p, _, _ in polygons if len(p)]
    arrays.extend(np.asarray(p).reshape(-1,2) for p, _ in points if len(p))
    joined = np.vstack(arrays)
    low, high = joined.min(0), joined.max(0)
    span = max(float((high-low).max())*1.16, .001)
    center = (low+high)/2
    scale = 3
    image = Image.new('RGB', (size*scale, size*scale), 'white')
    draw = ImageDraw.Draw(image)
    def screen(p):
        p = (np.asarray(p)-center)*size*scale/span+size*scale/2
        p[...,1] = size*scale-p[...,1]
        return [tuple(x) for x in p.reshape(-1,2)]
    for polygon, fill, outline in polygons:
        if len(polygon) >= 3:
            draw.polygon(screen(polygon), fill=fill)
            draw.line(screen(np.vstack([polygon, polygon[0]])), fill=outline, width=6)
        elif len(polygon):
            draw.line(screen(polygon), fill=outline, width=6)
    if line is not None:
        a, offset = line
        norm = float(a@a)
        if norm > 1e-20:
            origin = center-((center@a+offset)/norm)*a
            tangent = np.array([-a[1],a[0]])/np.sqrt(norm)
            draw.line(screen([origin-2*span*tangent,origin+2*span*tangent]), fill='#263e50', width=6)
    for locations, color in points:
        for x, y in screen(locations):
            r = 7*scale
            draw.line([(x-r,y-r),(x+r,y+r)], fill=color, width=3*scale)
            draw.line([(x-r,y+r),(x+r,y-r)], fill=color, width=3*scale)
    return image.resize((size,size), Image.Resampling.LANCZOS)
