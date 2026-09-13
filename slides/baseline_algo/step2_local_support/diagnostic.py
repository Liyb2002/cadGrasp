"""Display every sampled center and the saved process-access volume.

The smooth purple shell is a finite illustration, not collision geometry.
Candidate colors come from the stored checks, never from that shell.
"""
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from step2_local_support import render as R
from step4_floor_contact import draw_work_volume as V

COLORS = {'valid': '#267fcb', 'work_volume_collision': '#c43d48',
          'work_volume_unresolved': '#bd8400', 'other': '#48414e'}


def key(row):
    return row['status'] if row['status'] in COLORS else 'other'


def shell_overlay(picture, domain, shell, cap, basis, focus, width, size, ground):
    floor = R.floor_triangles(domain) if ground else np.empty((0, 3, 3))
    body = np.concatenate([floor, domain.mesh.triangles])
    _, depth = V.layer(body, np.tile(R.GREY, (len(body), 1)), focus, width, size, basis=basis)
    _, shell_depth = V.layer(shell.triangles, np.tile(V.PURPLE, (len(shell.faces), 1)),
                            focus, width, size, basis=basis)
    return V.composite_shell(picture, depth, shell_depth, focus, width, size, cap, basis)


def markers(picture, domain, data, report, basis, focus, width, size, labels=False):
    centers = data.centers_m
    normals = domain.mesh.face_normals[data.center_faces]
    visible = ~domain.mesh.ray.intersects_any(centers+1e-6*normals,
                                             np.tile(basis[2], (len(centers), 1)))
    xy = R.project(centers, focus, basis, width, size)[:, :2]
    ink = ImageDraw.Draw(picture)
    for i in np.argsort(visible):
        p = xy[i]
        if not ((p >= 6) & (p < size-6)).all():
            continue
        color = COLORS[key(report['patches'][i])]
        r = 4.5
        ink.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r),
                    fill=color if visible[i] else None, outline=color, width=1)
        if labels and visible[i]:
            ink.text((p[0]+6, p[1]-10), report['patches'][i]['id'],
                     font=R.font(14), fill=color, stroke_width=1, stroke_fill=R.PAPER)
    return visible, xy


def centers_page(name, pose, domain, data, report, work, folder):
    if work is None:
        return surface_only_page(name, pose, domain, data, report, folder)
    shell, cap = V.shell_mesh(work)
    views = [R.axes([.6, .55, -.9]), R.axes([-.6, -.65, .9])]
    paper = Image.new('RGB', (1900, 2140), R.PAPER)
    ink = ImageDraw.Draw(paper)
    ink.text((35, 25), f'{name} / {pose} / Step 2 / all {len(data.valid)} sampled centers',
             font=R.font(37), fill=R.INK)
    counts = Counter(key(r) for r in report['patches'])
    x = 40
    for k, label in [('valid', 'Accepted'), ('work_volume_collision', 'Access collision'),
                     ('work_volume_unresolved', 'Access unresolved'), ('other', 'Other geometry')]:
        ink.text((x, 90), f'{label}: {counts[k]}', font=R.font(24), fill=COLORS[k])
        x += 465
    ink.text((40, 129), 'Filled: visible centers. Hollow: behind the object. No vertical orientation filter.',
             font=R.font(24), fill=R.INK)
    cameras = []
    for j, basis in enumerate(views):
        points = np.vstack([domain.mesh.vertices, shell.vertices,
                            R.floor_triangles(domain).reshape(-1, 3)])
        q = points@basis.T
        low, high = q.min(axis=0), q.max(axis=0)
        focus = ((low+high)/2)@basis
        width = 1.07*max(high[:2]-low[:2])
        for row, overlay in enumerate((False, True)):
            pic, _ = R.scene(domain, data, basis, focus, width, 890,
                             ground=j == 0, show_patches=False, indices=[])
            if overlay:
                pic = shell_overlay(pic, domain, shell, cap, basis, focus, width, 890, j == 0)
            visible, xy = markers(pic, domain, data, report, basis, focus, width, 890)
            x, y = 35+940*j, 200+940*row
            ink.text((x, y-30), ('All samples' if not overlay else 'Same samples + process-access preview')+
                     (' / oblique' if j == 0 else ' / underside, floor omitted'), font=R.font(23), fill=R.INK)
            paper.paste(pic, (x, y))
        # An additional labeled image lets users trace a location back to JSON.
        detail, _ = R.scene(domain, data, basis, focus, width, 1600,
                             ground=j == 0, show_patches=False, indices=[])
        markers(detail, domain, data, report, basis, focus, width, 1600, labels=True)
        detail.save(folder/f'centers_labeled_{j+1}.png')
        cameras.append(dict(basis=basis.tolist(), focus_m=focus.tolist(), width_m=float(width),
                            visible_center_ids=[report['patches'][i]['id'] for i in np.flatnonzero(visible)]))
    ink.text((40, 2047), 'Green: work surface. Purple: finite smooth preview of the unbounded access volume.',
             font=R.font(23), fill=R.INK)
    ink.text((40, 2084), 'The preview fills gaps; stored geometric checks determine the red/yellow classifications.',
             font=R.font(23), fill=R.INK)
    paper.save(folder/'centers.png')
    return shell, cap, dict(all_sampled_centers_displayed=True, status_counts=dict(counts),
        cameras=cameras, preview_is_acceptance_geometry=False,
        preview='Finite smooth convex illustration; may fill gaps and includes unresolved families.',
        labeled_images=['centers_labeled_1.png', 'centers_labeled_2.png'])


def surface_only_page(name, pose, domain, data, report, folder):
    views = [R.axes([.6, .55, -.9]), R.axes([-.6, -.65, .9])]
    paper = Image.new('RGB', (1900, 1170), R.PAPER); ink = ImageDraw.Draw(paper)
    counts = Counter(key(r) for r in report['patches'])
    assert not counts['work_volume_collision'] and not counts['work_volume_unresolved']
    ink.text((35,25), f'{name} / {pose} / Step 2 / all {len(data.valid)} sampled centers', font=R.font(38), fill=R.INK)
    ink.text((40,87), f"Blue: {counts['valid']} accepted. Gray: {counts['other']} object/floor geometry failures. Hollow: behind the object.", font=R.font(25), fill=R.INK)
    cameras=[]
    for j,basis in enumerate(views):
        focus,width=R.overall_camera(domain,basis)
        pic,_=R.scene(domain,data,basis,focus,width,890,ground=j==0,show_patches=False,indices=[])
        visible,_=markers(pic,domain,data,report,basis,focus,width,890)
        paper.paste(pic,(35+940*j,155))
        detail,_=R.scene(domain,data,basis,focus,width,1600,ground=j==0,show_patches=False,indices=[])
        markers(detail,domain,data,report,basis,focus,width,1600,labels=True)
        detail.save(folder/f'centers_labeled_{j+1}.png')
        cameras.append(dict(basis=basis.tolist(),focus_m=focus.tolist(),width_m=float(width),
            visible_center_ids=[report['patches'][i]['id'] for i in np.flatnonzero(visible)]))
    ink.text((40,1080),'Green: work surface, excluded from contacts. All surface orientations allowed; process-access volume omitted.',font=R.font(24),fill=R.INK)
    ink.text((40,1120),'Object/floor clearance and the 90-degree wrap limit remain. Right: underside, floor omitted.',font=R.font(24),fill=R.INK)
    paper.save(folder/'centers.png')
    return None,None,dict(all_sampled_centers_displayed=True,status_counts=dict(counts),cameras=cameras,
        process_access_enforced=False,process_access_volume_drawn=False,
        labeled_images=['centers_labeled_1.png','centers_labeled_2.png'])
