"""Force/moment projections and four-configuration figure layout."""
import matplotlib.colors
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import patch as P
ORANGE, NEED = P.ORANGE, P.NEED


def page(out, panels, records):
    assert len(panels) == len(records) == 4
    width = max(im.width for im in panels)
    sidebar, separation = 570, 30
    im = Image.new('RGB', (width+2*P.GAP+separation+sidebar,
                          P.GAP+sum(panel.height+P.GAP for panel in panels)), P.PAPER)
    draw = ImageDraw.Draw(im)
    regular = font_manager.findfont('DejaVu Sans')
    heading = ImageFont.truetype(regular, 36)
    count_font = ImageFont.truetype(regular, 52)
    x, y = P.GAP+width+separation+sidebar/2, P.GAP
    for panel, row in zip(panels, records):
        im.paste(panel, (P.GAP, y))
        mid = y+panel.height/2
        fraction = row['coverage_fraction']
        if fraction['method'] == 'continuous_domain_certificate':
            assert row['continuous_coverage']['status'] == 'certified'
            text = '100%'
        else:
            assert fraction['method'] == 'deterministic_polygon_cubature'
            value = fraction['estimate']*100
            places = 3 if 0 < value < .1 else 2
            text = f'{value:.{places}f}%'
        draw.text((x, mid-60), 'Covered', font=heading, fill='#6b6b66', anchor='mm')
        draw.text((x, mid+10), text, font=count_font, fill='#1b1b1a', anchor='mm')
        y += panel.height+P.GAP
    im.save(out)
    print('wrote', out.name, im.size, flush=True)


def balls(S, paint_m, ok_f, ok_m, few):
    """Paint each sphere by its own three-row equilibrium checks.

    Red wins a shared direction bin if any sample in that bin fails.
    The relief contains only the actual tested moment bins.
    """
    ff, n_pf, n_bf = P.demand_sheet(S.tiles, S.tree, S.A_t, S.uf, ok_f, drawn_at=-1)
    fm, n_pm, n_bm = P.demand_sheet(S.tiles, S.tree, S.A_t, S.um, ok_m, drawn_at=+1)
    want_f, cov_f = np.r_[ff[0][0], ff[1][0]], ff[0][0]
    quills = [(S.uf[few][~ok_f[few]], NEED, False),
              (S.uf[few][ok_f[few]], ORANGE, False)]
    force = P.globe_png(S.ico, [([(want_f, NEED, .95), (cov_f, ORANGE, .95)], [],
                                 quills, [(-P.UP, P.FLOOR_MARK, "")], "")],
                        px=P.BALL, reach=1.60, triad=False)
    rgba = np.tile(matplotlib.colors.to_rgba(NEED), (len(S.tiles), 1))
    rgba[fm[0][0]] = matplotlib.colors.to_rgba(ORANGE)
    h = np.where(paint_m, S.h, np.nan)          # only the actually tested bins
    drawn_m = np.isfinite(h)
    bad_m = np.zeros(len(S.tiles), dtype=bool)
    bad_m[fm[1][0]] = True
    n_pm, n_bm = int(drawn_m.sum()), int((drawn_m & bad_m).sum())
    moment = P.globe_png(S.ico, [([], [], [], [], "", (), (),
                                  (h, rgba, P.BARE, tuple(S.ring_r)))],
                         px=P.BALL, reach=1.60, triad=False, weight=False,
                         rings_front=True)
    im = Image.new("RGB", (force.size[0] + moment.size[0], force.size[1]), P.PAPER)
    im.paste(force, (0, 0))
    im.paste(moment, (force.size[0], 0))
    top = P.BALL - P.PX
    paper = np.array(Image.new("RGB", (1, 1), P.PAPER))[0, 0]
    assert (np.asarray(im)[:top] == paper).all(), "ink in the title band"
    return im.crop((0, top, im.size[0], im.size[1])), (n_pf, n_bf, n_pm, n_bm)
