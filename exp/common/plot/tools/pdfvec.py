#!/usr/bin/env python3
"""Minimal vector extractor for single-page matplotlib/cairo PDFs.

Usage: pdfvec.py file.pdf [out.json]
Emits JSON: {"page":[w,h], "paths":[...], "marks":[...], "texts":[...]}
  path : {"ops":[["m",x,y],["l",x,y],["c",x1,y1,x2,y2,x3,y3],["h"],["re",x,y,w,h]],
          "paint":"S"|"f"|"B"|..., "stroke":[r,g,b], "fill":[r,g,b], "lw":w,
          "dash":[..], "salpha":a, "falpha":a, "clip":[x0,y0,x1,y1]|null, "hatch":name|null}
  mark : {"name":"M0", "x":x, "y":y, "stroke":..., "fill":..., "lw":.., "salpha":..,"falpha":..}
  text : {"s":"...", "x":x, "y":y, "size":pt, "angle":deg, "fill":[r,g,b]}
All coordinates in page space (PDF points, origin bottom-left).
"""
import re, sys, zlib, json, math

def load_objects(data):
    objs = {}
    for m in re.finditer(rb'(\d+)\s+0\s+obj(.*?)endobj', data, re.S):
        objs[int(m.group(1))] = m.group(2)
    # object streams (PDF 1.5+)
    for num, body in list(objs.items()):
        if re.search(rb'/Type\s*/ObjStm', body[:300]):
            n = int(re.search(rb'/N\s+(\d+)', body).group(1))
            first = int(re.search(rb'/First\s+(\d+)', body).group(1))
            s = stream_of(body)
            hdr = [int(x) for x in s[:first].split()]
            offs = [(hdr[2*i], hdr[2*i+1]) for i in range(n)]
            for i, (onum, off) in enumerate(offs):
                end = offs[i+1][1] if i + 1 < n else len(s) - first
                objs.setdefault(onum, s[first+off:first+end])
    return objs

def stream_of(obj):
    m = re.search(rb'stream\r?\n(.*?)\r?\n?endstream', obj, re.S)
    if not m:
        return None
    raw = m.group(1)
    if b'/FlateDecode' in obj[:m.start()]:
        try:
            return zlib.decompressobj().decompress(raw)
        except Exception:
            return b''
    return raw

def ref(objs, s):
    m = re.match(rb'\s*(\d+)\s+0\s+R', s)
    return objs.get(int(m.group(1))) if m else None

def parse_dict_refs(s):
    """'<< /A 3 0 R /B 4 0 R >>' -> {A:3,B:4}"""
    return {k.decode(): int(v) for k, v in re.findall(rb'/([A-Za-z0-9_.+-]+)\s+(\d+)\s+0\s+R', s)}

def inline_or_ref_dict(objs, body, key):
    m = re.search(rb'/' + key + rb'\s+(\d+)\s+0\s+R', body)
    if m:
        return objs[int(m.group(1))]
    m = re.search(rb'/' + key + rb'\s*(<<.*?>>)', body, re.S)
    return m.group(1) if m else b''

def tounicode_map(cm):
    mp = {}
    if not cm:
        return mp
    t = cm.decode('latin1')
    for blk in re.findall(r'beginbfchar(.*?)endbfchar', t, re.S):
        for a, b in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            mp[int(a, 16)] = bytes.fromhex(b).decode('utf-16-be', 'replace')
    for blk in re.findall(r'beginbfrange(.*?)endbfrange', t, re.S):
        for a, b, rest in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]+>)', blk):
            lo, hi = int(a, 16), int(b, 16)
            if rest.startswith('['):
                vals = re.findall(r'<([0-9A-Fa-f]+)>', rest)
                for i, v in enumerate(vals):
                    mp[lo + i] = bytes.fromhex(v).decode('utf-16-be', 'replace')
            else:
                s = int(rest[1:-1], 16)
                for i, g in enumerate(range(lo, hi + 1)):
                    mp[g] = chr(s + i)
    return mp

TOK = re.compile(rb'\s*(?:(%[^\n]*)|(\[)|(\])|(<<)|(>>)|(<[0-9A-Fa-f\s]*>)|(\((?:\\.|[^\\)])*\))|(/[^\s/\[\]()<>]+)|([-+]?(?:\d+\.?\d*|\.\d+))|([A-Za-z\'"*]+[0-9]?\*?))', re.S)

def tokenize(b):
    pos = 0
    n = len(b)
    while pos < n:
        m = TOK.match(b, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        pos = m.end()
        if m.group(1):
            continue
        for i in range(2, 11):
            g = m.group(i)
            if g is not None:
                yield i, g
                break

def unescape_lit(s):
    s = s[1:-1]
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5c and i + 1 < len(s):
            nx = s[i + 1]
            mp = {ord('n'): 10, ord('r'): 13, ord('t'): 9, ord('b'): 8, ord('f'): 12}
            if nx in mp:
                out.append(mp[nx]); i += 2
            elif 0x30 <= nx <= 0x37:
                j = i + 1; v = 0
                while j < len(s) and j < i + 4 and 0x30 <= s[j] <= 0x37:
                    v = v * 8 + s[j] - 0x30; j += 1
                out.append(v & 255); i = j
            else:
                out.append(nx); i += 2
        else:
            out.append(c); i += 1
    return bytes(out)

def mul(a, b):
    # a then b  (row-vector convention: p' = p*a*b)
    return [a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3],
            a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3],
            a[4]*b[0] + a[5]*b[2] + b[4], a[4]*b[1] + a[5]*b[3] + b[5]]

def apply(m, x, y):
    return (x*m[0] + y*m[2] + m[4], x*m[1] + y*m[3] + m[5])

class Ctx:
    def __init__(self):
        self.ctm = [1, 0, 0, 1, 0, 0]
        self.stroke = [0, 0, 0]
        self.fill = [0, 0, 0]
        self.lw = 1.0
        self.dash = []
        self.salpha = 1.0
        self.falpha = 1.0
        self.clip = None
        self.fpattern = None
    def copy(self):
        c = Ctx(); c.__dict__.update({k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()})
        return c

def run(content, res, objs, out, ctx, fonts, depth=0):
    stack = []
    gs = []
    path = []
    pending_clip = False
    text = None
    xobjs = parse_dict_refs(inline_or_ref_dict(objs, res, b'XObject'))
    extg = {}
    eg = inline_or_ref_dict(objs, res, b'ExtGState')
    for name, body in re.findall(rb'/([A-Za-z0-9_]+)\s*<<(.*?)>>', eg, re.S):
        d = {}
        m = re.search(rb'/CA\s+([\d.]+)', body); d['CA'] = float(m.group(1)) if m else None
        m = re.search(rb'/ca\s+([\d.]+)', body); d['ca'] = float(m.group(1)) if m else None
        extg[name.decode()] = d
    for name, num in parse_dict_refs(eg).items():
        body = objs[num]
        d = {}
        m = re.search(rb'/CA\s+([\d.]+)', body); d['CA'] = float(m.group(1)) if m else None
        m = re.search(rb'/ca\s+([\d.]+)', body); d['ca'] = float(m.group(1)) if m else None
        extg[name] = d
    patterns = parse_dict_refs(inline_or_ref_dict(objs, res, b'Pattern'))

    def pt(x, y):
        return apply(ctx.ctm, x, y)

    def flush(paint):
        nonlocal path
        if path:
            out['paths'].append({'ops': path, 'paint': paint, 'stroke': ctx.stroke, 'fill': ctx.fill,
                                 'lw': ctx.lw * math.sqrt(abs(ctx.ctm[0]*ctx.ctm[3] - ctx.ctm[1]*ctx.ctm[2])),
                                 'dash': ctx.dash, 'salpha': ctx.salpha, 'falpha': ctx.falpha,
                                 'clip': ctx.clip, 'hatch': ctx.fpattern if paint in ('f', 'f*', 'B', 'B*', 'b', 'b*') else None})
        path = []

    for kind, tok in tokenize(content):
        if kind == 9:
            stack.append(float(tok)); continue
        if kind in (6, 7, 8, 2, 3, 4, 5):
            stack.append((kind, tok)); continue
        op = tok.decode('latin1')
        a = stack
        try:
            if op == 'q':
                gs.append(ctx.copy())
            elif op == 'Q':
                if gs:
                    ctx.__dict__.update(gs.pop().__dict__)
            elif op == 'cm':
                ctx.ctm = mul(a[-6:], ctx.ctm)
            elif op == 'w':
                ctx.lw = a[-1]
            elif op == 'd':
                arr = [x for x in a if isinstance(x, float)]
                ctx.dash = arr[:-1]
            elif op == 'RG':
                ctx.stroke = a[-3:]
            elif op == 'rg':
                ctx.fill = a[-3:]; ctx.fpattern = None
            elif op == 'G':
                ctx.stroke = [a[-1]] * 3
            elif op == 'g':
                ctx.fill = [a[-1]] * 3; ctx.fpattern = None
            elif op in ('SCN', 'scn', 'SC', 'sc'):
                nums = [x for x in a if isinstance(x, float)]
                names = [x[1].decode()[1:] for x in a if isinstance(x, tuple) and x[0] == 8]
                if op.lower() == op:  # fill
                    if names:
                        ctx.fpattern = names[-1]
                        if nums: ctx.fill = nums[-3:]
                    elif len(nums) >= 3:
                        ctx.fill = nums[-3:]
                else:
                    if len(nums) >= 3: ctx.stroke = nums[-3:]
            elif op in ('CS', 'cs'):
                pass
            elif op == 'gs':
                nm = a[-1][1].decode()[1:]
                d = extg.get(nm, {})
                if d.get('CA') is not None: ctx.salpha = d['CA']
                if d.get('ca') is not None: ctx.falpha = d['ca']
            elif op == 'm':
                path.append(['m', *pt(a[-2], a[-1])])
            elif op == 'l':
                path.append(['l', *pt(a[-2], a[-1])])
            elif op == 'c':
                p = a[-6:]
                path.append(['c', *pt(p[0], p[1]), *pt(p[2], p[3]), *pt(p[4], p[5])])
            elif op in ('v', 'y'):
                p = a[-4:]
                path.append(['c', *pt(p[0], p[1]), *pt(p[0], p[1]), *pt(p[2], p[3])])
            elif op == 'h':
                path.append(['h'])
            elif op == 're':
                x, y, w, h = a[-4:]
                p0 = pt(x, y); p1 = pt(x + w, y + h)
                path.append(['re', min(p0[0], p1[0]), min(p0[1], p1[1]), abs(p1[0]-p0[0]), abs(p1[1]-p0[1])])
            elif op in ('W', 'W*'):
                pending_clip = True
            elif op in ('S', 's', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*', 'n'):
                if pending_clip:
                    xs = []; ys = []
                    for o in path:
                        if o[0] == 're':
                            xs += [o[1], o[1]+o[3]]; ys += [o[2], o[2]+o[4]]
                        elif o[0] != 'h':
                            xs += o[1::2]; ys += o[2::2]
                    if xs:
                        ctx.clip = [min(xs), min(ys), max(xs), max(ys)]
                    pending_clip = False
                if op == 'n':
                    path = []
                else:
                    flush(op)
            elif op == 'Do':
                nm = a[-1][1].decode()[1:]
                num = xobjs.get(nm)
                if num is not None:
                    xo = objs[num]
                    if b'/Image' in xo[:300]:
                        out['images'].append({'name': nm, 'ctm': ctx.ctm})
                    else:
                        x, y = apply(ctx.ctm, 0, 0)
                        sub = {'paths': [], 'marks': [], 'texts': [], 'images': []}
                        c2 = ctx.copy(); c2.clip = None
                        m = re.search(rb'/Matrix\s*\[([^\]]*)\]', xo[:400])
                        if m:
                            c2.ctm = mul([float(v) for v in m.group(1).split()], c2.ctm)
                        xres = inline_or_ref_dict(objs, xo, b'Resources') or res
                        run(stream_of(xo) or b'', xres, objs, sub, c2, fonts, depth + 1)
                        if depth == 0 and len(sub['paths']) <= 4 and not sub['texts']:
                            out['marks'].append({'name': nm, 'x': x, 'y': y, 'stroke': ctx.stroke, 'fill': ctx.fill,
                                                 'lw': ctx.lw, 'salpha': ctx.salpha, 'falpha': ctx.falpha,
                                                 'clip': ctx.clip, 'shape': sub['paths']})
                        else:
                            for k in ('paths', 'marks', 'texts', 'images'):
                                out[k] += sub[k]
            elif op == 'BT':
                text = {'tm': [1, 0, 0, 1, 0, 0], 'lm': [1, 0, 0, 1, 0, 0], 'font': None, 'size': 1, 'lead': 0}
            elif op == 'ET':
                text = None
            elif op == 'Tf':
                text['font'] = a[-2][1].decode()[1:]; text['size'] = a[-1]
            elif op == 'Tm':
                text['tm'] = a[-6:]; text['lm'] = a[-6:]
            elif op in ('Td', 'TD'):
                text['lm'] = mul([1, 0, 0, 1, a[-2], a[-1]], text['lm']); text['tm'] = text['lm']
            elif op == 'T*':
                text['lm'] = mul([1, 0, 0, 1, 0, -text['lead']], text['lm']); text['tm'] = text['lm']
            elif op == 'TL':
                text['lead'] = a[-1]
            elif op in ('Tj', 'TJ', "'", '"'):
                parts = []
                for it in a:
                    if isinstance(it, tuple) and it[0] in (6, 7):
                        parts.append(it)
                s = ''
                fobj = fonts.get(text['font'], {})
                for k, raw in parts:
                    if k == 6:
                        hx = re.sub(rb'\s', b'', raw[1:-1])
                        bts = bytes.fromhex(hx.decode())
                    else:
                        bts = unescape_lit(raw)
                    if fobj.get('cid'):
                        for i in range(0, len(bts) - 1, 2):
                            g = bts[i] * 256 + bts[i + 1]
                            s += fobj['map'].get(g, '?')
                    else:
                        for bb in bts:
                            s += fobj.get('map', {}).get(bb, chr(bb))
                m = mul(text['tm'], ctx.ctm)
                x, y = m[4], m[5]
                size = text['size'] * math.hypot(m[0], m[1])
                ang = math.degrees(math.atan2(m[1], m[0]))
                if s.strip():
                    out['texts'].append({'s': s, 'x': x, 'y': y, 'size': round(size, 3), 'angle': round(ang, 1),
                                         'fill': ctx.fill, 'font': text['font']})
        except Exception as e:
            pass
        stack = []

def fonts_of(objs, res):
    fonts = {}
    fd = inline_or_ref_dict(objs, res, b'Font')
    for name, num in parse_dict_refs(fd).items():
        f = objs[num]
        info = {}
        if b'/Type0' in f:
            info['cid'] = True
        tu = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', f)
        if tu:
            info['map'] = tounicode_map(stream_of(objs[int(tu.group(1))]))
        else:
            # Type3 / simple fonts: try /Differences + glyph names
            mp = {}
            enc = re.search(rb'/Differences\s*\[(.*?)\]', f, re.S)
            if not enc:
                er = re.search(rb'/Encoding\s+(\d+)\s+0\s+R', f)
                if er:
                    enc = re.search(rb'/Differences\s*\[(.*?)\]', objs[int(er.group(1))], re.S)
            if enc:
                code = 0
                for t in re.findall(rb'(\d+)|/([^\s/\[\]]+)', enc.group(1)):
                    if t[0]:
                        code = int(t[0])
                    else:
                        mp[code] = GLYPH.get(t[1].decode(), t[1].decode()); code += 1
            info['map'] = mp
        fonts[name] = info
    return fonts

GLYPH = {'space': ' ', 'period': '.', 'comma': ',', 'hyphen': '-', 'minus': '−', 'plus': '+',
         'parenleft': '(', 'parenright': ')', 'slash': '/', 'colon': ':', 'at': '@', 'percent': '%',
         'equal': '=', 'underscore': '_', 'asterisk': '*', 'greater': '>', 'less': '<', 'quotesingle': "'",
         'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
         'eight': '8', 'nine': '9', 'bracketleft': '[', 'bracketright': ']', 'ampersand': '&', 'numbersign': '#',
         'rho': 'ρ', 'greaterequal': '≥', 'lessequal': '≤', 'multiply': '×', 'times': '×'}

def extract(fn):
    data = open(fn, 'rb').read()
    objs = load_objects(data)
    page = None
    for k, v in objs.items():
        if re.search(rb'/Type\s*/Page\b', v):
            page = v; break
    mb = [float(x) for x in re.search(rb'/MediaBox\s*\[([^\]]*)\]', page).group(1).split()]
    res = inline_or_ref_dict(objs, page, b'Resources')
    contents = re.search(rb'/Contents\s+(\d+)\s+0\s+R', page)
    content = stream_of(objs[int(contents.group(1))])
    fonts = fonts_of(objs, res)
    out = {'page': [mb[2] - mb[0], mb[3] - mb[1]], 'paths': [], 'marks': [], 'texts': [], 'images': []}
    run(content, res, objs, out, Ctx(), fonts)
    return out

if __name__ == '__main__':
    o = extract(sys.argv[1])
    if len(sys.argv) > 2:
        json.dump(o, open(sys.argv[2], 'w'))
    print('page', o['page'], 'paths', len(o['paths']), 'marks', len(o['marks']), 'texts', len(o['texts']), 'images', len(o['images']))
    for t in o['texts'][:400]:
        print('T %-40r x=%.1f y=%.1f size=%.2f ang=%s' % (t['s'], t['x'], t['y'], t['size'], t['angle']))
