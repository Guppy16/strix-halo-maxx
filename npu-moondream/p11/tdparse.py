"""Build a latency table for AIE2P from llvm-aie's generated TableGen: mnemonic (+ operand syntax) -> itinerary ->
per-operand latencies. Output: JSON {instr def name: {mnemonic, args, itin, outs, ins, lat: {opname: cycles}}}.
Usage: tdparse.py td_dir out.json"""
import json, re, sys
td = sys.argv[1]
info = open(f"{td}/AIE2PGenInstrInfo.td").read()
sched = open(f"{td}/AIE2PGenSchedule.td").read()
# itineraries: InstrItinData<II_X, [stages], [/*op*/lat, ...] ...>
itin = {}
for m in re.finditer(r"(?:Mem)?InstrItinData<\s*(II_\w+)\s*,\s*\[(.*?)\]\s*,\s*\[(.*?)\]", sched, re.S):
    name, stages, lats = m.group(1), m.group(2), m.group(3)
    ops = re.findall(r"/\*(\w+)\*/\s*(-?\d+)", lats)
    itin[name] = {"lat": {k: int(v) for k, v in ops}, "stages": stages.strip()}
# instructions: optional "let Itinerary = II_X[, ...] in" then "def NAME : CLASS <(outs ...), (ins ...), "mn", "args">"
defs = {}
pat = re.compile(r'(?:let\s+Itinerary\s*=\s*(II_\w+)[^\n]*\bin\s*\n)?def\s+(\w+)\s*:\s*(\w+)\s*<\s*\(outs([^)]*)\)\s*,\s*\(ins([^)]*)\)\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"', re.S)
for m in pat.finditer(info):
    ii, name, cls, outs, ins, mn, args = m.groups()
    opn = lambda s: [x.split(":")[-1].strip().lstrip("$") for x in s.split(",") if x.strip()]
    defs[name] = {"mn": mn, "args": args.strip(), "cls": cls, "outs": opn(outs), "ins": opn(ins), "itin": ii,
                  "lat": itin.get(ii, {}).get("lat", {}) if ii else {}}
json.dump({"defs": defs, "itin": itin}, open(sys.argv[2], "w"))
print(f"{len(defs)} instruction defs, {sum(1 for d in defs.values() if d['itin'])} with itineraries, {len(itin)} itineraries")
want = ["vmul", "vmac", "vmac.f", "vmul.f", "vconv.bf16.fp32", "vconv.fp32.bf16", "vldb.pop.576", "vlda", "vst", "vadd.f", "vsub.f", "vmax_lt.bf16", "vshuffle", "vsrs.2x", "vups.2x", "vst.push.576.conv.bfp16ebs8.fp32", "vlda.conv.fp32.bf16", "vldb", "mova", "movxm", "vbcst.16", "vmov", "vadd.16", "vsub.16", "vshift", "vbcst.32", "vextbcst.16", "vmax.bf16", "vlt.bf16", "vadd", "vsub", "vmul.cm", "lda", "st", "add", "lshl", "padda", "movs", "mov", "sel.nez", "eq", "jnz", "jz", "jl", "ret"]
for w in want:
    rows = [(n, d) for n, d in defs.items() if d["mn"] == w]
    if not rows: continue
    lat = {}
    for n, d in rows:
        for k, v in d["lat"].items(): lat.setdefault(k, set()).add(v)
    print(f"{w:36s} {len(rows):3d} defs  lat {{{', '.join(f'{k}:{sorted(v)}' for k, v in list(lat.items())[:6])}}}")
