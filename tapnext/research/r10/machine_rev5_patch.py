"""Applies the rev-5 edits to MACHINE.md in place (run on testbox from npu-research/, now tapnext/research). One-shot, history: MACHINE.md is now common/MACHINE.md and
rev4/ is gone (git history)."""

from pathlib import Path

p = Path("../../common/MACHINE.md")
s = p.read_text()
assert "Revision 5" not in s, "already patched"


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


sub(
    "Revision 4, 2026-10-09 night: section 2.2",
    "Revision 5, 2026-10-10: section 3 corrects the round-9 reading of the A path. The k-step is not A-port-bound: its\n"
    "steady state is 2,055 + 67 cycles, and the ~4 B/cycle figures were a mean that includes the op-start wait, plus the\n"
    "DDR-bound slot load. Section 2.4 (perf counters and DMA events for P38) and section 2.5 (the L1 maps of the GEMM and\n"
    "EW cores) are new. Revision 4 is in `rev4/`.\n\n"
    "Revision 4, 2026-10-09 night: section 2.2",
)

sub(
    "| **memtile -> core (A broadcast to 3 cores)** | **>= 13.8 GB/s = 7.6 B/cycle per core.**",
    "| **memtile -> core (A broadcast to 3 cores)** | **rev 5: >= 5.5 B/cycle hard, ~7.6 fitted (idle core); >= 5.03 in situ under mm_ks (its steady "
    "2,055-cycle k-step needs it and gets it).** The round-9 \"~4 B/cycle\" was the mean k-step including the op-start wait (ANALYSIS rev 5 §1). "
    "Rev 2's reading follows: **>= 13.8 GB/s = 7.6 B/cycle per core.**",
)

sub(
    "| per shim stream (DDR -> array) | ~6.4 GB/s = **3.6 B/cycle** at 1.8 GHz | [MEAS moondream p2/bw.py] |",
    "| per shim stream (DDR -> array) | ~6.4 GB/s = **3.6 B/cycle** at 1.8 GHz | [MEAS moondream p2/bw.py] |\n"
    "| A-slot load, 8 columns at once (rev 5) | 8 x 248,832 B in ~33 us from mm_ks entry (~36 us from the previous op's last send): **55-60 GB/s in total = the DDR read peak, ~7 GB/s = 4.1 B/cycle per column**. The op-start wait is DDR-bound | [MEAS g11-r9sp first-tile records, tools/profdec.py] |\n"
    "| S2MM out-of-order mode | `DMA_S2MM_n_CTRL.Enable_Out_of_Order` (bit 3), `XAie_DmaSetOutofOrderBdId`: the packet header picks the BD, so one S2MM channel can land several packet streams in separate rings. FoT (finish-on-TLAST) mode and decompression also exist | [DOC aie-rt params / xaie_dma.h] |",
)

sub(
    "Take-away: inside the array, data movement is fast (>= 7.6 B/cycle per core port). The slow links are the shim\nchannels (~6.4 GB/s each) and, in total, DDR.",
    "Take-away: inside the array, data movement is fast (>= 5.5 B/cycle hard, ~7.6 fitted, per core port). The slow links are\n"
    "the shim channels (~6.4 GB/s each) and, in total, DDR. Rev 5: the op-start A-slot loads hit the DDR peak (8 columns at\n"
    "once).",
)

sub(
    "## 3. Streams and ports",
    "### 2.4 Counters and DMA events for in-situ stall accounting (rev 5, for P38)\n\n"
    "| item | value | tag |\n"
    "|---|---|---|\n"
    "| core-module counters | 4: `Performance_Control0/1` 0x31500 / 0x31504 (start / stop event, 7 bits each per counter), `Control2` 0x31508 (reset), `Counter0-3` 0x31520-0x3152C | [DOC PARAMS] |\n"
    "| memory-module counters | 2: `Performance_Control0` 0x11000, `Control1` 0x11008, `Counter0/1` 0x11020 / 0x11024 | [DOC PARAMS] |\n"
    "| core stall events | MEMORY_STALL 23, STREAM_STALL 24, LOCK_STALL 26, ACTIVE 28 | [DOC EVENTS] |\n"
    "| memory-module DMA events | S2MM_0/1 STALLED_LOCK 31 / 32, STREAM_STARVATION 35 / 36, MEMORY_BACKPRESSURE 39 / 40; MM2S_0/1 STALLED_LOCK 33 / 34, STREAM_BACKPRESSURE 37 / 38, MEMORY_STARVATION 41 / 42; S2MM / MM2S FINISHED_BD 23-26 | [DOC EVENTS `XAIE2P_EVENTS_MEM_*`] |\n"
    "| counting an event's cycles | start event = stop event = the event (the AIE convention) [INF]: P38 self-checks it with a TRUE counter against the timer | |\n"
    "| data memory width | 256 bit (`MEMORY_MODULE_DATAMEMORY_DATA_WIDTH`); core load/store bus 256 bit, 512-bit vectors need 64 B alignment | [DOC PARAMS, TM] |\n\n"
    "### 2.5 L1 maps of the deployed cores (full12r9s, rev 5)\n\n"
    "| core | layout | free |\n"
    "|---|---|---|\n"
    "| GEMM (rows 2-4) | stack 0xB00; data 0x70B00 + 28,416 B (acc_hi 24 KB, tables, records); A ring 3 x 1,152 (banks 1, 2, 2); C object 18 KB (bank 2); B fifo 2 x 4,608 (bank 3); .aie.bank3 2 KB (GELU cd tables) | **~0.9 KB** (data 100 B, bank 1 192 B, bank 3 640 B). Text 15,840 / 16,384 |\n"
    "| EW (row 5) | stack 6,144 (max call path 4,224: norm_rows<LN> 2,944 + ln_stats 1,024 + 256); ewo0 9 KB (bank 0); data: ssig_ab 2.5 KB, g_arena 22,016, g_panels 1 KB, g_ptmp 1 KB; .aie.bank0 ssqrt_ab 1 KB, .aie.bank2 ssig_cd 2.5 KB, .aie.bank3 ssqrt_cd 1 KB; ewi0 15 KB (bank 3) | **bank-2 region tail 2,560 B + data tail 960 B; 1.9 KB of stack never used**. Text 16,240 / 16,384 (144 B) |\n\n"
    "## 3. Streams and ports",
)

sub(
    "3. Is aie2p's stream 64-bit?",
    "3. (rev 5) Still not on disk. In situ the A path keeps >= 5.03 B/cycle while mm_ks runs; idle it gave >= 5.5 (hard). P34\n"
    "   measures it with today's ring, P38 shows whether the core ever waits for it. Rev 4 text: Is aie2p's stream 64-bit?",
)

p.write_text(s)
print("MACHINE.md patched")
