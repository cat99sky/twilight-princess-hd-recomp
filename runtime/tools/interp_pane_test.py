#!/usr/bin/env python3
"""Linux regression for the production pane blender without generated game code.

Extract the pane state and blender verbatim, then exercise guest-memory writes
with the runtime's PPC accessors. Run with python3 runtime/tools/interp_pane_test.py.
Requires g++ and Linux mmap support.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
src = (root / 'runtime/src/interp.cpp').read_text()
struct = src[src.index('struct PanePrev {'):src.index('// a material animation', src.index('struct PanePrev {'))]
func = src[src.index('static uint32_t alpha_at('):src.index('// ---------------------------------------------------------------- the pass structure')]
helpers = src[src.index('void read_words('):src.index('// whether a model', src.index('void read_words('))]
# The Linux compatibility patch renames this helper to mix. Use the actual
# definition so the harness follows both its name and its production behavior.
blend_helper = re.search(r'^float (?:lerp|mix)\([^\n]+\{[^\n]+\}$', src, re.MULTILINE)
if blend_helper is None:
    raise RuntimeError('Could not locate the production lerp/mix helper')
harness = r'''
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include "ppc.h"
namespace mem { uint8_t* ptr(uint32_t addr) { return PPC_MEM_BASE + addr; } }
namespace interp {
bool g_on = true, g_hold = false;
int g_phase = 0, g_n = 3;
uint64_t g_records = 0;
BLEND_HELPER
void blend_pane(uint32_t, std::vector<std::pair<uint32_t, uint32_t>>&);
STRUCT
HELPERS
}
using namespace interp;
FUNCTION
using Saved = std::vector<std::pair<uint32_t,uint32_t>>;
constexpr uint32_t pane = 0x1000;
void pose(float x, uint8_t alpha) { stf32(pane+0x54,x); st8(pane+0xAA,alpha); }
void record(float x, uint8_t alpha) {
    pose(x, alpha);
    Saved saved;
    blend_pane(pane, saved);
    assert(saved.empty());
}
void check_reused(int holds) {
    g_n=holds;
    g_hold=false;
    record(10,20);
    record(50,90);
    record(90,180);
    g_hold=true;
    for (g_phase=1;g_phase<=g_n;g_phase++) {
        for (int i=0;i<3;i++) {
            float x=10+40*i;
            uint8_t alpha=20+70*i;
            pose(x,alpha);
            Saved saved;
            blend_pane(pane,saved);
            if (g_reused_pane_fix) {
                assert(saved.empty());
                assert(ldf32(pane+0x54)==x);
                assert(ld8(pane+0xAA)==alpha);
            } else {
                // Default behavior matches the original last-draw history.
                const float t=float(g_phase)/float(g_n+1);
                assert(ldf32(pane+0x54)==90+(x-90)*t);
                assert(ld8(pane+0xAA)==std::lround(180+(alpha-180)*t));
            }
            for(auto it=saved.rbegin();it!=saved.rend();++it) st32(it->first,it->second);
            assert(ldf32(pane+0x54)==x && ld8(pane+0xAA)==alpha);
        }
    }
}
void check_single() {
    g_hold=false;
    record(10,40);
    g_hold=true;
    g_n=3;
    for(g_phase=1;g_phase<=3;g_phase++) {
        pose(50,200);
        Saved saved;
        blend_pane(pane,saved);
        assert(ldf32(pane+0x54)==10+10*g_phase);
        assert(ld8(pane+0xAA)==40+40*g_phase);
        for(auto it=saved.rbegin();it!=saved.rend();++it) st32(it->first,it->second);
        assert(ldf32(pane+0x54)==50 && ld8(pane+0xAA)==200);
    }
}
int main() {
    auto memory = mmap(mem::ptr(pane),4096,PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
    assert(memory != MAP_FAILED);
    // The first recorded generation is zero, so a new entry must not look reused.
    check_single();
    for(int holds: {1,3,7}) { ++g_records; check_reused(holds); }
    ++g_records; check_single(); // reuse expires at the next logic generation
    ++g_records; // stale history cannot blend
    pose(80,120);
    Saved saved;
    blend_pane(pane,saved);
    assert(saved.empty() && ldf32(pane+0x54)==80 && ld8(pane+0xAA)==120);
    g_on=false;
    record(120,180);
    assert(ldf32(pane+0x54)==120 && ld8(pane+0xAA)==180);
    munmap(memory,4096);
}
'''.replace('STRUCT',struct).replace('HELPERS',helpers).replace('FUNCTION',func).replace('BLEND_HELPER',blend_helper.group(0))
with tempfile.TemporaryDirectory() as td:
    cpp=Path(td)/'panes.cpp'
    cpp.write_text(harness)
    exe=Path(td)/'test'
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=undefined','-I'+str(root/'runtime/include'),str(cpp),'-o',str(exe)],check=True)
    env = os.environ.copy()
    env.pop('WWHD_REUSED_PANE_FIX', None)
    subprocess.run([str(exe)],check=True,env=env)
    env['WWHD_REUSED_PANE_FIX'] = '0'
    subprocess.run([str(exe)],check=True,env=env)
    env['WWHD_REUSED_PANE_FIX'] = '1'
    subprocess.run([str(exe)],check=True,env=env)
print('PASS: default/or explicitly disabled baseline and opt-in reused-pane handling, 60/120/240 draws, alpha, restoration, reuse reset, stale history, interpolation off')
