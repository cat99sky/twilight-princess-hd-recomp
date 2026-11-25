#include "../platform/host.h"
#ifdef __APPLE__
#include <pthread/qos.h>
#endif
#include <array>
#include <bitset>
#include <condition_variable>
#include <deque>
#include <cstdlib>
// GX2 core: command execution, display lists, context states, draws, clears,
// copies and presentation.
#include <algorithm>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gx2.h"
#include "gx2_cmd.h"
#include "gx2_regs.h"
#include "gx2_texture_regs.h"
#ifdef WWHD_HAS_VULKAN
#include "shader_key_dirty.h"
#include "gfx/vulkan/api.h"
#endif
#include "runtime.h"
#include "../aspect.h"
#include "gfx/renderer.h"
#include "platform/perf_hint.h"
#include "render_prof.h"

using namespace Latte;

namespace gx2 {

// ---------------------------------------------------------------- register file and context states
static uint32 g_regs[kNumRegs];
static uint32* g_shadow = nullptr;  // register copy of the active GX2ContextState
static std::unordered_map<uint32, std::vector<uint32>> g_contexts;
static std::recursive_mutex g_exec_mutex;

uint32* regs() { return g_regs; }

// Register writes that can change how shaders are translated bump g_shader_state_gen; the
// renderer reuses its last shader lookup while it is unchanged. Uniforms, uniform/vertex buffer
// addresses and rewrites of an identical value don't count.
extern "C" { uint64_t g_shader_state_gen = 1; }

static bool shader_irrelevant(uint32 reg) {
#ifdef WWHD_HAS_VULKAN
    // Vulkan resolves texture addresses freshly in bind_stage. These words do
    // not participate in shader translation; keep Metal's broader dirty gate.
    static const bool addressMemo = [] {
        const char* e = getenv("WWHD_VK_SHADER_ADDRESS_MEMO");
        return render::vulkan() && e && !strcmp(e, "1");
    }();
    if (addressMemo)
        for (uint32 base : {uint32(REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS),
                            uint32(REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS),
                            uint32(REGADDR::SQ_TEX_RESOURCE_WORD0_N_GS)})
            if (reg >= base && reg < base + 7 * 18) {
                uint32 word = (reg - base) % 7;
                if (word == 2 || word == 3) return true;
            }
#endif
    if (reg >= mmSQ_ALU_CONSTANT0_0 && reg < mmSQ_ALU_CONSTANT0_0 + 0x1000) return true;
    for (uint32 base : {(uint32)mmSQ_VTX_UNIFORM_BLOCK_START, (uint32)mmSQ_PS_UNIFORM_BLOCK_START, (uint32)mmSQ_GS_UNIFORM_BLOCK_START})
        if (reg >= base && reg < base + 7 * 16) return true;
    if (reg >= mmSQ_VTX_ATTRIBUTE_BLOCK_START && reg < mmSQ_VTX_ATTRIBUTE_BLOCK_START + 7 * 16) {
        uint32 w = (reg - mmSQ_VTX_ATTRIBUTE_BLOCK_START) % 7;
        return w != 2;  // word 2 holds the stride
    }
    return false;
}
static ShaderKeyDirtyStats shaderKeyDirtyStats;
ShaderKeyDirtyStats shader_key_dirty_stats() { return shaderKeyDirtyStats; }

#ifdef WWHD_HAS_VULKAN
static void apply_small_regs(uint32 first, const uint32* v, uint32 n) {
    static const bool keyDirty = [] {
        const char* e = getenv("WWHD_VK_SHADER_KEY_DIRTY");
        return e && !strcmp(e, "1");
    }();
    static const bool collectStats = getenv("WWHD_VK_STATS") != nullptr;
    const bool classify = rprof::enabled();
    bool changed = false, baselineBump = false, actualBump = false;
    uint64_t maskedWords = 0;
    for (uint32 i = 0; i < n; ++i) {
        const uint32 reg = first + i, value = v[i], old = g_regs[reg];
        if (old != value) {
            changed = true;
            if (classify) {
                if (rprof::fast_class_reg(reg)) rprof::g_reg_dirty |= 1;
                else rprof::note_other_reg(reg);
            }
            if ((collectStats || !actualBump) && !shader_irrelevant(reg)) {
                baselineBump = true;
                uint32 mask;
                if (keyDirty && vulkan_shader_key_mask(reg, mask) && !((old ^ value) & mask)) {
                    if (collectStats) ++maskedWords;
                } else actualBump = true;
            }
            g_regs[reg] = value;
        }
        // Shadow can differ even when registers already match: draw mutates
        // primitive type directly, and context setup initializes only shadow.
        if (g_shadow) g_shadow[reg] = value;
    }
    if (actualBump) ++g_shader_state_gen;
    if (changed && collectStats) {
        ++shaderKeyDirtyStats.changedBatches;
        shaderKeyDirtyStats.baselineWouldBumps += baselineBump;
        shaderKeyDirtyStats.actualBumps += actualBump;
        shaderKeyDirtyStats.avoidedBumps += baselineBump && !actualBump;
        shaderKeyDirtyStats.maskedWords += maskedWords;
    }
}
#endif

// Commands execute under g_exec_mutex, including nested list playback and inline draws.
struct ListWord {
    uint32 list = 0, offset = 0, header = 0;
};
struct ShaderWrite {
    const char* source;
    ListWord word;
    uint32 ctx, first, count, reg, before, after;
};
static const bool g_trace_shader_writes = [] {
    const char* v = getenv("WWHD_TRACE_SHADER_WRITES");
    return v && !strcmp(v, "1");
}();
static std::array<ShaderWrite, 32> g_shader_writes{};
static uint64_t g_shader_write_count = 0;
static uint32 active_context() {
    for (const auto& [ctx, shadow] : g_contexts)
        if (shadow.data() == g_shadow) return ctx;
    return 0;
}
static void trace_shader_writes(uint32 first, const uint32* v, uint32 n,
                                const char* source, ListWord word) {
    if (!g_trace_shader_writes || first > mmSQ_PGM_START_VS || first + n <= mmSQ_PGM_START_PS) return;
    for (uint32 reg : {mmSQ_PGM_START_PS, mmSQ_PGM_START_VS}) {
        if (reg < first || reg - first >= n) continue;
        g_shader_writes[g_shader_write_count++ % g_shader_writes.size()] =
            {source, word, active_context(), first, n, reg, g_regs[reg], v[reg - first]};
    }
}
void dump_missing_shader_writes(uint64_t frame, uint64_t draw) {
    if (!g_trace_shader_writes) return;
    static uint64_t lastFrame = UINT64_MAX;
    if (lastFrame == frame) return;
    lastFrame = frame;
    const uint64_t entries = std::min<uint64_t>(g_shader_write_count, g_shader_writes.size());
    LOG("[gx2] shader-history frame=%llu draw=%llu ctx=%08X PS=%08X VS=%08X entries=%u",
        (unsigned long long)frame, (unsigned long long)draw, active_context(),
        g_regs[mmSQ_PGM_START_PS], g_regs[mmSQ_PGM_START_VS], unsigned(entries));
    for (uint64_t i = g_shader_write_count - entries; i < g_shader_write_count; ++i) {
        const auto& w = g_shader_writes[i % g_shader_writes.size()];
        LOG("[gx2] shader-write seq=%llu source=%s list=%08X offset=%X header=%08X ctx=%08X "
            "first=%04X count=%u reg=%04X %08X->%08X", (unsigned long long)i, w.source,
            w.word.list, w.word.offset, w.word.header, w.ctx, w.first, w.count, w.reg, w.before, w.after);
    }
}

bool stream_out_trace_enabled() {
    static const bool enabled = [] {
        const char* value = getenv("WWHD_TRACE_STREAM_OUT");
        return value && !strcmp(value, "1");
    }();
    return enabled;
}
// High bits of the runtime's recorded draw primitive are unused. Carry
// stream-out state there so enabling it never moves a guest patch offset.
static constexpr uint32 kStreamOutKnown = 1u << 31, kStreamOutEnabled = 1u << 30;
static constexpr uint32 kLegacyStreamOutEnable = 1u << 16;
static bool g_stream_out_active = false;

bool stream_out_active() { return g_stream_out_active; }

static void apply_stream_out_enable(bool enabled, uint32 lr, ListWord word) {
    g_stream_out_active = enabled;
    if (!stream_out_trace_enabled()) return;
    static unsigned reports = 0;
    if (reports < 32) {
        ++reports;
        LOG("[gx2] so-enable value=%u lr=%08X list=%08X packet=%X",
            unsigned(enabled), lr, word.list, word.offset);
    }
}

static void apply_regs(uint32 first, const uint32* v, uint32 n, const char* source, ListWord word) {
    if (first >= kNumRegs || n > kNumRegs - first) return;
    trace_shader_writes(first, v, n, source, word);
    if (first <= mmVGT_STRMOUT_EN && mmVGT_STRMOUT_EN - first < n)
        apply_stream_out_enable((v[mmVGT_STRMOUT_EN - first] & 1) != 0, 0, word);
#ifdef WWHD_HAS_VULKAN
    // Vulkan renderer only (the Metal renderer keeps the original bulk path)
    static const bool fusedSmall = [] {
        const char* e = getenv("WWHD_VK_FUSE_SMALL_REGS");
        // Enabled by default; explicit zero retains the original bulk path.
        return render::vulkan() && (!e || strcmp(e, "0") != 0);
    }();
    if (fusedSmall && n <= 16) {
        apply_small_regs(first, v, n);
        return;
    }
#endif
    if (memcmp(&g_regs[first], v, n * 4) != 0) {
        if (rprof::enabled())  // draw classifier (render_prof.h)
            for (uint32 i = 0; i < n; i++)
                if (g_regs[first + i] != v[i]) {
                    if (rprof::fast_class_reg(first + i)) rprof::g_reg_dirty |= 1;
                    else rprof::note_other_reg(first + i);
                }
#ifdef WWHD_HAS_VULKAN
        static const bool keyDirty = [] {
            const char* value = getenv("WWHD_VK_SHADER_KEY_DIRTY");
            return render::vulkan() && value && !strcmp(value, "1");
        }();
        static const bool collectStats = render::vulkan() && getenv("WWHD_VK_STATS") != nullptr;
        if(keyDirty || collectStats) {
            bool baselineBump = false, actualBump = false;
            uint64_t maskedWords = 0;
            for(uint32 i = 0; i < n; ++i) {
                uint32 reg = first + i;
                if(g_regs[reg] == v[i] || shader_irrelevant(reg)) continue;
                baselineBump = true;
                uint32 mask;
                if(keyDirty && vulkan_shader_key_mask(reg, mask) && !((g_regs[reg] ^ v[i]) & mask)) {
                    if(collectStats) ++maskedWords;
                } else actualBump = true;
                if(actualBump && !collectStats) break;
            }
            if(actualBump) ++g_shader_state_gen;
            if(collectStats) {
                ++shaderKeyDirtyStats.changedBatches;
                shaderKeyDirtyStats.baselineWouldBumps += baselineBump;
                shaderKeyDirtyStats.actualBumps += actualBump;
                shaderKeyDirtyStats.avoidedBumps += baselineBump && !actualBump;
                shaderKeyDirtyStats.maskedWords += maskedWords;
            }
        } else
#endif
        for (uint32 i = 0; i < n; i++)
            if (g_regs[first + i] != v[i] && !shader_irrelevant(first + i)) { g_shader_state_gen++; break; }
        memcpy(&g_regs[first], v, n * 4);
    }
    if (g_shadow) memcpy(&g_shadow[first], v, n * 4);
}

// ---------------------------------------------------------------- display list recording
struct Recording {
    uint32 start = 0, pos = 0, end = 0, streamOutDrawFlags = 0;
};
static thread_local Recording t_rec;
static std::atomic<uint64_t> g_stat_draws{0};  // executed draws, reported with the frame counter

// WWHD_GX2_DL_WATCH is a hexadecimal guest-endian word. Sample matching buffers
// at the producer and the failing packet boundary independently. Copy and call
// hits are byte evidence only: the watched value could also occur in payload.
enum class DlStage { Copy, Call, DirectCall, Replay, Applied };
static void trace_dl_word(DlStage stage, uint32 addr, uint32 bytes, uint32 lr,
                          uint32 destination = 0, uint32 writeAt = 0,
                          std::optional<uint32> failingOffset = {}) {
    static const std::optional<uint32> watch = []() -> std::optional<uint32> {
        const char* value = getenv("WWHD_GX2_DL_WATCH");
        if (!value || !*value) return {};
        char* end;
        auto parsed = std::strtoull(value, &end, 16);
        if (*end || parsed > UINT32_MAX) return {};
        return static_cast<uint32>(parsed);
    }();
    if (!watch || !addr || bytes < 4 || uint64(addr) + bytes > (1ull << 32)) return;

    constexpr unsigned limit = 12;
    static std::atomic<unsigned> samples[5]{};
    auto& sampled = samples[static_cast<unsigned>(stage)];
    if (sampled.load(std::memory_order_relaxed) >= limit) return;
    uint32 offset = failingOffset.value_or(0);
    while (offset <= bytes - 4 && ld32(addr + offset) != *watch) {
        if (failingOffset) return;
        offset += 4;
    }
    if (offset > bytes - 4 || sampled.fetch_add(1, std::memory_order_relaxed) >= limit) return;

    // Capture the suspected packet plus the next header when it fits.
    char window[8 * 9 + 1]{};
    unsigned words = std::min(8u, (bytes - offset) / 4);
    for (unsigned i = 0; i < words; ++i)
        std::snprintf(window + 9 * i, 10, " %08X", ld32(addr + offset + 4 * i));
    static constexpr const char* names[] = {"copy", "call", "direct-call", "replay-error", "replay-applied"};
    LOG("[gx2] DL watch %s list=%08X bytes=%u lr=%08X record=%08X write=%08X "
        "first_be=%08X offset=%X words_be=%s",
        names[static_cast<unsigned>(stage)], addr, bytes, lr, destination, writeAt,
        ld32(addr), offset, window);
}

static void execute_one(Op op, const uint32* p, uint32 n, ListWord word = {});

// ---------------------------------------------------------------- render thread
// Like the real GPU, command execution runs asynchronously to the game: GX2 calls append to a
// queue that a render thread turns into Metal work. WWHD_NO_RENDER_THREAD=1 executes inline.
static const bool g_render_thread = getenv("WWHD_NO_RENDER_THREAD") == nullptr;
static std::mutex g_q_mutex;
static std::condition_variable g_q_cv, g_q_done_cv;
static std::vector<uint32> g_q_pending, g_q_work;
static bool g_q_waiting = false;
static uint64_t g_fence_issued = 0, g_fence_done = 0;

static void render_thread_main() {
    host::set_thread_name("GX2 render");
    host::boost_thread_priority();
    perf_hint::add_current_thread();
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_q_mutex);
            g_q_waiting = true;
            if (g_q_pending.empty()) {
                const uint64_t idle = rprof::enabled() ? rprof::now_ns() : 0;
                g_q_cv.wait(lk, [] { return !g_q_pending.empty(); });
                if (idle) rprof::add_idle(rprof::now_ns() - idle);
            }
            g_q_waiting = false;
            g_q_work.swap(g_q_pending);
        }
        {
            std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
            render::with_autorelease_pool([] { execute(g_q_work.data(), (uint32)g_q_work.size()); });
        }
        g_q_work.clear();
    }
}

static void enqueue(Op op, const uint32* payload, uint32 n) {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(render_thread_main).detach(); });
    std::lock_guard<std::mutex> lk(g_q_mutex);
    g_q_pending.push_back(op | (n << 8));
    g_q_pending.insert(g_q_pending.end(), payload, payload + n);
    if (g_q_waiting) g_q_cv.notify_one();
}

// block the game thread until the render thread has executed everything queued so far
// debug: WWHD_SYNC_STATS=1 logs, every 5 s, how often each caller waited for the render thread to
// catch up (render_sync) and for how long
enum SyncSite { kSyncShutdown, kSyncFlip, kSyncDrawDone, kSyncVsyncUncapped, kSyncVsyncFlip, kSyncSaveState, kSyncCopySurface, kSyncSites };
static void sync_stat(int site, std::chrono::steady_clock::duration waited) {
    static const bool on = getenv("WWHD_SYNC_STATS") != nullptr;
    if (!on) return;
    static std::mutex mu;
    static uint64_t count[kSyncSites], ns[kSyncSites];
    static auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu);
    count[site]++;
    ns[site] += std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count();
    const auto now = std::chrono::steady_clock::now();
    if (now - t0 < std::chrono::seconds(5)) return;
    const double secs = std::chrono::duration<double>(now - t0).count();
    static const char* names[kSyncSites] = {"shutdown", "flip", "DrawDone", "vsync-uncapped", "vsync-flip", "savestate", "CopySurface"};
    char buf[400];
    int k = snprintf(buf, sizeof buf, "[gx2] render_sync per second:");
    for (int i = 0; i < kSyncSites; i++)
        if (count[i]) k += snprintf(buf + k, sizeof buf - k, " %s %.1f x %.2f ms", names[i], count[i] / secs, ns[i] / 1e6 / count[i]);
    LOG("%s", buf);
    memset(count, 0, sizeof count); memset(ns, 0, sizeof ns);
    t0 = now;
}
static void render_sync(int site = kSyncShutdown) {
    if (!g_render_thread) return;
    const auto started = std::chrono::steady_clock::now();
    struct Done {
        int site;
        std::chrono::steady_clock::time_point t;
        ~Done() {
            const auto waited = std::chrono::steady_clock::now() - t;
            sync_stat(site, waited);
            if (rprof::enabled())
                rprof::add_sync(site == kSyncDrawDone ? rprof::kSyncDrawDone : site == kSyncCopySurface ? rprof::kSyncCopySurface
                                : site == kSyncFlip || site == kSyncVsyncFlip ? rprof::kSyncFlip : rprof::kSyncOther,
                                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count());
        }
    } done{site, started};
    uint64_t id;
    {
        std::lock_guard<std::mutex> lk(g_q_mutex);
        id = ++g_fence_issued;
    }
    uint32 w = (uint32)id;
    enqueue(OP_FENCE, &w, 1);
    std::unique_lock<std::mutex> lk(g_q_mutex);
    g_q_done_cv.wait(lk, [&] { return g_fence_done >= id; });
}

// Guest byte 0 for a legacy type-1 PM4 header is 0x40 (two 11-bit
// register indices). Use reserved bits to distinguish recorded host commands
// from raw type-0 headers without changing any guest offsets or capacities.
constexpr uint32 kRecordedOpTag = 0x60;
static_assert(OP_COUNT <= 32);
static bool g_legacy_dl_headers = false; // only needed for imported save states

void emit(Op op, const uint32* payload, uint32 n) {
    if (t_rec.start) {
        uint32 bytes = 4 * (n + 1);
        if (t_rec.pos > t_rec.end || bytes > t_rec.end - t_rec.pos) {  // subtraction cannot overflow
            LOG("[gx2] display list overflow at %08X", t_rec.start);
            return;
        }
        uint32* w = (uint32*)mem::ptr(t_rec.pos);
        w[0] = kRecordedOpTag | op | (n << 8);
        memcpy(w + 1, payload, n * 4);
        if (op == OP_DRAW || op == OP_DRAW_INDEXED)
            w[1] |= t_rec.streamOutDrawFlags;
        t_rec.pos += bytes;
        return;
    }
    if (g_render_thread) {
        enqueue(op, payload, n);
        return;
    }
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    host::with_autorelease_pool([&] { execute_one(op, payload, n); });
}

// host-only commands never go into display lists
static void emit_host(Op op, std::initializer_list<uint32> payload) {
    if (g_render_thread) {
        enqueue(op, payload.begin(), (uint32)payload.size());
        return;
    }
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    host::with_autorelease_pool([&] { execute_one(op, payload.begin(), (uint32)payload.size()); });
}

void set_stream_out_enable(bool enabled, uint32 lr) {
#ifdef WWHD_HAS_VULKAN
    if (!render::vulkan() && !stream_out_trace_enabled()) return;
#else
    if (!stream_out_trace_enabled()) return;
#endif
    if (t_rec.start) {
        t_rec.streamOutDrawFlags = kStreamOutKnown | (enabled ? kStreamOutEnabled : 0);
    } else {
        emit_host(OP_STREAM_OUT_TRACE, {kLegacyStreamOutEnable, uint32(enabled), lr});
    }
}

#ifdef WWHD_HAS_VULKAN
void checkpoint_vulkan_caches() {
    // Wait for queued work, then exclude further renderer mutations while the
    // SDL thread writes the final cache checkpoint during orderly shutdown.
    render_sync();
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    host::with_autorelease_pool([] {
        gfxvk::wait_idle();
        gfxvk::save_renderer_caches();
    });
}
#endif

void set_regs(uint32 first, const uint32* values, uint32 count) {
    if (!count) return;
    static thread_local std::vector<uint32> buf;
    buf.resize(count + 1);
    buf[0] = first;
    memcpy(&buf[1], values, count * 4);
    emit(OP_SET_REGS, buf.data(), count + 1);
}
void set_reg(uint32 reg, uint32 value) { emit(OP_SET_REGS, {reg, value}); }

constexpr uint32 kColorBufferWords = 0x9C / 4, kDepthBufferWords = 0xAC / 4, kSurfaceWords = 0x74 / 4;

static bool recorded_payload(Op op, uint32 n) {
    switch (op) {
    case OP_NOP: return n == 0;
    case OP_SET_REGS: return n >= 2 && n <= kNumRegs + 1;
    case OP_DRAW: return n == 4;
    case OP_DRAW_INDEXED: return n == 6;
    case OP_CLEAR_COLOR: return n == kColorBufferWords + 4;
    case OP_CLEAR_DEPTH: return n == kDepthBufferWords + 3;
    case OP_CLEAR_BUFFERS: return n == kColorBufferWords + kDepthBufferWords + 7;
    case OP_COPY_SURFACE: return n == 2 * kSurfaceWords + 4;
    case OP_COPY_TO_SCAN: return n == kColorBufferWords + 1;
    case OP_CALL: return n == 2;
    case OP_SET_CONTEXT: case OP_EXPAND_COLOR: case OP_EXPAND_DEPTH: return n == 1;
    case OP_INVALIDATE: return n == 3;
    case OP_PEEK_Z: return n >= 4 && n % 3 == 1;
    case OP_STREAM_OUT_TRACE: return n == 3;
    case OP_STREAM_OUT_CONTEXT: return n == 5;
    default: return false;
    }
}

// PM4 values are big-endian. Go through the same batched write as GX2 HLE
// so the active context shadow and shader invalidation observe the update.
static void pm4_regs(uint32 first, const uint32* payload, uint32 n, ListWord word) {
    static thread_local std::vector<uint32> values;
    values.resize(n);
    for (uint32 i = 0; i < n; ++i) values[i] = __builtin_bswap32(payload[i]);
    apply_regs(first, values.data(), n, "PM4", word);
}

static void stream_out_context(const uint32* words) {
    const uint32 control = words[0];
    const uint32 slot = (control >> 8) & 3;
    auto& cursor = g_regs[mmVGT_STRMOUT_BUFFER_OFFSET_0 + 4 * slot];
    const auto wordAddress = [](uint32 addr, uint32 high) {
        return !high && addr >= 0x10000 && !(addr & 3);
    };
    if ((control & 1) && wordAddress(words[1], words[2])) st32(words[1], cursor);
    switch ((control >> 1) & 3) {
    case 0: cursor = words[3]; break;
    case 2: if (wordAddress(words[3], words[4])) cursor = ld32(words[3]); break;
    default: break;
    }
}

static void report_skipped_pm4(unsigned kind, uint32 header, uint32 list, uint32 offset) {
    // Execution is serialized on g_exec_mutex. Report once per kind, including
    // predicated SET_CONTEXT_REG, to make missing packet semantics observable.
    static std::bitset<259> reported;
    if (!reported.test(kind)) {
        reported.set(kind);
        LOG("[gx2] skipped PM4 header=%08X list=%08X offset=%X", header, list, offset);
    }
}

static void execute_list(uint32 list, uint32 bytes) {
    if (!bytes) return;
    if (!list || ((list | bytes) & 3) || uint64(list) + bytes > (1ull << 32)) {
        LOG("[gx2] invalid display list range %08X + %u", list, bytes);
        return;
    }
    const uint32* words = (const uint32*)mem::ptr(list);
    const uint32 count = bytes / 4;
    for (uint32 i = 0; i < count;) {
        uint32 native = words[i], n = native >> 8, id = native & 0xff;
        uint32 left = count - i - 1;
        const uint32* payload = words + i + 1;
        bool tagged = (id & 0xe0) == kRecordedOpTag;
        Op op = Op(tagged ? id & 31 : id);
        if (tagged || (g_legacy_dl_headers && n <= left && recorded_payload(op, n))) {
            if (n > left || !recorded_payload(op, n)) {
                LOG("[gx2] invalid recorded command %08X list=%08X offset=%X", native, list, i * 4);
                return;
            }
            execute_one(op, payload, n, {list, i * 4, native});
            i += 1 + n;
            continue;
        }

        uint32 header = __builtin_bswap32(native);
        uint32 type = header >> 30;
        n = ((header >> 16) & 0x3fff) + 1; // types 0 and 3, excludes the header
        if (type == 1) n = 2;
        if (type == 2) n = 0;
        if (n > left) {
            trace_dl_word(DlStage::Replay, list, bytes, 0, 0, 0, i * 4);
            LOG("[gx2] truncated PM4 header=%08X list=%08X offset=%X need=%u left=%u",
                header, list, i * 4, n, left);
            return;
        }
        if (type == 0) {
            uint32 first = header & 0xffff;
            if (n <= kNumRegs - first) pm4_regs(first, payload, n, {list, i * 4, header});
            else report_skipped_pm4(256, header, list, i * 4);
        } else if (type == 1) {
            report_skipped_pm4(257, header, list, i * 4);
        } else if (type == 3) {
            uint32 opcode = (header >> 8) & 0xff;
            if (opcode == 0x69 && !(header & 1)) { // SET_CONTEXT_REG
                uint32 offset = __builtin_bswap32(payload[0]) & 0xffff;
                if (n >= 2 && offset < 0x400 && n - 1 <= 0x400 - offset) {
                    pm4_regs(0xa000 + offset, payload + 1, n - 1, {list, i * 4, header});
                    trace_dl_word(DlStage::Applied, list, bytes, 0, 0, 0, i * 4);
                } else {
                    report_skipped_pm4(258, header, list, i * 4);
                }
            } else if (opcode == 0x34 && n == 5 && !(header & 1)) { // STRMOUT_BUFFER_UPDATE
                uint32 words[5];
                for (unsigned j = 0; j < std::size(words); ++j) words[j] = __builtin_bswap32(payload[j]);
                stream_out_context(words);
            } else if (opcode != 0x10) { // NOP
                report_skipped_pm4(opcode, header, list, i * 4);
            }
        }
        i += 1 + n;
    }
}

void execute(const uint32* words, uint32 count) {
    for (uint32 i = 0; i < count;) {
        uint32 hdr = words[i], n = hdr >> 8;
        Op op = Op(hdr & 0xff);
        if (op >= OP_COUNT || n > count - i - 1) {
            LOG("[gx2] corrupt render queue command %08X offset=%X", hdr, i * 4);
            return;
        }
        execute_one(op, words + i + 1, n);
        i += 1 + n;
    }
}

static void set_context(uint32 ctx, ListWord word) {
    if (!ctx) {
        g_shadow = nullptr;
        return;
    }
    auto it = g_contexts.find(ctx);
    if (it == g_contexts.end()) {
        g_shadow = nullptr;
        return;
    }
    g_shadow = it->second.data();
    trace_shader_writes(0, g_shadow, kNumRegs, "context-restore", word);
    memcpy(g_regs, g_shadow, sizeof(g_regs));
    g_stream_out_active = (g_regs[mmVGT_STRMOUT_EN] & 1) != 0;
    g_shader_state_gen++;
    rprof::g_reg_dirty |= 2;  // draw classifier: a context load counts as a full state change
}

// struct copies carried in a command, placed back in guest memory for the renderer (commands run
// one at a time, so a couple of fixed slots suffice)
static uint32 unpack_struct(const uint32* words, uint32 count, int slot) {
    static uint32 scratch = 0;
    if (!scratch) scratch = mem::host_alloc(2 * 0x100, 0x40);
    uint32 addr = scratch + slot * 0x100;
    memcpy(mem::ptr(addr), words, count * 4);
    return addr;
}

// Vulkan: GX2DrawDone queues the work instead of waiting for an idle device (the default on every
// platform since 2026-10-07; WWHD_VK_LAZY_DRAW_DONE=0 restores the full wait). The Metal renderer
// reads large vertex buffers straight from guest memory, so it keeps the real GPU wait.
static bool lazy_draw_done() {
    static const bool on = [] {
        const char* e = getenv("WWHD_VK_LAZY_DRAW_DONE");
        return render::vulkan() && (!e || atoi(e) != 0);
    }();
    return on;
}

static void execute_op(Op op, const uint32* p, uint32 n, ListWord word);
// every op is counted and (sampled) timed for the render-thread profiler (render_prof.h); display-list
// calls are not timed themselves: their ops are
static void execute_one(Op op, const uint32* p, uint32 n, ListWord word) {
    rprof::Op kind;
    switch (op) {
    case OP_CALL: execute_op(op, p, n, word); return;
    case OP_SET_REGS: kind = rprof::kOpRegs; break;
    case OP_DRAW: case OP_DRAW_INDEXED: kind = rprof::kOpDraw; rprof::classify_draw(); break;
    case OP_CLEAR_COLOR: case OP_CLEAR_DEPTH: case OP_CLEAR_BUFFERS: kind = rprof::kOpClear; break;
    case OP_COPY_SURFACE: case OP_PEEK_Z: kind = rprof::kOpCopy; break;
    case OP_COPY_TO_SCAN: kind = rprof::kOpScan; break;
    case OP_INVALIDATE: kind = rprof::kOpInvalidate; break;
    case OP_FLUSH: kind = rprof::kOpFlush; break;
    case OP_DRAW_DONE: kind = rprof::kOpDrawDone; break;
    case OP_SWAP: kind = rprof::kOpSwap; break;
    default: kind = rprof::kOpOther; break;
    }
    const uint64_t started = rprof::op_begin(kind);
    execute_op(op, p, n, word);
    rprof::op_end(kind, started);
    if (op == OP_SWAP) rprof::frame_end(n >= 2 && p[1]);
}

static void execute_op(Op op, const uint32* p, uint32 n, ListWord word) {
    switch (op) {
    case OP_NOP: break;
    case OP_STREAM_OUT_CONTEXT: stream_out_context(p); break;
    case OP_STREAM_OUT_TRACE:
        // A saved list from the initial probe may still contain these packets.
        if (p[0] == kLegacyStreamOutEnable) apply_stream_out_enable(p[1] != 0, p[2], word);
        break;
    case OP_SET_REGS: apply_regs(p[0], p + 1, n - 1, word.list ? "recorded-HLE" : "HLE", word); break;
    case OP_DRAW:
    case OP_DRAW_INDEXED: {
        const bool indexed = op == OP_DRAW_INDEXED;
        const bool before = g_stream_out_active;
        if (p[0] & kStreamOutKnown)
            g_stream_out_active = (p[0] & kStreamOutEnabled) != 0;
        g_stat_draws.fetch_add(1, std::memory_order_relaxed);
        render::draw(g_regs, p[0] & ~(kStreamOutKnown | kStreamOutEnabled), p[1], indexed ? p[2] : 0,
                     indexed ? p[3] : 0, p[indexed ? 4 : 2], p[indexed ? 5 : 3]);
        g_stream_out_active = before;
        break;
    }
    case OP_CLEAR_COLOR: {
        const uint32* q = p + kColorBufferWords;
        float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])};
        render::clear_color(g_regs, unpack_struct(p, kColorBufferWords, 0), rgba);
        break;
    }
    case OP_CLEAR_DEPTH: {
        const uint32* q = p + kDepthBufferWords;
        render::clear_depth_stencil(g_regs, unpack_struct(p, kDepthBufferWords, 0), bitsf(q[0]), q[1], q[2]);
        break;
    }
    case OP_CLEAR_BUFFERS: {
        uint32 cb = unpack_struct(p, kColorBufferWords, 0), db = unpack_struct(p + kColorBufferWords, kDepthBufferWords, 1);
        const uint32* q = p + kColorBufferWords + kDepthBufferWords;
        float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])};
        render::clear_color(g_regs, cb, rgba);
        render::clear_depth_stencil(g_regs, db, bitsf(q[4]), q[5], q[6]);
        break;
    }
    case OP_COPY_SURFACE: {
        // debug: WWHD_GX2_DELAY_COPY=ms stalls the render thread before each surface copy (a slow
        // or busy render thread; reproduced the agl boot crash every time before GX2CopySurface waited)
        static const int delay = getenv("WWHD_GX2_DELAY_COPY") ? atoi(getenv("WWHD_GX2_DELAY_COPY")) : 0;
        if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        uint32 src = unpack_struct(p, kSurfaceWords, 0);
        const uint32* q = p + kSurfaceWords;
        uint32 dst = unpack_struct(q + 2, kSurfaceWords, 1);
        const uint32* r = q + 2 + kSurfaceWords;
        render::copy_surface(src, q[0], q[1], dst, r[0], r[1]);
        break;
    }
    case OP_COPY_TO_SCAN: render::copy_to_scan(unpack_struct(p, kColorBufferWords, 0), p[kColorBufferWords]); break;
    case OP_CALL: execute_list(p[0], p[1]); break;
    case OP_SET_CONTEXT: set_context(p[0], word); break;
    case OP_INVALIDATE: render::invalidate(p[0], p[1], p[2]); break;
    case OP_PEEK_Z: render::peek_z(p, n); break;
    case OP_EXPAND_COLOR: case OP_EXPAND_DEPTH: break;  // MSAA/HiZ decompression: nothing to do on the host
    case OP_FLUSH: render::guest_flush(); break;  // Vulkan: asynchronous submission
    case OP_DRAW_DONE:
        // The Vulkan renderer writes GPU results back to guest memory only for linear surfaces (guest data is copied
        // into fenced upload slices when work is recorded), so GX2DrawDone needs this op executed
        // (render_sync in the HLE), not an idle GPU. Lazy DrawDone (lazy_draw_done(), the default)
        // queues the work instead of waiting for the whole device every frame. A payload word of 1
        // (save states) always waits for the idle GPU.
        if (lazy_draw_done() && !(n && p[0])) render::guest_flush();
        else render::wait_idle();
        // except what the CPU reads back: linear render targets (the Picto Box picture, issue #53)
        render::write_back();
        break;
    case OP_SWAP:
        if (n) render::set_frame_aspect(gx2::bitsf(p[0]));  // aspect ratio from the next frame on (aspect.cpp)
        render::swap();
        break;
    case OP_SETUP_CONTEXT:
        g_contexts[p[0]].assign(kNumRegs, 0);
        g_shadow = g_contexts[p[0]].data();
        break;
    case OP_FENCE: {
        std::lock_guard<std::mutex> lk(g_q_mutex);
        g_fence_done = std::max<uint64_t>(g_fence_done, p[0]);
        g_q_done_cv.notify_all();
        break;
    }
    default: break;
    }
}

// ---------------------------------------------------------------- default state
static void set_default_state() {
    // GX2SetShaderModeEx(UNIFORM_REGISTER, ...)
    LATTE_SQ_CONFIG sq;
    sq.set_DX9_CONSTS(true).set_ALU_INST_PREFER_VECTOR(true).set_PS_PRIO(3).set_VS_PRIO(2).set_GS_PRIO(1).set_ES_PRIO(0);
    set_reg(REGADDR::SQ_CONFIG, sq.getRawValue());
    set_reg(REGADDR::VGT_GS_MODE, 0);
    LATTE_PA_CL_VTE_CNTL vte{};
    vte.set_VPORT_X_OFFSET_ENA(true).set_VPORT_X_SCALE_ENA(true).set_VPORT_Y_OFFSET_ENA(true).set_VPORT_Y_SCALE_ENA(true);
    vte.set_VPORT_Z_OFFSET_ENA(true).set_VPORT_Z_SCALE_ENA(true).set_VTX_W0_FMT(true);
    set_reg(REGADDR::PA_CL_VTE_CNTL, vte.getRawValue());
    set_reg(REGADDR::DB_DEPTH_CONTROL, (1 << 1) | (1 << 2) | (1 << 4));  // z test + write, LESS
    LATTE_SX_ALPHA_TEST_CONTROL at;
    at.set_ALPHA_FUNC(LATTE_SX_ALPHA_TEST_CONTROL::E_ALPHA_FUNC::LESS).set_ALPHA_TEST_ENABLE(false);
    set_reg(REGADDR::SX_ALPHA_TEST_CONTROL, at.getRawValue());
    set_reg(REGADDR::SX_ALPHA_REF, 0);
    LATTE_PA_SU_SC_MODE_CNTL pm{};
    pm.set_FRONT_FACE(LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CCW);
    pm.set_FRONT_POLY_MODE(LATTE_PA_SU_SC_MODE_CNTL::E_PTYPE::TRIANGLES).set_BACK_POLY_MODE(LATTE_PA_SU_SC_MODE_CNTL::E_PTYPE::TRIANGLES);
    set_reg(REGADDR::PA_SU_SC_MODE_CNTL, pm.getRawValue());
    set_reg(REGADDR::VGT_MULTI_PRIM_IB_RESET_INDX, 0xFFFFFFFF);
    set_reg(REGADDR::CB_TARGET_MASK, 0xFFFFFFFF);
    for (int i = 0; i < 4; i++) set_reg(REGADDR::CB_BLEND_RED + i, 0);
    set_reg(REGADDR::PA_SU_POINT_SIZE, LATTE_PA_SU_POINT_SIZE().set_WIDTH(8).set_HEIGHT(8).getRawValue());
    LATTE_CB_COLOR_CONTROL cc;
    cc.set_SPECIAL_OP(LATTE_CB_COLOR_CONTROL::E_SPECIALOP::NORMAL).set_ROP(LATTE_CB_COLOR_CONTROL::E_LOGICOP::COPY);
    set_reg(REGADDR::CB_COLOR_CONTROL, cc.getRawValue());
    LATTE_PA_CL_CLIP_CNTL clip{};
    clip.set_DX_LINEAR_ATTR_CLIP_ENA(true);
    set_reg(REGADDR::PA_CL_CLIP_CNTL, clip.getRawValue());
    set_reg(mmDB_DEPTH_CLEAR, fbits(1.0f));
}

}  // namespace gx2

using namespace gx2;

// ---------------------------------------------------------------- init / timing
// Display timing, modelled on the hardware: vsync ticks at 60 Hz on its own clock, and a requested
// flip executes on the first vsync that is at least `swap interval` vsyncs after the previous flip.
// Games pace themselves by waiting for vsync until their flips have executed.
// 120/240 fps frame interpolation needs 4/8 flips per logic step: the virtual vsync then ticks 2/4
// times per 59.94 Hz vsync (interp::vsync_rate()). The clock counts quarter vsyncs ("ticks"); a
// vsync at the current rate is a granule of 4/rate ticks, and every vsync count below is in ticks,
// rounded to granules, so at 30/60 fps (granule 4) the timing is the 59.94 Hz one exactly.
static uint64_t g_swap_count = 0, g_flip_count = 0;
namespace gx2 { uint64_t flips_presented() { return __atomic_load_n(&g_flip_count, __ATOMIC_RELAXED); } }  // live fps in the title
static uint32 g_swap_interval = 1;  // as set by the game (frame interpolation halves it)
namespace interp { uint32_t effective_swap_interval(uint32_t game); void step_done(); int vsync_rate(); }
static std::mutex g_flip_mutex;
static const auto g_vsync_epoch = std::chrono::steady_clock::now();
static constexpr std::chrono::nanoseconds kVsyncPeriod(16683333);  // 59.94 Hz
static constexpr uint64_t kTicksPerVsync = 4;
// a flip also waits for the GPU to finish that frame, as on hardware: the game reuses a frame's
// buffers once its flip has executed
struct PendingFlip { uint64_t vsync, swap; };  // (vsync: tick of the swap)
static std::deque<PendingFlip> g_pending_flips;
static uint64_t g_last_flip_vsync = 0;  // tick (a granule boundary)
static uint64_t g_last_flip_time = 0;  // timebase
static int64_t g_count_offset = 0;     // guest-visible swap/flip counts minus ours (set by a loaded save state)

static uint64_t vsync_index() {  // in ticks
    return uint64_t((std::chrono::steady_clock::now() - g_vsync_epoch).count()) * kTicksPerVsync / uint64_t(kVsyncPeriod.count());
}
static uint64_t vsync_granule() {  // ticks per vsync at the current rate: 4 (30/60 fps), 2 (120), 1 (240)
    const int rate = interp::vsync_rate();
    return rate >= 4 ? 1 : rate >= 2 ? 2 : kTicksPerVsync;
}
static std::chrono::steady_clock::time_point tick_time(uint64_t tick) {
    return g_vsync_epoch + std::chrono::nanoseconds(tick * uint64_t(kVsyncPeriod.count()) / kTicksPerVsync);
}
// the first tick at which this flip may execute: the vsync after its swap, and `swap interval`
// vsyncs after the previous flip
static uint64_t flip_due(const PendingFlip& f) {
    const uint64_t g = vsync_granule();
    return std::max((f.vsync / g + 1) * g, g_last_flip_vsync + interp::effective_swap_interval(g_swap_interval) * g);
}
// Uncapped (gx2.h): debug, to see how fast the renderer can go; GPU completion ordering is kept
static std::atomic<int> g_uncapped{-1};  // -1: not read from the environment yet
bool gx2::uncapped() {
    int v = g_uncapped.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* any = getenv("WWHD_UNCAPPED");
        const char* vk = getenv("WWHD_VK_UNCAPPED");
        v = (any && !strcmp(any, "1")) || (vk && !strcmp(vk, "1") && render::vulkan()) ? 1 : 0;
        int expected = -1;
        g_uncapped.compare_exchange_strong(expected, v);
        v = g_uncapped.load(std::memory_order_relaxed);
    }
    return v == 1;
}
void gx2::set_uncapped(bool on) {
    if (uncapped() == on) return;
    g_uncapped = on ? 1 : 0;
    LOG("[gx2] uncapped %s", on ? "on (debug: no frame limit, no vsync; the game runs faster than real time)" : "off");
}
static std::atomic<bool> g_turbo{false};
bool gx2::turbo() { return g_turbo.load(std::memory_order_relaxed); }
void gx2::set_turbo(bool on) { g_turbo.store(on, std::memory_order_relaxed); }
static bool uncapped_benchmark() { return gx2::uncapped() || gx2::turbo(); }

static void update_flips() {  // g_flip_mutex held
    uint64_t now = vsync_index();
    while (!g_pending_flips.empty()) {
        uint64_t at = flip_due(g_pending_flips.front());
        if ((!uncapped_benchmark() && at > now) || render::frames_completed() < g_pending_flips.front().swap) break;
        at = now / vsync_granule() * vsync_granule();
        g_pending_flips.pop_front();
        g_last_flip_vsync = at;
        g_last_flip_time = timebase::now();
        g_flip_count++;
    }
}
#ifdef WWHD_HAS_VULKAN
static void ready_flip_before_resume() {
    bool needsSync;
    {
        std::lock_guard<std::mutex> lk(g_flip_mutex);
        if(g_pending_flips.empty()) return;
        const auto& front = g_pending_flips.front();
        const uint64_t at = flip_due(front);
        if(at > vsync_index()) return;
        needsSync = render::frames_completed() < front.swap;
    }
    if(needsSync) render_sync(kSyncFlip); // Core already released; no flip lock held.
    std::lock_guard<std::mutex> lk(g_flip_mutex);
    update_flips();
}
#endif

HLE(gx2, GX2Init) {
    set_default_state();
    LOG("[gx2] initialized (native GX2 -> %s)", render::api_name(render::active()));
    if(uncapped_benchmark())
        LOG("[gx2 benchmark] uncapped guest flips; GPU completion ordering retained; frame-based simulation accelerates while timebase/audio clocks remain real-time");
}

HLE(gx2, GX2SetupContextStateEx) {
    uint32 ctx = arg(c, 0);
    emit_host(OP_SETUP_CONTEXT, {ctx});
    set_default_state();
    // the context's "restore" display list lives inside the (0xA100 byte) context structure
    uint32 dl = ctx + 0x9800;
    uint32* w = (uint32*)mem::ptr(dl);
    w[0] = kRecordedOpTag | OP_SET_CONTEXT | (1u << 8);
    w[1] = ctx;
}
HLE(gx2, GX2SetContextState) { emit(OP_SET_CONTEXT, {arg(c, 0)}); }
HLE(gx2, GX2GetContextStateDisplayList) {
    if (arg(c, 1)) st32(arg(c, 1), arg(c, 0) + 0x9800);
    if (arg(c, 2)) st32(arg(c, 2), 8);
}

// ---------------------------------------------------------------- display lists
HLE(gx2, GX2BeginDisplayListEx) {
    t_rec = {};
    t_rec.start = t_rec.pos = arg(c, 0);
    t_rec.end = arg(c, 0) + arg(c, 1);
    static int traced = 0;
    if (getenv("WWHD_GX2_DL_TRACE") && traced++ < 10)
        LOG("[gx2] DL begin %08X (%u bytes)", t_rec.start, arg(c, 1));
}
HLE(gx2, GX2EndDisplayList) {
    uint32 size = t_rec.pos - t_rec.start;
    static int traced = 0;
    if (getenv("WWHD_GX2_DL_TRACE") && traced++ < 10)
        LOG("[gx2] DL end %08X: %u words, %u bytes", t_rec.start, size / 4, size);
    t_rec = Recording{};
    ret(c, size);
}
// GX2CopyDisplayList(list, bytes): append a recorded list into the list being recorded now
// A source may already mix PM4 with recorded ops. Keep guest patch offsets
// and bytes intact across nested copies. execute_list dispatches each header.
HLE(gx2, GX2CopyDisplayList) {
    uint32_t src = arg(c, 0), bytes = arg(c, 1);
    static int traced = 0;
    if (getenv("WWHD_GX2_DL_TRACE") && traced++ < 10)
        LOG("[gx2] DL copy %08X (%u bytes) -> %08X", src, bytes, t_rec.start);
    if (!bytes) return;
    if (!src || src + bytes < src) {  // null or 32-bit overflow
        LOG("[gx2] GX2CopyDisplayList: bad source %08X + %u", src, bytes);
        return;
    }
    if (!t_rec.start) {
        LOG("[gx2] GX2CopyDisplayList(%08X, %u) outside a display list", src, bytes);
        return;
    }
    if (t_rec.pos > t_rec.end || bytes > t_rec.end - t_rec.pos) {  // subtraction cannot overflow
        LOG("[gx2] display list overflow copying %u bytes at %08X", bytes, t_rec.pos);
        return;
    }
    trace_dl_word(DlStage::Copy, src, bytes, c->lr, t_rec.start, t_rec.pos);
    memmove(mem::ptr(t_rec.pos), mem::ptr(src), bytes);  // source may overlap the destination list
    t_rec.pos += bytes;
}
// The offset names the start of a shader register packet. Patch its first
// value in guest memory before the game copies or calls the display list.
HLE(gx2, GX2PatchDisplayList) {
    const uint32 list = arg(c, 0), type = arg(c, 1), offset = arg(c, 2), shader = arg(c, 3);
    if (!list || !shader || ((list | offset) & 3) || uint64_t(list) + offset + 12 > (1ull << 32)) return;
    uint32 program, size;
    switch (type) {
    case 1: program = ld32(shader + 0x0C); break; // fetch
    case 2: program = vertex_shader_program(shader, &size); break;
    case 3: // geometry copy shader
        program = ld32(shader + 0x58);
        if (!program) program = ld32(shader + 0xBC);
        break;
    case 4: // geometry shader
        program = ld32(shader + 0x50);
        if (!program) program = ld32(shader + 0xAC);
        break;
    case 5: program = pixel_shader_program(shader, &size); break;
    default: LOG("[gx2] unsupported display list shader patch type=%u", type); return;
    }
    const uint32 packet = list + offset;
    uint32 nativeHeader;
    memcpy(&nativeHeader, mem::ptr(packet), 4);
    const uint32 id = nativeHeader & 0xff, value = program >> 8;
    const bool recorded = id == (kRecordedOpTag | OP_SET_REGS) ||
                          (g_legacy_dl_headers && id == OP_SET_REGS && recorded_payload(OP_SET_REGS, nativeHeader >> 8));
    if (recorded) memcpy(mem::ptr(packet + 8), &value, 4);
    else st32(packet + 8, value);
    if (g_trace_shader_writes) {
        static std::atomic<unsigned> samples[6]{};
        if (samples[type].fetch_add(1, std::memory_order_relaxed) < 8)
            LOG("[gx2] shader-patch list=%08X offset=%X type=%u shader=%08X start=%08X encoding=%s",
                list, offset, type, shader, value, recorded ? "recorded-HLE" : "PM4");
    }
}
HLE(gx2, GX2GetCurrentDisplayList) {
    if (arg(c, 0)) st32(arg(c, 0), t_rec.start);
    if (arg(c, 1)) st32(arg(c, 1), t_rec.start ? t_rec.end - t_rec.start : 0);
    ret(c, t_rec.start != 0);
}
HLE(gx2, GX2CallDisplayList) {
    static int traced = 0;
    if (getenv("WWHD_GX2_DL_TRACE") && traced++ < 10)
        LOG("[gx2] DL call %08X (%u bytes)", arg(c, 0), arg(c, 1));
    trace_dl_word(DlStage::Call, arg(c, 0), arg(c, 1), c->lr, t_rec.start, t_rec.pos);
    emit(OP_CALL, {arg(c, 0), arg(c, 1)});
}
HLE(gx2, GX2DirectCallDisplayList) {
    trace_dl_word(DlStage::DirectCall, arg(c, 0), arg(c, 1), c->lr, t_rec.start, t_rec.pos);
    emit(OP_CALL, {arg(c, 0), arg(c, 1)});
}

// ---------------------------------------------------------------- draws
HLE(gx2, GX2DrawEx) { emit(OP_DRAW, {arg(c, 0), arg(c, 1), arg(c, 2), arg(c, 3)}); }
HLE(gx2, GX2DrawIndexedEx) { emit(OP_DRAW_INDEXED, {arg(c, 0), arg(c, 1), arg(c, 2), arg(c, 3), arg(c, 4), arg(c, 5)}); }

// ---------------------------------------------------------------- clears and copies
static float farg(Cpu* c, int i) { return (float)c->f[1 + i].ps0; }
// Struct parameters are copied into the command (as GX2 encodes them into the command buffer):
// games reuse one GX2ColorBuffer/GX2DepthBuffer and change its view between calls, also while
// recording display lists that run later.
static void put_struct(std::vector<uint32>& p, uint32 addr, uint32 words) {
    size_t at = p.size();
    p.resize(at + words);
    if (addr) memcpy(&p[at], mem::ptr(addr), words * 4);
}
HLE(gx2, GX2ClearColor) {
    std::vector<uint32> p;
    put_struct(p, arg(c, 0), kColorBufferWords);
    for (int i = 0; i < 4; i++) p.push_back(fbits(farg(c, i)));
    emit(OP_CLEAR_COLOR, p.data(), (uint32)p.size());
}
HLE(gx2, GX2ClearDepthStencilEx) {
    std::vector<uint32> p;
    put_struct(p, arg(c, 0), kDepthBufferWords);
    p.insert(p.end(), {fbits(farg(c, 0)), arg(c, 1) & 0xFF, arg(c, 2)});
    emit(OP_CLEAR_DEPTH, p.data(), (uint32)p.size());
}
HLE(gx2, GX2ClearBuffersEx) {
    std::vector<uint32> p;
    put_struct(p, arg(c, 0), kColorBufferWords);
    put_struct(p, arg(c, 1), kDepthBufferWords);
    p.insert(p.end(), {fbits(farg(c, 0)), fbits(farg(c, 1)), fbits(farg(c, 2)), fbits(farg(c, 3)), fbits(farg(c, 4)),
                       arg(c, 2) & 0xFF, arg(c, 3)});
    emit(OP_CLEAR_BUFFERS, p.data(), (uint32)p.size());
}
HLE(gx2, GX2SetClearDepthStencil) {
    auto* db = (GX2::GX2DepthBuffer*)mem::ptr(arg(c, 0));
    db->clearDepth = farg(c, 0);
    db->clearStencil = arg(c, 1) & 0xFF;
}
// (src, srcMip, srcSlice, dst, dstMip, dstSlice)
static void copy_surface(Cpu* c, uint32 src, uint32 srcMip, uint32 srcSlice, uint32 dst, uint32 dstMip, uint32 dstSlice) {
    // debug: WWHD_COPYDBG=1 logs each copy as issued (thread, caller, source and destination images)
    static const bool dbg = getenv("WWHD_COPYDBG") != nullptr;
    if (dbg)
        LOG("[copydbg] issue t=%.3f thread %08X lr %08X src %08X img %08X dst %08X img %08X size %X", timebase::now() / (double)timebase::kTicksPerSec,
            threads::current_thread(), c->lr, src, ld32(src + 0x24), dst, ld32(dst + 0x24), ld32(dst + 0x20));
    std::vector<uint32> p;
    put_struct(p, src, kSurfaceWords);
    p.insert(p.end(), {srcMip, srcSlice});
    put_struct(p, dst, kSurfaceWords);
    p.insert(p.end(), {dstMip, dstSlice});
    emit(OP_COPY_SURFACE, p.data(), (uint32)p.size());
    // The copy is complete when GX2CopySurface returns: the game uses the result (and frees the
    // surfaces) right away. agl's tile-mode conversion (027B5EEC) copies into a temporary surface,
    // OSBlockMoves it back and frees it at once; executed later on the render thread, the copy
    // wrote into the freed memory after the heap had reused it (boot crash: agl shader program
    // array 21EFE28C, program 0's +0x7c zeroed). Not for display lists (they run when called).
    if (!t_rec.start) {
        BlockingScope b;
        render_sync(kSyncCopySurface);
    }
}
HLE(gx2, GX2CopySurface) { copy_surface(c, arg(c, 0), arg(c, 1), arg(c, 2), arg(c, 3), arg(c, 4), arg(c, 5)); }
// (GX2DepthBuffer*, dst GX2Surface*, dstMip, dstSlice): a copy of the depth buffer's view, as Cemu.
// TP HD samples the result (R24_X8) in its water and other depth effects, and peeks it (dPeekZ).
static std::atomic<uint32> g_converted_depth{0};
uint32 gx2::converted_depth() { return g_converted_depth.load(std::memory_order_relaxed); }
HLE(gx2, GX2ConvertDepthBufferToTextureSurface) {
    uint32 db = arg(c, 0), dst = arg(c, 1);
    g_converted_depth = ld32(dst + 0x24);
    if (ld32(db + 0x24) == ld32(dst + 0x24)) return;  // same image: re-tiling in place, nothing to copy
    uint32 mip = ld32(db + 0x74), first = ld32(db + 0x78), n = std::max<uint32>(ld32(db + 0x7C), 1);
    for (uint32 i = 0; i < n; i++) copy_surface(c, db, mip, first + i, dst, arg(c, 2), arg(c, 3) + i);
}
// (src, srcMip, srcSlice, dst, dstMip, dstSlice, rectCount, GX2Rect* src rects; GX2Point* dst points on
// the stack): as Cemu, only a rect covering the whole source at (0, 0) is copied
HLE(gx2, GX2CopySurfaceEx) {
    uint32 src = arg(c, 0), rects = arg(c, 7), points = ld32(c->r[1] + 8);
    for (uint32 i = 0; i < arg(c, 6); i++) {
        uint32 r = rects + 16 * i, w = ld32(src + 4), h = ld32(src + 8);  // GX2Surface: dim, width, height
        if (ld32(r) == 0 && ld32(r + 4) == 0 && ld32(r + 8) == w && ld32(r + 12) == h && ld32(points + 8 * i) == 0 && ld32(points + 8 * i + 4) == 0)
            copy_surface(c, src, arg(c, 1), arg(c, 2), arg(c, 3), arg(c, 4), arg(c, 5));
        else
            LOG("[gx2] GX2CopySurfaceEx: partial rect %u,%u-%u,%u not copied (lr=%08X)", ld32(r), ld32(r + 4), ld32(r + 8), ld32(r + 12), c->lr);
    }
}
// (GX2ColorBuffer* cb, GX2Surface* dst, dstMip, dstSlice): resolve the colour buffer's (possibly
// multisampled) image into dst. GX2ColorBuffer starts with its GX2Surface, so the copy uses cb
// itself as the source surface. The game renders into the unresolved buffer and copies this
// resolved image to the scan buffer, so leaving it unimplemented produced a black TV picture.
HLE(gx2, GX2ResolveAAColorBuffer) {
    copy_surface(c, arg(c, 0), 0, 0, arg(c, 1), arg(c, 2), arg(c, 3));
}
HLE(gx2, GX2CopyColorBufferToScanBuffer) {
    std::vector<uint32> p;
    put_struct(p, arg(c, 0), kColorBufferWords);
    p.push_back(arg(c, 1));
    emit(OP_COPY_TO_SCAN, p.data(), (uint32)p.size());
}
HLE(gx2, GX2ExpandAAColorBuffer) { emit(OP_EXPAND_COLOR, {arg(c, 0)}); }
HLE(gx2, GX2ExpandDepthBuffer) { emit(OP_EXPAND_DEPTH, {arg(c, 0)}); }
HLE(gx2, GX2Invalidate) { emit(OP_INVALIDATE, {arg(c, 0), arg(c, 1), arg(c, 2)}); }

// ---------------------------------------------------------------- submission and presentation
HLE(gx2, GX2Flush) { emit_host(OP_FLUSH, {}); }
HLE(gx2, GX2DrawDone) {
    BlockingScope b;
    emit_host(OP_DRAW_DONE, {});
    render_sync(kSyncDrawDone);
    ret(c, 1);
}
HLE(gx2, GX2SwapScanBuffers) {
    // debug: WWHD_TRACE_SWAP=n logs the guest call chain of the first n swaps
    static int trace = getenv("WWHD_TRACE_SWAP") ? atoi(getenv("WWHD_TRACE_SWAP")) : 0;
    if (trace > 0) {
        trace--;
        char buf[256];
        int n = snprintf(buf, sizeof buf, "[gx2] swap from lr=%08X", c->lr);
        uint32_t sp = c->r[1];
        for (int i = 0; i < 8 && sp; i++) {
            uint32_t prev = ld32(sp);
            if (!prev || prev <= sp) break;
            n += snprintf(buf + n, sizeof buf - n, " <- %08X", ld32(prev + 4));
            sp = prev;
        }
        LOG("%s", buf);
    }
    float a = aspect::on_swap();  // aspect ratio of the next frame (game projections, render targets)
    uint32 ab;
    memcpy(&ab, &a, 4);
    interp::step_done();
    emit_host(OP_SWAP, {ab, 0u});  // 0: a new logic step (render-thread profiler; no hold passes at 30 fps)
    uint64_t front_due = 0;
    bool wait_for_slot = false;
    {
        std::lock_guard<std::mutex> lk(g_flip_mutex);
        update_flips();
        g_swap_count++;
        g_pending_flips.push_back({vsync_index(), g_swap_count});
        // The hardware swap blocks until a flip slot frees up (double buffering), which is what
        // paces games that never call GX2WaitForVsync. Without this the queue grows without
        // bound and the game logic free-runs far above the display rate.
        if (!uncapped_benchmark() && g_pending_flips.size() > 2) {
            front_due = flip_due(g_pending_flips.front());
            wait_for_slot = true;
        }
    }
    if (wait_for_slot) {
        BlockingScope b;
        threads::park_sleep_until(tick_time(front_due));
        std::lock_guard<std::mutex> lk(g_flip_mutex);
        update_flips();
    }
    // debug: WWHD_LOG_SLOW_SWAP=ms logs swaps that came more than ms after the previous one
    static const double slow_ms = getenv("WWHD_LOG_SLOW_SWAP") ? atof(getenv("WWHD_LOG_SLOW_SWAP")) : 0;
    if (slow_ms > 0) {
        static auto prev = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - prev).count();
        prev = now;
        if (ms > slow_ms) LOG("[gx2] slow swap %llu: %.1f ms", (unsigned long long)g_swap_count, ms);
    }
    if (g_swap_count % 300 == 1) {
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(now - last).count();
        last = now;
        LOG("[gx2] frame %llu, %.1f swaps/s, swap interval %u, draws %llu", (unsigned long long)g_swap_count, 300 / s, g_swap_interval, g_stat_draws.load(std::memory_order_relaxed));
        if (getenv("WWHD_SCHED_STATS")) threads::report_sched();
    }
}
HLE(gx2, GX2GetSwapStatus) {
    std::lock_guard<std::mutex> lk(g_flip_mutex);
    update_flips();
    if (arg(c, 0)) st32(arg(c, 0), (uint32)(g_swap_count + g_count_offset));
    if (arg(c, 1)) st32(arg(c, 1), (uint32)(g_flip_count + g_count_offset));
    if (arg(c, 2)) st64(arg(c, 2), timebase::to_guest(g_last_flip_time));
    if (arg(c, 3)) st64(arg(c, 3), timebase::guest_now());
}
HLE(gx2, GX2SetSwapInterval) { g_swap_interval = std::max<uint32>(arg(c, 0), 1); }
HLE(gx2, GX2GetSwapInterval) { ret(c, g_swap_interval); }
// ms: the game waits up to timeout * 60 / 1000 vsyncs for a flip before it reports a GPU hang
HLE(gx2, GX2GetGPUTimeout) { ret(c, 10000); }
static uint32 g_misc_params[8];
HLE(gx2, GX2SetMiscParam) { if (arg(c, 0) < 8) g_misc_params[arg(c, 0)] = arg(c, 1); ret(c, 1); }
HLE(gx2, GX2GetMiscParam) { ret(c, arg(c, 0) < 8 ? g_misc_params[arg(c, 0)] : 0); }
// debugging aids of the SDK: nothing to do
HLE(gx2, GX2PrintGPUStatus) {}
HLE(gx2, GX2DebugTagUserString) {}
HLE(gx2, GX2DebugTagUserStringVA) {}
HLE(gx2, GX2TempDumpResources) {}
HLE(gx2, GX2TempGetResourceCount) { ret(c, 0); }
HLE(gx2, GX2RSetAllocator) {}  // no GX2R buffer calls are imported
HLE(gx2, GX2RGetDebugOptions) { ret(c, 0); }
// the renderer takes MSAA from the surfaces, and hi-stencil is the hardware's early stencil reject
HLE(gx2, GX2SetAAMode) {}
HLE(gx2, GX2SetHiStencilInfo) {}
HLE(gx2, GX2Shutdown) {}
HLE(gx2, GX2SetTVGamma) {}
HLE(gx2, GX2SetDRCGamma) {}
HLE(gx2, GX2WaitForVsync) {
    if(uncapped_benchmark()) {
        BlockingScope b;
        // The queue fence follows earlier swaps, whose presentation path waits
        // for GPU completion. Never wait while holding the flip mutex.
        render_sync(kSyncVsyncUncapped);
        std::lock_guard<std::mutex> lk(g_flip_mutex);
        update_flips();
        return;
    }
#ifdef WWHD_HAS_VULKAN
    static const bool readyFlipWait = [] {
        const char* value = getenv("WWHD_VK_READY_FLIP_WAIT");
        return render::vulkan() && value && !strcmp(value, "1");
    }();
    if(readyFlipWait) {
        bool eligible = false, needsSync = false;
        {
            std::lock_guard<std::mutex> lk(g_flip_mutex);
            if(!g_pending_flips.empty()) {
                const auto& front = g_pending_flips.front();
                const uint64_t at = flip_due(front);
                eligible = at <= vsync_index();
                if(eligible) needsSync = render::frames_completed() < front.swap;
            }
        }
        if(eligible) {
            if(needsSync) {
                BlockingScope b;
                render_sync(kSyncVsyncFlip); // Queued swap completion; never hold flip mutex.
            }
            std::lock_guard<std::mutex> lk(g_flip_mutex);
            update_flips(); // Retains minimum interval and FIFO GPU guards.
            return;
        }
    }
#endif
    static const bool preciseSleep = [] {
#ifdef WWHD_HAS_VULKAN
        // Vulkan renderer's pacing (docs/vulkan.md); the Metal renderer keeps plain sleeping
        if (!render::vulkan()) return false;
        const char* value = getenv("WWHD_VSYNC_PRECISE");
#ifdef __APPLE__
        // Avoid the measured macOS sleep overshoot; explicit zero opts out.
        return !value || atoi(value) != 0;
#else
        return value && atoi(value) != 0;
#endif
#else
        return false;
#endif
    }();
    const uint64_t granule = vsync_granule();
    const auto deadline = tick_time((vsync_index() / granule + 1) * granule);  // the next (virtual) vsync
#ifdef WWHD_HAS_VULKAN
    static const bool readyFlipPark = [] {
        const char* value = getenv("WWHD_VK_READY_FLIP_PARK");
        return render::vulkan() && value && !strcmp(value, "1");
    }();
    threads::park_sleep_until(deadline, preciseSleep,
        readyFlipPark ? ready_flip_before_resume : nullptr);
#else
    threads::park_sleep_until(deadline, preciseSleep);
#endif
    std::lock_guard<std::mutex> lk(g_flip_mutex);
    update_flips();
    static uint64_t calls = 0;
    if (getenv("WWHD_LOG_VSYNC") && ++calls % 60 == 0)
        LOG("[gx2] vsync %llu: swaps %llu flips %llu pending %zu", (unsigned long long)vsync_index(),
            (unsigned long long)g_swap_count, (unsigned long long)g_flip_count, g_pending_flips.size());
}

// scan buffers: the game renders into its own color buffers and copies to "scan buffers";
// the renderer presents whatever was copied to the TV target.
// (buffer, size, mode, surfaceFormat, bufferingMode): an sRGB format means scan-out applies the encoding
HLE(gx2, GX2SetTVBuffer) {
    LOG("[gx2] TV buffer format %X", arg(c, 3));
    render::set_tv_format(arg(c, 3), true);
}
HLE(gx2, GX2SetDRCBuffer) { render::set_tv_format(arg(c, 3), false); }
HLE(gx2, GX2SetTVScale) {}
HLE(gx2, GX2SetDRCScale) {}
HLE(gx2, GX2SetTVEnable) {}
HLE(gx2, GX2SetDRCEnable) {}
HLE(gx2, GX2CalcTVSize) {
    // (mode, format, bufferingMode, uint32* size, bool* scaleNeeded)
    uint32 mode = arg(c, 0), buffers = std::max<uint32>(arg(c, 2), 1);
    uint32 w = mode >= 5 ? 1920 : mode <= 2 ? 854 : 1280, h = mode >= 5 ? 1080 : mode <= 2 ? 480 : 720;
    st32(arg(c, 3), w * h * 4 * buffers);
    st32(arg(c, 4), 0);
}
HLE(gx2, GX2CalcDRCSize) {
    st32(arg(c, 3), 854 * 480 * 4 * std::max<uint32>(arg(c, 2), 1));
    st32(arg(c, 4), 0);
}

// ---------------------------------------------------------------- misc queries
HLE(gx2, GX2TempGetGPUVersion) { ret(c, 2); }
HLE(gx2, GX2CalcGeometryShaderInputRingBufferSize) { ret(c, arg(c, 0) * 4 * 0x1000); }
HLE(gx2, GX2CalcGeometryShaderOutputRingBufferSize) { ret(c, arg(c, 0) * 4 * 0x1000); }
HLE(gx2, GX2CalcFetchShaderSizeEx) {
    uint32 n = arg(c, 0);
    uint32 cf = ((((n + 15) / 16) + 1) * 8 + 0xF) & ~0xFu;
    ret(c, std::max<uint32>(cf + n * 16, 16 + n * 16));
}
HLE(gx2, GX2GPUTimeToCPUTime) { ret64(c, arg64(c, 3)); }
HLE(gx2, GX2SampleTopGPUCycle) { if (arg(c, 0)) st64(arg(c, 0), timebase::guest_now()); }
HLE(gx2, GX2SampleBottomGPUCycle) { if (arg(c, 0)) st64(arg(c, 0), timebase::guest_now()); }

// ---------------------------------------------------------------- save states
#include "../savestate.h"

// the game is frozen between frames: finish all queued GPU work and let pending flips execute, so no
// command reads guest memory while it is replaced and the swap/flip counts agree
void gx2_ss_drain() {
    emit_host(OP_DRAW_DONE, {1});  // full GPU wait, also with lazy DrawDone
    render_sync(kSyncSaveState);
    for (int i = 0; i < 300; i++) {
        {
            std::lock_guard<std::mutex> lk(g_flip_mutex);
            update_flips();
            if (g_pending_flips.empty()) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG("[savestate] flips still pending after 300 ms");
}

void gx2_ss_save(ss::Writer& w) {
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    w.u32(kNumRegs);
    w.bytes(g_regs, sizeof g_regs);
    uint32 active = 0;
    std::vector<uint32> keys;
    for (auto& [k, v] : g_contexts) {
        keys.push_back(k);
        if (g_shadow == v.data()) active = k;
    }
    std::sort(keys.begin(), keys.end());
    w.u32((uint32)keys.size());
    for (uint32 k : keys) {
        w.u32(k);
        w.u32((uint32)g_contexts[k].size());
        w.bytes(g_contexts[k].data(), g_contexts[k].size() * 4);
    }
    w.u32(active);
    w.u32(g_swap_interval);
    std::lock_guard<std::mutex> fl(g_flip_mutex);
    w.u64(g_swap_count + g_count_offset);
    w.u8(g_legacy_dl_headers);
}

bool gx2_ss_check(ss::Reader r, std::string& why) {
    if (r.u32() != kNumRegs) { why = "GX2 register file size differs"; return false; }
    return r.ok;
}

void gx2_ss_load(ss::Reader& r) {
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    r.u32();
    const uint32 beforePS = g_regs[mmSQ_PGM_START_PS], beforeVS = g_regs[mmSQ_PGM_START_VS];
    r.bytes(g_regs, sizeof g_regs);
    g_stream_out_active = (g_regs[mmVGT_STRMOUT_EN] & 1) != 0;
    g_contexts.clear();
    uint32 n = r.u32();
    for (uint32 i = 0; i < n && r.ok; i++) {
        uint32 k = r.u32(), words = r.u32();
        auto& v = g_contexts[k];
        v.resize(words);
        r.bytes(v.data(), (size_t)words * 4);
    }
    uint32 active = r.u32();
    auto it = g_contexts.find(active);
    g_shadow = active && it != g_contexts.end() ? it->second.data() : nullptr;
    if (g_trace_shader_writes) {
        for (uint32 reg : {mmSQ_PGM_START_PS, mmSQ_PGM_START_VS})
            g_shader_writes[g_shader_write_count++ % g_shader_writes.size()] =
                {"savestate-load", {}, active, 0, kNumRegs, reg,
                 reg == mmSQ_PGM_START_PS ? beforePS : beforeVS, g_regs[reg]};
    }
    g_shader_state_gen++;
    g_swap_interval = std::max<uint32>(r.u32(), 1);
    uint64_t guest_swaps = r.u64();
    g_legacy_dl_headers = r.at_end() || r.u8() != 0;
    {
        std::lock_guard<std::mutex> fl(g_flip_mutex);
        g_count_offset = (int64_t)guest_swaps - (int64_t)g_swap_count;
    }
    render::ss_reset();  // the renderer forgets surface contents and shader memos (Vulkan)
}
