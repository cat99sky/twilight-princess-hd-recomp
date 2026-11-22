// Libraries the game uses for system integration and online features.
// Online services (Miiverse, SpotPass, accounts) report "unavailable".
#include "../crashrec.h"
#include <mutex>
#include <chrono>
#include <thread>
#include <algorithm>
#include "../runtime.h"
#include "../input.h"
#include "../rumble.h"
#include "../motion/motion.h"

namespace interp { uint64_t logic_steps(); }

// nn::Result: bit 31 set = failure
static constexpr uint32_t kResultOk = 0;
static constexpr uint32_t kResultFail = 0xA0000000;

// ---- coreinit MP task queues (Cemu's 0x50-byte queue / wut's 0x2C-byte task)
// All queue transitions use one process-wide mutex. This avoids resetting the
// guest spin lock while a worker dequeues or completes a task (known YWW race).
static std::mutex g_mp_queue_mutex;
static constexpr uint32_t MP_Q_INIT = 1, MP_Q_READY = 2, MP_Q_STOPPING = 4,
                          MP_Q_STOPPED = 8, MP_Q_FINISHED = 16;
static constexpr uint32_t MP_T_INIT = 1, MP_T_READY = 2, MP_T_RUNNING = 4,
                          MP_T_FINISHED = 8;
static bool mp_valid_queue(uint32_t q) { return q && ld32(q) == q; }
static uint32_t mp_take_task(uint32_t q) {
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (!mp_valid_queue(q) || ld32(q + 4) != MP_Q_READY) return 0;
    uint32_t next = ld32(q + 0x24), end = ld32(q + 0x30);
    if (next >= end || end > ld32(q + 0x3C)) return 0;
    const uint32_t task = ld32(ld32(q + 0x38) + 4 * next);
    st32(q + 0x24, next + 1);
    return task;
}
static bool mp_execute_task(Cpu* c, uint32_t t) {
    uint32_t q = 0, fn = 0, a = 0, b = 0;
    {
        std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
        if (!t || ld32(t) != t || ld32(t + 8) != MP_T_READY) return false;
        q = ld32(t + 4);
        if (!mp_valid_queue(q) || ld32(q + 4) == MP_Q_STOPPING ||
            ld32(q + 4) == MP_Q_STOPPED || ld32(q + 0x0C) == 0) return false;
        fn = ld32(t + 0x0C);
        if (!fn) return false;
        a = ld32(t + 0x10); b = ld32(t + 0x14);
        st32(q + 0x0C, ld32(q + 0x0C) - 1);
        st32(q + 0x10, ld32(q + 0x10) + 1);
        st32(t + 8, MP_T_RUNNING);
        st32(t + 0x1C, c->core);
    }
    const auto started = std::chrono::steady_clock::now();
    uint32_t value = 0;
    bool completed = false;
    try {
        value = guest_call(c, fn, {a, b});
        completed = true;
    } catch (...) {
        // Finish bookkeeping before propagating the guest/host exception.
        // A failed task is not reported as successfully completed.
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    {
        std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
        st32(t + 0x18, value);
        st64(t + 0x20, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 16));
        st32(t + 8, completed ? MP_T_FINISHED : MP_T_INIT);
        // A queue may be reinitialized by the main thread during work.
        // Only update counters while this task still belongs to that queue.
        if (mp_valid_queue(q) && ld32(t + 4) == q) {
            const uint32_t running = ld32(q + 0x10);
            if (running) st32(q + 0x10, running - 1);
            if (completed) st32(q + 0x18, ld32(q + 0x18) + 1);
            else st32(q + 0x0C, ld32(q + 0x0C) + 1);
            if (ld32(q + 4) == MP_Q_STOPPING && ld32(q + 0x10) == 0)
                st32(q + 4, MP_Q_STOPPED);
            if (completed && ld32(q + 0x08) == ld32(q + 0x18))
                st32(q + 4, MP_Q_FINISHED);
        }
    }
    if (!completed) fatal("[mp] guest task at %08X threw an exception", fn);
    return true;
}
HLE(coreinit, OSIsMainCore) { ret(c, c->core == 1 ? 1u : 0u); }
HLE(coreinit, MPInitTaskQ) {
    const uint32_t q = arg(c, 0);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (!q) fatal("[mp] MPInitTaskQ(NULL)");
    memset(mem::ptr(q), 0, 0x50);
    st32(q, q);
    st32(q + 4, MP_Q_INIT);
    st32(q + 0x38, arg(c, 1));
    st32(q + 0x3C, arg(c, 2));
}
HLE(coreinit, MPStartTaskQ) {
    const uint32_t q = arg(c, 0);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    const bool ok = mp_valid_queue(q) && (ld32(q + 4) == MP_Q_INIT || ld32(q + 4) == MP_Q_STOPPED);
    if (ok) st32(q + 4, MP_Q_READY);
    ret(c, ok ? 1 : 0);
}
HLE(coreinit, MPGetTaskQInfo) {
    const uint32_t q = arg(c, 0), info = arg(c, 1);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (!info || !mp_valid_queue(q)) { ret(c, 0); return; }
    st32(info + 0, ld32(q + 4));
    st32(info + 4, ld32(q + 8));
    st32(info + 8, ld32(q + 0x0C));
    st32(info + 12, ld32(q + 0x10));
    st32(info + 16, ld32(q + 0x18));
    ret(c, 1);
}
HLE(coreinit, MPInitTask) {
    const uint32_t task = arg(c, 0);
    if (!task) fatal("[mp] MPInitTask(NULL)");
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    memset(mem::ptr(task), 0, 0x2C);
    st32(task, task); st32(task + 8, MP_T_INIT);
    st32(task + 0x0C, arg(c, 1));
    st32(task + 0x10, arg(c, 2)); st32(task + 0x14, arg(c, 3));
    st32(task + 0x1C, 3);  // no core assigned
}
HLE(coreinit, MPEnqueTask) {
    const uint32_t q = arg(c, 0), task = arg(c, 1);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (!mp_valid_queue(q) || !task || ld32(task) != task ||
        ld32(task + 8) != MP_T_INIT) { ret(c, 0); return; }
    const uint32_t state = ld32(q + 4), end = ld32(q + 0x30);
    if (state != MP_Q_INIT && state != MP_Q_READY && state != MP_Q_STOPPING &&
        state != MP_Q_STOPPED && state != MP_Q_FINISHED) { ret(c, 0); return; }
    if (end >= ld32(q + 0x3C) || !ld32(q + 0x38)) { ret(c, 0); return; }
    st32(task + 4, q); st32(task + 8, MP_T_READY);
    st32(ld32(q + 0x38) + end * 4, task);
    st32(q + 0x30, end + 1);
    st32(q + 8, ld32(q + 8) + 1);
    st32(q + 0x0C, ld32(q + 0x0C) + 1);
    if (state == MP_Q_FINISHED) st32(q + 4, MP_Q_READY);
    ret(c, 1);
}
HLE(coreinit, MPDequeTask) { ret(c, mp_take_task(arg(c, 0))); }
HLE(coreinit, MPDequeTasks) {
    const uint32_t q = arg(c, 0), out = arg(c, 1), max = arg(c, 2);
    if (!out) { ret(c, 0); return; }
    uint32_t count = 0;
    while (count < max) {
        const uint32_t t = mp_take_task(q);
        if (!t) break;
        st32(out + 4 * count++, t);
    }
    ret(c, count);
}
HLE(coreinit, MPRunTask) { ret(c, mp_execute_task(c, arg(c, 0)) ? 1 : 0); }
HLE(coreinit, MPRunTasksFromTaskQ) {
    const uint32_t q = arg(c, 0), batch = std::max(1u, arg(c, 1));
    uint32_t n = 0;
    for (;;) {
        const uint32_t t = mp_take_task(q);
        if (!t) break;
        if (mp_execute_task(c, t)) ++n;
        if (n && n % batch == 0) {
            BlockingScope yield_core;
            std::this_thread::yield();
        }
    }
    ret(c, n ? 1 : 0);
}
HLE(coreinit, MPGetTaskInfo) {
    const uint32_t t = arg(c, 0), info = arg(c, 1);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (!t || !info || ld32(t) != t) { ret(c, 0); return; }
    st32(info, ld32(t + 8));
    st32(info + 4, ld32(t + 0x18));
    st32(info + 8, ld32(t + 0x1C));
    st64(info + 0x0C, ld64(t + 0x20));
    ret(c, 1);
}
HLE(coreinit, MPGetTaskUserData) {
    const uint32_t t = arg(c, 0);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    ret(c, t && ld32(t) == t ? ld32(t + 0x28) : 0);
}
HLE(coreinit, MPSetTaskUserData) {
    const uint32_t t = arg(c, 0);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    if (t && ld32(t) == t) st32(t + 0x28, arg(c, 1));
}
HLE(coreinit, MPStopTaskQ) {
    const uint32_t q = arg(c, 0);
    std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
    const bool ok = mp_valid_queue(q) && ld32(q + 4) == MP_Q_READY;
    if (ok) st32(q + 4, ld32(q + 0x10) ? MP_Q_STOPPING : MP_Q_STOPPED);
    ret(c, ok ? 1 : 0);
}
HLE(coreinit, MPWaitTaskQ) {
    const uint32_t q = arg(c, 0), mask = arg(c, 1);
    if (!mask) { ret(c, 0); return; }
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
            if (!mp_valid_queue(q)) { ret(c, 0); return; }
            if (ld32(q + 4) & mask) { ret(c, 1); return; }
        }
        BlockingScope wait;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
HLE(coreinit, MPTermTaskQ) {
    const uint32_t q = arg(c, 0);
    {
        std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
        if (!mp_valid_queue(q)) { ret(c, 0); return; }
        if (ld32(q + 4) == MP_Q_READY)
            st32(q + 4, ld32(q + 0x10) ? MP_Q_STOPPING : MP_Q_STOPPED);
    }
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(g_mp_queue_mutex);
            if (!mp_valid_queue(q)) { ret(c, 0); return; }
            if (ld32(q + 0x10) == 0) { ret(c, 1); return; }
        }
        BlockingScope wait;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
// ---- nn_ac (network access): offline console semantics
// Smash4PC writes the "not connected" status (0xFFFFFFFF) so a title takes its offline
// path instead of treating a zeroed status as a live connection.
HLE(nn_ac, GetStatus__Q2_2nn2acFPQ3_2nn2ac6Status) {
    st32(arg(c, 0), 0xFFFFFFFFu);
    ret(c, kResultOk);
}
HLE(nn_ac, IsApplicationConnected__Q2_2nn2acFPb) {
    st8(arg(c, 0), 0);
    ret(c, kResultOk);
}


// ---- nn_act (accounts)
HLE(nn_act, Initialize__Q2_2nn3actFv) { ret(c, kResultOk); }
HLE(nn_act, Finalize__Q2_2nn3actFv) { ret(c, kResultOk); }
HLE(nn_act, GetSlotNo__Q2_2nn3actFv) { ret(c, 1); }
HLE(nn_act, GetPrincipalId__Q2_2nn3actFv) { ret(c, 0); }
HLE(nn_act, GetParentalControlSlotNoEx__Q2_2nn3actFPUcUc) { st8(arg(c, 0), 1); ret(c, kResultOk); }
HLE(nn_act, GetUuid__Q2_2nn3actFP7ACTUuid) { memset(mem::ptr(arg(c, 0)), 0, 16); ret(c, kResultOk); }

// ---- nn_ac (network connection)
HLE(nn_ac, Initialize__Q2_2nn2acFv) { ret(c, kResultOk); }
HLE(nn_ac, Finalize__Q2_2nn2acFv) { ret(c, kResultOk); }
HLE(nn_ac, Connect__Q2_2nn2acFv) { ret(c, kResultFail); }
HLE(nn_ac, Close__Q2_2nn2acFv) { ret(c, kResultOk); }
HLE(nn_ac, GetLastErrorCode__Q2_2nn2acFPUi) { st32(arg(c, 0), 1100); ret(c, kResultOk); }

// ---- nn_boss (SpotPass)
HLE(nn_boss, Initialize__Q2_2nn4bossFv) { ret(c, kResultFail); }
HLE(nn_boss, IsInitialized__Q2_2nn4bossFv) { ret(c, 0); }
HLE(nn_boss, Finalize__Q2_2nn4bossFv) {}
// play report (usage statistics upload): accepted and dropped
HLE(nn_boss, __ct__Q3_2nn4boss17PlayReportSettingFv) { ret(c, arg(c, 0)); }
HLE(nn_boss, __dt__Q3_2nn4boss17PlayReportSettingFv) {}
HLE(nn_boss, Initialize__Q3_2nn4boss17PlayReportSettingFPvUi) {}
HLE(nn_boss, Set__Q3_2nn4boss17PlayReportSettingFUiT1) { ret(c, 1); }
HLE(nn_boss, __ct__Q3_2nn4boss4TaskFv) { ret(c, arg(c, 0)); }
HLE(nn_boss, __dt__Q3_2nn4boss4TaskFv) {}
HLE(nn_boss, Initialize__Q3_2nn4boss4TaskFPCcUi) { ret(c, kResultFail); }
HLE(nn_boss, IsRegistered__Q3_2nn4boss4TaskCFv) { ret(c, 0); }
HLE(nn_boss, Register__Q3_2nn4boss4TaskFRQ3_2nn4boss11TaskSetting) { ret(c, kResultFail); }
HLE(nn_boss, StartScheduling__Q3_2nn4boss4TaskFb) { ret(c, kResultFail); }

// ---- nn_olv (Miiverse)
HLE(nn_olv, Initialize__Q2_2nn3olvFPCQ3_2nn3olv15InitializeParam) { ret(c, kResultFail); }
HLE(nn_olv, IsInitialized__Q2_2nn3olvFv) { ret(c, 0); }
HLE(nn_olv, Finalize__Q2_2nn3olvFv) { ret(c, kResultOk); }
HLE(nn_olv, GetErrorCode__Q2_2nn3olvFRCQ2_2nn6Result) { ret(c, 0); }
// posting parameters are built and accepted; the post itself fails
HLE(nn_olv, __ct__Q3_2nn3olv15InitializeParamFv) { ret(c, arg(c, 0)); }
HLE(nn_olv, SetWork__Q3_2nn3olv15InitializeParamFPUcUi) { ret(c, kResultOk); }
HLE(nn_olv, __ct__Q3_2nn3olv28UploadPostDataByPostAppParamFv) { ret(c, arg(c, 0)); }
HLE(nn_olv, __dt__Q3_2nn3olv28UploadPostDataByPostAppParamFv) {}
HLE(nn_olv, SetWork__Q3_2nn3olv28UploadPostDataByPostAppParamFPUcUi) { ret(c, kResultOk); }
HLE(nn_olv, SetFlags__Q3_2nn3olv28UploadPostDataByPostAppParamFUi) { ret(c, kResultOk); }
HLE(nn_olv, SetBodyTextMaxLength__Q3_2nn3olv28UploadPostDataByPostAppParamFUi) { ret(c, kResultOk); }
HLE(nn_olv, SetExternalImageData__Q3_2nn3olv15UploadParamBaseFPCUcUi) { ret(c, kResultOk); }
HLE(nn_olv, AddCommonData__Q3_2nn3olv15UploadParamBaseFUiPCUcT1) { ret(c, kResultOk); }
HLE(nn_olv, GetDataTypeNum__Q3_2nn3olv15UploadParamBaseCFUi) { ret(c, 0); }
HLE(nn_olv, UploadPostDataByPostApp__Q2_2nn3olvFPCQ3_2nn3olv28UploadPostDataByPostAppParam) { ret(c, kResultFail); }

// ---- nn_nfp (amiibo): see nfp.cpp

// ---- sockets / curl
HLE(nsysnet, socket_lib_init) { ret(c, 0); }
HLE(nsysnet, socket_lib_finish) { ret(c, 0); }
HLE(nsysnet, NSSLInit) { ret(c, 0); }
HLE(nsysnet, NSSLFinish) { ret(c, 0); }
HLE(nlibcurl, curl_global_init) { ret(c, 0); }  // CURLE_OK (the network itself is unavailable: nn_ac)
HLE(nlibcurl, curl_global_init_mem) { ret(c, 0); }
HLE(nlibcurl, curl_global_cleanup) {}

// ---- proc_ui (foreground/background lifecycle)
HLE(proc_ui, ProcUIInit) {}
HLE(proc_ui, ProcUIInitEx) {}
HLE(proc_ui, ProcUIShutdown) {}
HLE(proc_ui, ProcUIRegisterCallback) {}
HLE(proc_ui, ProcUIDrawDoneRelease) {}
HLE(proc_ui, ProcUIProcessMessages) { ret(c, 0); }  // PROCUI_STATUS_IN_FOREGROUND

// ---- vpad (GamePad): keyboard / host controllers
HLE(vpad, VPADSetButtonProcMode) {}
HLE(vpad, VPADEnableGyroZeroPlay) {}
HLE(vpad, VPADSetGyroZeroPlayParam) {}
HLE(vpad, VPADRead) {
    // (chan, VPADStatus* buf, count, int32* error) -> samples written
    uint32_t chan = arg(c, 0), st = arg(c, 1), count = arg(c, 2), err = arg(c, 3);
    // debug: WWHD_TRACE_VPAD=n logs the guest call chain of the first n reads
    static int trace = getenv("WWHD_TRACE_VPAD") ? atoi(getenv("WWHD_TRACE_VPAD")) : 0;
    if (trace > 0) {
        trace--;
        char buf[256];
        int n = snprintf(buf, sizeof buf, "[vpad] read from lr=%08X", c->lr);
        for (uint32_t sp = c->r[1], i = 0; i < 8 && sp; i++) {
            uint32_t prev = ld32(sp);
            if (!prev || prev <= sp) break;
            n += snprintf(buf + n, sizeof buf - n, " <- %08X", ld32(prev + 4));
            sp = prev;
        }
        LOG("%s", buf);
    }
    if (chan != 0 || !st || (int32_t)count <= 0) {
        if (err) st32(err, (uint32_t)-2);  // VPAD_READ_INVALID_CONTROLLER
        ret(c, 0);
        return;
    }
    static uint32_t last_hold = 0;
    const bool repeat = false;  // (frame interpolation repeated samples; TP HD runs at 30 fps)
    input::PadState p = crashrec::read(0);  // live input, recorded/replayed by crash recovery
    if (input::pro_controller()) {  // GamePad on the table: screen and touch only
        p.buttons = 0;
        p.lx = p.ly = p.rx = p.ry = 0;
    } else if (!repeat) {
        motion::right_stick(p.rx, p.ry);  // for the gyro log
    }
    uint32_t hold = p.buttons;
    auto stick_dirs = [&](float x, float y, uint32_t up, uint32_t down, uint32_t left, uint32_t right) {
        // stick-as-button bits with hysteresis, like the real VPAD library
        auto dir = [&](float v, bool held, uint32_t bit) { if (v >= 0.5f || (held && v >= 0.1f)) hold |= bit; };
        dir(-x, last_hold & left, left); if (!(hold & left)) dir(x, last_hold & right, right);
        dir(-y, last_hold & down, down); if (!(hold & down)) dir(y, last_hold & up, up);
    };
    stick_dirs(p.lx, p.ly, 0x10000000, 0x08000000, 0x40000000, 0x20000000);
    stick_dirs(p.rx, p.ry, 0x01000000, 0x00800000, 0x04000000, 0x02000000);
    if (repeat) hold = last_hold;  // buttons (and the stick-as-button bits) change on full passes only

    memset(mem::ptr(st), 0, 0xAC);
    st32(st + 0x00, hold);
    st32(st + 0x04, hold & ~last_hold);   // trig
    st32(st + 0x08, last_hold & ~hold);   // release
    last_hold = hold;
    {  // debug: WWHD_PAD_TRACE=path logs each read: logic step, repeated, buttons, trigger
        static FILE* pt = getenv("WWHD_PAD_TRACE") ? fopen(getenv("WWHD_PAD_TRACE"), "w") : nullptr;
        if (pt) { fprintf(pt, "%llu %d %08X %08X\n", (unsigned long long)interp::logic_steps(), (int)repeat, hold, ld32(st + 4)); fflush(pt); }
    }
    stf32(st + 0x0C, p.lx); stf32(st + 0x10, p.ly);
    stf32(st + 0x14, p.rx); stf32(st + 0x18, p.ry);
    {  // motion sensors (motion/motion.h). WWHD reads only the direction matrix (0x6C..0x8F): its
       // first-person camera turns by the change from one frame to the next (dCamera_c::CalcSubjectAngle)
        const motion::VpadMotion m = motion::vpad(repeat);
        auto vec = [&](uint32_t at, const motion::Vec3& v) { stf32(at, v.x); stf32(at + 4, v.y); stf32(at + 8, v.z); };
        vec(st + 0x1C, m.acc);
        stf32(st + 0x28, m.acc_magnitude);
        stf32(st + 0x2C, m.acc_variation);
        stf32(st + 0x30, m.acc_xy[0]); stf32(st + 0x34, m.acc_xy[1]);
        vec(st + 0x38, m.gyro);
        vec(st + 0x44, m.angle);
        for (int i = 0; i < 3; i++) vec(st + 0x6C + i * 0xC, m.dir[i]);
    }
    // touch panel, raw coordinates as the hardware reports them (mapping from Cemu)
    static uint16_t last_tx = 0, last_ty = 0;
    if (p.touch) {
        last_tx = (uint16_t)(p.tx * 3883.0f + 92.0f);
        last_ty = (uint16_t)(4095.0f - p.ty * 3694.0f - 254.0f);
    }
    for (uint32_t tp = 0x52; tp <= 0x62; tp += 8) {
        st16(st + tp + 0, last_tx);
        st16(st + tp + 2, last_ty);
        st16(st + tp + 4, p.touch ? 1 : 0);
        st16(st + tp + 6, p.touch ? 0 : 3);  // validity: 0 = valid, 3 = invalid XY
    }
    st8(st + 0xA0, 0xFF); st8(st + 0xA3, 0xFF);                // slide volume
    st8(st + 0xA1, 0xC0);                                      // battery full
    if (err) st32(err, 0);
    ret(c, 1);
}
// raw touch coordinates -> 1280x720 screen space
static void tp_to_screen(uint32_t out, uint32_t raw, int w, int h) {
    int x = std::max<int>(ld16(raw) - 92, 0), y = std::max<int>(4095 - (int)ld16(raw + 2) - 254, 0);
    st16(out, (uint16_t)(x / 3883.0 * w));
    st16(out + 2, (uint16_t)(y / 3694.0 * h));
    st16(out + 4, ld16(raw + 4));
    st16(out + 6, ld16(raw + 6));
}
HLE(vpad, VPADGetTPCalibratedPoint) { tp_to_screen(arg(c, 1), arg(c, 2), 1280, 720); }
HLE(vpad, VPADGetTPCalibratedPointEx) {
    int res = (int)arg(c, 1);  // 0 = 1920x1080, 1 = 1280x720, 2 = 854x480
    tp_to_screen(arg(c, 2), arg(c, 3), res == 0 ? 1920 : res == 2 ? 854 : 1280, res == 0 ? 1080 : res == 2 ? 480 : 720);
}
// The GamePad motor: the game sends a pattern of up to 120 bits (not bytes), played at 120 bits a
// second; it stops by itself at the end, and an empty pattern or VPADStopMotor stops it at once.
// rumble.h turns that into what the host controllers' motors do.
HLE(vpad, VPADControlMotor) {
    // (chan, uint8* pattern, uint8 length in bits) -> int32 error
    uint32_t chan = arg(c, 0), pattern = arg(c, 1), nbits = std::min<uint32_t>(arg(c, 2) & 0xFF, rumble::Motor::kMaxBits);
    uint8_t bits[rumble::Motor::kMaxBits / 8] = {};
    for (uint32_t i = 0; pattern && i < (nbits + 7) / 8; i++) bits[i] = ld8(pattern + i);
    TRACE("[pad] VPADControlMotor(%u, %u bits)", chan, nbits);
    rumble::gamepad_pattern(chan, pattern ? bits : nullptr, nbits);
    ret(c, 0);
}
HLE(vpad, VPADStopMotor) { rumble::gamepad_stop(arg(c, 0)); }
HLE(vpadbase, VPADBASEGetHeadphoneStatus) { ret(c, 0); }

// ---- padscore (Wii Remote / Pro Controller): see padscore.cpp

// ---- sysapp
HLE(sysapp, SYSLaunchSettings) { ret(c, 0); }
// launched directly, with no arguments: the caller's request slots keep their null data
HLE(sysapp, SYSGetArguments) { ret(c, 0); }
HLE(sysapp, SYSSwitchToEManual) {}  // there is no electronic manual to open

// ---- erreula (looked up through OSDynLoad): the error viewer is never shown
HLE(erreula, ErrEulaSetVersion__3RplFi) {}
HLE(erreula, ErrEulaPlayAppearSE__3RplFb) {}
HLE(erreula, ErrEulaJump__3RplFPCcUi) {}
HLE(sysapp, SYSLaunchAccount) { ret(c, 0); }
