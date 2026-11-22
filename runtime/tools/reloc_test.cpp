// Unit tests for the RPX data relocation pass (runtime/src/rpx_reloc.cpp).
//
// Builds synthetic section tables in memory, so the test needs no game data:
//   .text   holds an instruction word that must never be patched
//   .data   holds four ADDR32 sites plus one unsupported and one out-of-range site
//   .rela.data / .rela.text  carry the RELA records
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rpx_reloc.h"

namespace {

int g_failures = 0;

void check(const char* name, bool ok, const char* detail = "") {
    if (!ok) {
        std::printf("FAIL %s %s\n", name, detail);
        g_failures++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

void put_be32(std::vector<uint8_t>& v, uint32_t value) {
    v.push_back((uint8_t)(value >> 24));
    v.push_back((uint8_t)(value >> 16));
    v.push_back((uint8_t)(value >> 8));
    v.push_back((uint8_t)value);
}

uint32_t read_be32(const std::vector<uint8_t>& v) {
    return (uint32_t)v[0] << 24 | (uint32_t)v[1] << 16 | (uint32_t)v[2] << 8 | v[3];
}

struct Guest {
    // DATA_BASE..0x10010000 stands in for .data, 0x02000000.. for .text.
    static constexpr uint32_t DATA_BASE = 0x10290000;
    std::vector<uint8_t> data = std::vector<uint8_t>(0x10000, 0);
    std::vector<uint8_t> text = std::vector<uint8_t>(0x100, 0xDE);  // 0xDEADBEEF-friendly filler

    void write32(uint32_t addr, uint32_t value) {
        if (addr >= DATA_BASE && addr + 4 <= DATA_BASE + data.size()) {
            uint8_t* p = &data[addr - DATA_BASE];
            p[0] = (uint8_t)(value >> 24); p[1] = (uint8_t)(value >> 16);
            p[2] = (uint8_t)(value >> 8);  p[3] = (uint8_t)value;
        } else if (addr >= 0x02000000 && addr + 4 <= 0x02000000 + text.size()) {
            g_failures++;
            std::printf("FAIL relocation wrote into .text at %08X\n", addr);
        } else {
            g_failures++;
            std::printf("FAIL relocation wrote outside mapped data at %08X\n", addr);
        }
    }

    uint32_t read32(uint32_t addr) const {
        const uint8_t* p = &data[addr - DATA_BASE];
        return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    }
};

std::vector<rpx::Section> build_sections() {
    std::vector<rpx::Section> s;

    rpx::Section text;
    text.index = 1; text.type = 1; text.flags = 0x6; text.addr = 0x02000100; text.size = 8;
    text.data = {0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF};
    s.push_back(text);

    rpx::Section data;
    data.index = 2; data.type = 1; data.flags = 0x3; data.addr = 0x10290000; data.size = 16;
    data.data.assign(64, 0);  // sh_size stays 16: a compressed section decompresses to 64 bytes
    s.push_back(data);

    // .symtab: symbol 0 null, 1 -> .text 0x02000100, 2 -> .data 0x10290000
    rpx::Section symtab;
    symtab.index = 3; symtab.type = 2; symtab.flags = 0; symtab.size = 48; symtab.link = 4; symtab.entsize = 16;
    for (int i = 0; i < 3; i++) {
        put_be32(symtab.data, 0);                     // st_name
        put_be32(symtab.data, i == 1 ? 0x02000100 : (i == 2 ? 0x10290000 : 0));  // st_value
        put_be32(symtab.data, 0);                     // st_size
        symtab.data.push_back(0); symtab.data.push_back(0);
        symtab.data.push_back(0); symtab.data.push_back(0);
    }
    s.push_back(symtab);

    rpx::Section strtab;
    strtab.index = 4; strtab.type = 3; strtab.flags = 0; strtab.data = {0};
    s.push_back(strtab);

    // .rela.data: two good ADDR32, one bad symbol, one out of range, one misaligned, one unsupported type.
    auto rela_entry = [](std::vector<uint8_t>& v, uint32_t site, uint32_t sym, uint32_t type, int32_t addend) {
        put_be32(v, site);
        put_be32(v, (sym << 8) | type);
        put_be32(v, (uint32_t)addend);
    };
    rpx::Section rela;
    rela.index = 5; rela.type = 4; rela.flags = 0; rela.link = 3; rela.info = 2; rela.entsize = 12;
    rela_entry(rela.data, 0x10290000, 1, 1, 0);        // -> 0x02000100
    rela_entry(rela.data, 0x10290028, 1, 1, 0);        // beyond sh_size, inside the decompressed extent
    rela_entry(rela.data, 0x10290004, 2, 1, 0x20);     // -> 0x10290020
    rela_entry(rela.data, 0x10290008, 999, 1, 0);      // bad symbol -> skipped
    rela_entry(rela.data, 0xDEADBEEF, 1, 1, 0);        // out of range -> skipped
    rela_entry(rela.data, 0x1029000A, 1, 1, 0);        // misaligned -> skipped
    rela_entry(rela.data, 0x1029000C, 1, 4, 0);        // unsupported type -> unsupported
    s.push_back(rela);

    // .rela.text: must be ignored entirely (instructions are already translated).
    rpx::Section rela_text;
    rela_text.index = 6; rela_text.type = 4; rela_text.flags = 0; rela_text.link = 3; rela_text.info = 1;
    rela_text.entsize = 12;
    rela_entry(rela_text.data, 0x02000100, 1, 1, 0);
    s.push_back(rela_text);

    return s;
}

}  // namespace

int main() {
    Guest guest;
    const auto sections = build_sections();
    auto writer = [&guest](uint32_t addr, uint32_t value) { guest.write32(addr, value); };

    const rpx::RelocStats first = rpx::apply_data_relocations(sections, 0x02000000, 0x02000108, writer);
    check("applied count", first.applied == 3, std::to_string(first.applied).c_str());
    check("skipped count", first.skipped == 3, std::to_string(first.skipped).c_str());
    check("unsupported count", first.unsupported == 1, std::to_string(first.unsupported).c_str());

    check("text symbol resolved", guest.read32(0x10290000) == 0x02000100);
    check("data symbol + addend resolved", guest.read32(0x10290004) == 0x10290020);
    check("beyond sh_size applied", guest.read32(0x10290028) == 0x02000100);
    check("bad symbol left alone", guest.read32(0x10290008) == 0);
    check("misaligned site left alone", guest.read32(0x1029000A) == 0);  // reads 0x000A's word
    check("unsupported type left alone", guest.read32(0x1029000C) == 0);
    check("big-endian byte order", guest.data[0] == 0x02 && guest.data[1] == 0x00 &&
                                   guest.data[2] == 0x01 && guest.data[3] == 0x00);

    // Idempotency: a second pass (e.g. reloading the image) must not stack values.
    guest.data.assign(0x10000, 0);
    const rpx::RelocStats second = rpx::apply_data_relocations(sections, 0x02000000, 0x02000108, writer);
    check("second pass same stats", second.applied == first.applied && second.skipped == first.skipped &&
                                    second.unsupported == first.unsupported);
    check("second pass same values", guest.read32(0x10290000) == 0x02000100 &&
                                     guest.read32(0x10290004) == 0x10290020);

    if (g_failures) {
        std::printf("RELOC TEST FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("RELOC TEST OK\n");
    return 0;
}
