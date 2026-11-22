#include "rpx_reloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rpx {
namespace {

constexpr uint32_t SHT_SYMTAB = 2;
constexpr uint32_t SHT_RELA = 4;
constexpr uint32_t SHT_NOBITS = 8;
constexpr uint32_t R_PPC_ADDR32 = 1;

uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

int32_t be32s(const uint8_t* p) { return (int32_t)be32(p); }

struct SymbolTable {
    const Section* section = nullptr;
    std::vector<uint32_t> values;
};

const SymbolTable* table_for(const std::vector<SymbolTable>& tables, uint32_t link) {
    for (const SymbolTable& t : tables)
        if (t.section->index == link) return &t;
    return nullptr;
}

// The guest extent of a section is the size of what was actually loaded: a
// compressed .rodata decompresses to far more bytes than its sh_size (the
// compressed file size), and relocations target those bytes.
uint32_t mapped_extent(const Section& s) { return s.data.empty() ? s.size : (uint32_t)s.data.size(); }

const Section* section_by_addr(const std::vector<Section>& sections, uint32_t addr, uint32_t width) {
    for (const Section& s : sections) {
        if (s.type == SHT_NOBITS || s.addr == 0) continue;
        const uint32_t extent = mapped_extent(s);
        if (extent == 0) continue;
        if (addr >= s.addr && (uint64_t)addr + width <= (uint64_t)s.addr + extent) return &s;
    }
    return nullptr;
}

bool in_code(uint32_t addr, uint32_t lo, uint32_t hi) { return addr >= lo && addr < hi; }

}  // namespace

RelocStats apply_data_relocations(const std::vector<Section>& sections, uint32_t code_lo, uint32_t code_hi,
                                  const Write32& write) {
    RelocStats stats;
    const bool verbose = std::getenv("RPX_RELOC_VERBOSE") != nullptr;
    int shown = 0;

    std::vector<SymbolTable> tables;
    for (const Section& s : sections) {
        if (s.type != SHT_SYMTAB || s.data.size() < 16) continue;
        SymbolTable t;
        t.section = &s;
        for (size_t off = 0; off + 16 <= s.data.size(); off += 16)
            t.values.push_back(be32(&s.data[off + 4]));  // st_value
        tables.push_back(std::move(t));
    }

    for (const Section& rela : sections) {
        if (rela.type != SHT_RELA || rela.data.empty()) continue;

        // A RELA section whose sites live in the recompiled range describes
        // instructions that already became C. Skipped wholesale.
        if (rela.data.size() >= 12 && in_code(be32(&rela.data[0]), code_lo, code_hi)) continue;

        const SymbolTable* symtab = table_for(tables, rela.link);
        if (!symtab) {
            stats.skipped++;
            if (verbose)
                std::fprintf(stderr, "[reloc] section %u: no symbol table for link %u\n", rela.index, rela.link);
            continue;
        }

        const uint32_t entry = rela.entsize ? rela.entsize : 12;
        uint32_t sec_applied = 0;
        for (size_t off = 0; off + entry <= rela.data.size(); off += entry) {
            const uint32_t site = be32(&rela.data[off]);
            const uint32_t info = be32(&rela.data[off + 4]);
            const int32_t addend = be32s(&rela.data[off + 8]);
            const uint32_t type = info & 0xFF;
            const uint32_t sym = info >> 8;

            if (type != R_PPC_ADDR32) {
                stats.unsupported++;
                continue;
            }
            if (sym >= symtab->values.size()) {
                stats.skipped++;
                if (verbose && shown++ < 10)
                    std::fprintf(stderr, "[reloc] skip %08X: symbol %u >= %zu\n", site, sym,
                                 symtab->values.size());
                continue;
            }
            const Section* dest = section_by_addr(sections, site, 4);
            if (!dest || in_code(site, code_lo, code_hi) || (site & 3) != 0) {
                stats.skipped++;
                if (verbose && shown++ < 10)
                    std::fprintf(stderr, "[reloc] skip %08X: dest=%s code=%d align=%u\n", site,
                                 dest ? "found" : "none", (int)in_code(site, code_lo, code_hi), site & 3);
                continue;
            }
            const uint32_t value = symtab->values[sym] + (uint32_t)addend;
            write(site, value);
            stats.applied++;
            sec_applied++;
        }
        if (verbose)
            std::fprintf(stderr, "[reloc] section %u: %u applied of %zu entries\n", rela.index, sec_applied,
                         rela.data.size() / entry);
    }
    return stats;
}

}  // namespace rpx
