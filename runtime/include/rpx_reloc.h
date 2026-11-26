/* Data-section relocations for Wii U RPX images.
 *
 * The loader copies sections into guest memory, but an RPX stores pointers in
 * .rodata/.data/.fexports as zeros plus ELF RELA records. Without applying them,
 * every function pointer and string pointer in a data table reads as 0 or as a
 * link-time placeholder.
 *
 * Two ranges matter and they are not the same thing:
 *   - the recompiled instruction range (code_lo..code_hi) must never be patched,
 *     because those bytes have already been turned into C;
 *   - everything else that was loaded (including sections that carry the
 *     SHF_EXECINSTR flag, such as an export table full of function pointers) is
 *     a legal relocation target.
 */
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace rpx {

struct Section {
    uint32_t index = 0;
    uint32_t type = 0;      // sh_type
    uint32_t flags = 0;     // sh_flags
    uint32_t addr = 0;      // sh_addr (guest)
    uint32_t offset = 0;    // sh_offset (file)
    uint32_t size = 0;      // sh_size (file size for compressed sections)
    uint32_t link = 0;      // sh_link (symbol table for RELA, string table for SYMTAB)
    uint32_t info = 0;      // sh_info
    uint32_t entsize = 0;   // sh_entsize
    std::vector<uint8_t> data;  // loaded payload: the *decompressed* bytes when the section was compressed
};

struct RelocStats {
    uint32_t applied = 0;
    uint32_t skipped = 0;       // out of range, misaligned, or no target section
    uint32_t unsupported = 0;   // relocation type we do not implement for data
};

using Write32 = std::function<void(uint32_t addr, uint32_t value)>;

// Applies R_PPC_ADDR32 relocations to loaded data. `write` receives the guest
// address and the resolved value. Applying twice is a no-op in effect: the value
// is always symbol+addend, never the previous contents, so re-loading cannot
// stack relocations.
RelocStats apply_data_relocations(const std::vector<Section>& sections, uint32_t code_lo, uint32_t code_hi,
                                  const Write32& write);

}  // namespace rpx
