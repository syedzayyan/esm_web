#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Minimal re-implementation of the ESM2 tokenizer described by the repo's
// tokenizer.json: a character-level "Split" pre-tokenizer feeding a
// WordLevel vocabulary, with a TemplateProcessing post-processor that
// prepends <cls> (id 0) to every sequence (no trailing <eos>).
namespace esm2_tokenizer {

constexpr int32_t CLS_ID = 0;
constexpr int32_t UNK_ID = 3;

// Maps an ASCII character (residue code) to its vocab id, mirroring the
// 33-entry WordLevel vocab in tokenizer.json. Falls back to UNK_ID.
inline int32_t char_to_id(char c) {
    static const std::array<int32_t, 256> table = [] {
        std::array<int32_t, 256> t{};
        t.fill(UNK_ID);
        t[(unsigned char) 'L'] = 4;
        t[(unsigned char) 'A'] = 5;
        t[(unsigned char) 'G'] = 6;
        t[(unsigned char) 'V'] = 7;
        t[(unsigned char) 'S'] = 8;
        t[(unsigned char) 'E'] = 9;
        t[(unsigned char) 'R'] = 10;
        t[(unsigned char) 'T'] = 11;
        t[(unsigned char) 'I'] = 12;
        t[(unsigned char) 'D'] = 13;
        t[(unsigned char) 'P'] = 14;
        t[(unsigned char) 'K'] = 15;
        t[(unsigned char) 'Q'] = 16;
        t[(unsigned char) 'N'] = 17;
        t[(unsigned char) 'F'] = 18;
        t[(unsigned char) 'Y'] = 19;
        t[(unsigned char) 'M'] = 20;
        t[(unsigned char) 'H'] = 21;
        t[(unsigned char) 'W'] = 22;
        t[(unsigned char) 'C'] = 23;
        t[(unsigned char) 'X'] = 24;
        t[(unsigned char) 'B'] = 25;
        t[(unsigned char) 'U'] = 26;
        t[(unsigned char) 'Z'] = 27;
        t[(unsigned char) 'O'] = 28;
        t[(unsigned char) '.'] = 29;
        t[(unsigned char) '-'] = 30;
        return t;
    }();
    return table[(unsigned char) c];
}

// Tokenizes a protein sequence: prepend <cls>, then one token per residue
// character (case-insensitive, lowercase letters are upper-cased first).
inline std::vector<int32_t> encode(const std::string & sequence) {
    std::vector<int32_t> ids;
    ids.reserve(sequence.size() + 1);
    ids.push_back(CLS_ID);
    for (char c : sequence) {
        if (c >= 'a' && c <= 'z') {
            c = (char) (c - 'a' + 'A');
        }
        ids.push_back(char_to_id(c));
    }
    return ids;
}

} // namespace esm2_tokenizer
