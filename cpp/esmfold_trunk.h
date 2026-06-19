#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct esmfold_trunk_model;

// Load a trunk GGUF produced by convert_esmfold_trunk.py. Returns nullptr on failure.
esmfold_trunk_model * esmfold_trunk_load(const std::string & path);
void                  esmfold_trunk_free(esmfold_trunk_model * model);
int                   esmfold_trunk_c_s(const esmfold_trunk_model * model);
int                   esmfold_trunk_c_z(const esmfold_trunk_model * model);

// Run the folding trunk (num_recycles=0 mode).
// s_s_0   : input sequence state, flat [n_tokens * seq_state_dim], token-major.
// residx  : residue index per token (usually 0,1,2,…,L-1), length n_tokens.
// s_s_out : output sequence state  [n_tokens * seq_state_dim]
// s_z_out : output pair state       [n_tokens * n_tokens * pair_state_dim], (i,j,d)-major
// Returns false on error.
bool esmfold_trunk_eval(
        esmfold_trunk_model       * model,
        const std::vector<float>  & s_s_0,
        const std::vector<int32_t>& residx,
        std::vector<float>        & s_s_out,
        std::vector<float>        & s_z_out,
        int                         n_blocks_override = -1); // -1 = all 48
