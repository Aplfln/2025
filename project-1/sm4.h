// include/sm4.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SM4_BLOCK_BYTES 16
#define SM4_KEY_BYTES   16

    typedef struct {
        uint32_t rk[32];
    } sm4_key_t;

    // key schedule
    void sm4_setkey_enc(sm4_key_t* ks, const uint8_t key[SM4_KEY_BYTES]);
    void sm4_setkey_dec(sm4_key_t* ks, const uint8_t key[SM4_KEY_BYTES]);

    // block encrypt - reference implementation
    void sm4_encrypt_block_ref(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]);

    // ECB helpers (multi-block, len must be multiple of 16)
    int sm4_ecb_encrypt_ref(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len);
    int sm4_ecb_decrypt_ref(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len);

    // Placeholder for other backends (ttable / aesni) to be provided later
    void sm4_encrypt_block_ttable(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]);
    void sm4_encrypt_block_aesni(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]);

#ifdef __cplusplus
}
#endif
