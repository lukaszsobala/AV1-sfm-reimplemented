/*
 * av1of_inspect_shim.c
 *
 *   Created by Julien Zouein on 22/06/2026.
 *   Copyright © 2026 Sigmedia.tv. All rights reserved.
 *   Copyright © 2026 Julien Zouein (zoueinj@tcd.ie)
 * --------------------------------------------------------------------------
 *
 * In-memory AV1 block-metadata extraction shim around patched libdav1d.
 *
 * This moves the per-frame array construction out of the Python callback and
 * into C: the whole file is decoded in a single call (so Python can release the
 * GIL for the entire multi-threaded decode), and the dav1d inspection callback
 * — which fires on dav1d's worker threads — transforms each frame's 4x4
 * `refmvs_block` grid directly into the final output layout WITHOUT touching the
 * Python interpreter. This removes the GIL serialization that otherwise pins the
 * pipeline near single-thread speed.
 *
 * Output per frame (matches src/modules/dav1d_inspect.py / the AOM JSON):
 *   motion_vectors : int16 [blk_h][blk_w][4] = (mv0_x, mv0_y, mv1_x, mv1_y), 1/8-pel
 *   reference_map  : int16 [blk_h][blk_w][2] = (ref0, ref1)
 *   block_map      : uint8 [blk_h][blk_w]    = AOM BLOCK_* enum
 *
 * Linked directly against our patched libdav1d with an rpath relative to the
 * shim, so it binds to that specific library (and is not interposed by another
 * libdav1d in the process, e.g. the one OpenCV bundles). See setup.sh:
 *   macOS:  cc ... -o libav1of_inspect.dylib shim.c \
 *               dav1d/build/src/libdav1d.7.dylib -Wl,-rpath,@loader_path/dav1d/build/src
 *   Linux:  cc ... -o libav1of_inspect.so shim.c \
 *               -L dav1d/build/src -ldav1d -Wl,-rpath,'$ORIGIN/dav1d/build/src'
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>

#include <dav1d/dav1d.h>

// dav1d `enum BlockSize` (index) -> AOM `BLOCK_*` enum value, matched by WxH
// name (BS_128x128=0..BS_4x4=21 -> BLOCK_4X4=0..BLOCK_64X16=21).
static const uint8_t BS_DAV1D_TO_AOM[22] = {
    15, 14, 13, 12, 11, 21, 10, 9, 8, 19, 20, 7, 6, 5, 17, 18, 4, 3, 2, 16, 1, 0,
};

// INVALID_MV component sentinel (refmvs.h INVALID_MV == 0x80008000).
#define INVALID_MV_COMPONENT ((int16_t) 0x8000)

typedef struct {
    unsigned decode_seq;
    unsigned frame_offset;
    int frame_type;
    int width, height;
    int blk_w, blk_h;
    int8_t refidx[7];
    unsigned refpoc[7];
    int16_t *motion_vectors; // blk_h * blk_w * 4
    int16_t *reference_map;  // blk_h * blk_w * 2
    uint8_t *block_map;      // blk_h * blk_w
} av1of_frame;

typedef struct {
    av1of_frame *frames;
    int n, cap;
    pthread_mutex_t lock;
    int alloc_error;
} av1of_handle;

// One packed refmvs_block is 12 bytes:
//   int16 mv0_y, mv0_x, mv1_y, mv1_x; int8 ref0, ref1; uint8 bs, mf
#define REFMVS_BLOCK_SZ 12

static void inspect_cb(void *cookie, const Dav1dInspectData *d) {
    av1of_handle *const h = cookie;
    const int W = d->blk_w, H = d->blk_h;
    const size_t n = (size_t) W * H;

    av1of_frame f;
    memset(&f, 0, sizeof(f));
    f.decode_seq = d->decode_seq;
    f.frame_offset = d->frame_offset;
    f.frame_type = d->frame_type;
    f.width = d->width;
    f.height = d->height;
    f.blk_w = W;
    f.blk_h = H;
    memcpy(f.refidx, d->refidx, sizeof(f.refidx));
    memcpy(f.refpoc, d->refpoc, sizeof(f.refpoc));

    f.motion_vectors = malloc(n * 4 * sizeof(int16_t));
    f.reference_map = malloc(n * 2 * sizeof(int16_t));
    f.block_map = malloc(n);
    if (!f.motion_vectors || !f.reference_map || !f.block_map) {
        free(f.motion_vectors);
        free(f.reference_map);
        free(f.block_map);
        pthread_mutex_lock(&h->lock);
        h->alloc_error = 1;
        pthread_mutex_unlock(&h->lock);
        return;
    }

    const uint8_t *const blocks = d->blocks; // NULL for intra/key frames
    const ptrdiff_t stride = d->blk_stride;  // in 12-byte records

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const size_t i = (size_t) y * W + x;
            int16_t *const mv = &f.motion_vectors[i * 4];
            int16_t *const ref = &f.reference_map[i * 2];

            if (!blocks) {
                // Intra / key frame: no motion field. Mirror AOM's intra
                // convention (zero MVs, ref = [0 = INTRA, -1 = none]).
                mv[0] = mv[1] = mv[2] = mv[3] = 0;
                ref[0] = 0;
                ref[1] = -1;
                f.block_map[i] = 0;
                continue;
            }

            const uint8_t *const rec =
                blocks + ((size_t) y * stride + x) * REFMVS_BLOCK_SZ;
            int16_t mv0_y, mv0_x, mv1_y, mv1_x;
            memcpy(&mv0_y, rec + 0, 2);
            memcpy(&mv0_x, rec + 2, 2);
            memcpy(&mv1_y, rec + 4, 2);
            memcpy(&mv1_x, rec + 6, 2);

            // Normalise INVALID_MV (0x8000) components to 0, as AOM does.
            if (mv0_x == INVALID_MV_COMPONENT) mv0_x = 0;
            if (mv0_y == INVALID_MV_COMPONENT) mv0_y = 0;
            if (mv1_x == INVALID_MV_COMPONENT) mv1_x = 0;
            if (mv1_y == INVALID_MV_COMPONENT) mv1_y = 0;

            // Channel order [x, y] per MV to match the .flo / AOM convention.
            mv[0] = mv0_x;
            mv[1] = mv0_y;
            mv[2] = mv1_x;
            mv[3] = mv1_y;
            ref[0] = (int8_t) rec[8];
            ref[1] = (int8_t) rec[9];

            const uint8_t bs = rec[10];
            f.block_map[i] = BS_DAV1D_TO_AOM[bs < 22 ? bs : 0];
        }
    }

    pthread_mutex_lock(&h->lock);
    if (h->n == h->cap) {
        const int cap = h->cap ? h->cap * 2 : 64;
        av1of_frame *const grown =
            realloc(h->frames, (size_t) cap * sizeof(av1of_frame));
        if (!grown) {
            h->alloc_error = 1;
            pthread_mutex_unlock(&h->lock);
            free(f.motion_vectors);
            free(f.reference_map);
            free(f.block_map);
            return;
        }
        h->frames = grown;
        h->cap = cap;
    }
    h->frames[h->n++] = f;
    pthread_mutex_unlock(&h->lock);
}

static void drain(Dav1dContext *const c, Dav1dPicture *const pic) {
    for (;;) {
        memset(pic, 0, sizeof(*pic));
        const int r = dav1d_get_picture(c, pic);
        if (r < 0) break; // DAV1D_ERR(EAGAIN) or genuine error: stop draining
        dav1d_picture_unref(pic);
    }
}

static int cmp_decode_seq(const void *a, const void *b) {
    const unsigned sa = ((const av1of_frame *) a)->decode_seq;
    const unsigned sb = ((const av1of_frame *) b)->decode_seq;
    return (sa > sb) - (sa < sb);
}

// Decode an entire IVF/AV1 file in memory, transforming every frame's block
// metadata in C. Returns 0 on success (*out owns the results), <0 on error.
int av1of_decode(const char *const path, const int n_threads,
                 av1of_handle **const out) {
    *out = NULL;
    av1of_handle *const h = calloc(1, sizeof(*h));
    if (!h) return -1;
    pthread_mutex_init(&h->lock, NULL);

    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = n_threads;
    s.inspect_cb = inspect_cb;
    s.inspect_cookie = h;

    Dav1dContext *c = NULL;
    if (dav1d_open(&c, &s) < 0) {
        pthread_mutex_destroy(&h->lock);
        free(h);
        return -2;
    }

    FILE *const fp = fopen(path, "rb");
    if (!fp) {
        dav1d_close(&c);
        pthread_mutex_destroy(&h->lock);
        free(h);
        return -3;
    }

    int rc = 0;
    uint8_t hdr[32];
    if (fread(hdr, 1, 32, fp) != 32 || memcmp(hdr, "DKIF", 4) != 0) {
        rc = -4;
        goto done;
    }

    Dav1dPicture pic;
    uint8_t pkt_hdr[12];
    while (fread(pkt_hdr, 1, 12, fp) == 12) {
        const uint32_t sz = (uint32_t) pkt_hdr[0] | ((uint32_t) pkt_hdr[1] << 8) |
                            ((uint32_t) pkt_hdr[2] << 16) |
                            ((uint32_t) pkt_hdr[3] << 24);
        if (sz == 0) break;

        Dav1dData data;
        uint8_t *const buf = dav1d_data_create(&data, sz);
        if (!buf) {
            rc = -5;
            goto done;
        }
        if (fread(buf, 1, sz, fp) != sz) {
            dav1d_data_unref(&data);
            rc = -6;
            goto done;
        }

        while (data.sz > 0) {
            const int r = dav1d_send_data(c, &data);
            if (r < 0 && r != DAV1D_ERR(EAGAIN)) {
                dav1d_data_unref(&data);
                rc = -7;
                goto done;
            }
            drain(c, &pic);
        }
    }
    drain(c, &pic); // flush buffered frames at end of stream

done:
    fclose(fp);
    dav1d_close(&c); // joins worker threads: all callbacks have fired

    if (rc == 0 && h->alloc_error) rc = -8;
    if (rc != 0) {
        // Free any partial results.
        for (int i = 0; i < h->n; i++) {
            free(h->frames[i].motion_vectors);
            free(h->frames[i].reference_map);
            free(h->frames[i].block_map);
        }
        free(h->frames);
        pthread_mutex_destroy(&h->lock);
        free(h);
        return rc;
    }

    qsort(h->frames, h->n, sizeof(av1of_frame), cmp_decode_seq);
    *out = h;
    return 0;
}

int av1of_num_frames(const av1of_handle *const h) { return h->n; }

const av1of_frame *av1of_get_frame(const av1of_handle *const h, const int i) {
    return (i >= 0 && i < h->n) ? &h->frames[i] : NULL;
}

void av1of_free(av1of_handle *const h) {
    if (!h) return;
    for (int i = 0; i < h->n; i++) {
        free(h->frames[i].motion_vectors);
        free(h->frames[i].reference_map);
        free(h->frames[i].block_map);
    }
    free(h->frames);
    pthread_mutex_destroy(&h->lock);
    free(h);
}
