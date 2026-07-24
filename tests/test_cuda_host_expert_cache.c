/* The CUDA host-RAM expert cache on machines it does not fit.
 *
 * The cache is only an optimization, so expert loading has to keep working
 * on any machine, and the user has to be told what the cache did. There are
 * three outcomes:
 *
 *   active    it started, and printed how many experts it pinned;
 *   refused   it stayed off, and printed why plus the SSD fallback;
 *   bypassed  it does not apply to this model shape, and stays quiet.
 *
 * ds4_gpu_stream_expert_cache_begin_selected_load() must succeed in all
 * three. It may only fail for a VRAM buffer no GPU can hold, and there it
 * prints the failed allocation and returns 0 for the engine to handle
 * instead of aborting. The stderr capture follows
 * tests/test_engine_mgpu_refusal.c, since the message is the point.
 *
 * The model is a small synthetic buffer, so there is no GGUF and no large
 * working set. Host RAM is varied through the ratio of the RAM budget to
 * the per-expert slab size, which is what the sizing code looks at. VRAM is
 * varied by asking for a layer buffer that is far too big. */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
static int checks;

/* Small enough that even a "large" cache is a few MiB, big enough that the
 * slab arithmetic still means something. MAX_SCALE covers the boosted-layer
 * case, which asks for larger experts than the slab. */
enum {
    N_LAYER    = 8,
    N_TOTAL    = 32,
    N_SELECTED = 6,
    MAX_SCALE  = 2,
    GATE_BYTES = 48 * 1024,
    DOWN_BYTES = 48 * 1024
};
#define SLAB_BYTES  ((uint64_t)(2 * GATE_BYTES + DOWN_BYTES))
#define MODEL_BYTES ((uint64_t)N_TOTAL * SLAB_BYTES * MAX_SCALE)

/* Larger than any machine's RAM, so "the planner budget does not fit" is
 * reached on every host this test can run on. */
#define UNFITTABLE_SLAB (4ull * 1024ull * 1024ull * 1024ull * 1024ull)

static char *g_model;

static ds4_gpu_stream_expert_table table_for(uint32_t layer, uint64_t scale) {
    ds4_gpu_stream_expert_table t;
    memset(&t, 0, sizeof(t));
    t.model_map         = g_model;
    t.model_size        = MODEL_BYTES;
    t.layer             = layer;
    t.n_total_expert    = N_TOTAL;
    t.gate_expert_bytes = (uint64_t)GATE_BYTES * scale;
    t.down_expert_bytes = (uint64_t)DOWN_BYTES * scale;
    t.gate_offset       = 0;
    t.up_offset         = (uint64_t)N_TOTAL * t.gate_expert_bytes;
    t.down_offset       = 2u * t.up_offset;
    return t;
}

/* stderr capture: the messages are half of what is being tested. */
static const char *g_cap_path = "/tmp/ds4_host_expert_cache_stderr.log";
static int g_saved_stderr = -1;

static void cap_begin(void) {
    fflush(stderr);
    g_saved_stderr = dup(fileno(stderr));
    if (g_saved_stderr < 0 || !freopen(g_cap_path, "w+", stderr)) {
        printf("FAIL: cannot capture stderr\n");
        exit(1);
    }
}

static char *cap_end(void) {
    fflush(stderr);
    if (g_saved_stderr >= 0) {
        dup2(g_saved_stderr, fileno(stderr));
        close(g_saved_stderr);
        g_saved_stderr = -1;
    }
    FILE *f = fopen(g_cap_path, "rb");
    if (!f) return strdup("");
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)(n > 0 ? n : 0) + 1);
    if (!buf) { fclose(f); return strdup(""); }
    const size_t r = n > 0 ? fread(buf, 1, (size_t)n, f) : 0;
    fclose(f);
    buf[r] = '\0';
    return buf;
}

static int count_occurrences(const char *hay, const char *needle) {
    int n = 0;
    for (const char *p = strstr(hay, needle); p; p = strstr(p + 1, needle)) n++;
    return n;
}

static void expect(int cond, const char *what, const char *log) {
    checks++;
    if (cond) return;
    failures++;
    fprintf(stderr,
            "FAIL: %s\n--- captured stderr ---\n%s-----------------------\n",
            what, log);
}

/* Bring the cache to a known state and push one engine configuration, in the
 * order ds4_engine_open() uses: streaming on, slot budget, slab size class. */
static void configure(const char *env, uint32_t budget, uint64_t slab, int glm) {
    ds4_gpu_set_ssd_streaming(false);   /* frees the arena and the maps */
    ds4_gpu_set_glm_model(glm ? true : false);
    if (env) setenv("DS4_CUDA_HOST_EXPERT_CACHE_GB", env, 1);
    else     unsetenv("DS4_CUDA_HOST_EXPERT_CACHE_GB");
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(budget);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(slab);
}

static int load_passes(uint64_t scale, int passes, uint32_t layers) {
    int32_t ids[N_SELECTED];
    for (int p = 0; p < passes; p++) {
        for (uint32_t l = 0; l < layers; l++) {
            for (int i = 0; i < N_SELECTED; i++) {
                ids[i] = (int32_t)((l * 3u + (uint32_t)i * 5u) % N_TOTAL);
            }
            const ds4_gpu_stream_expert_table t = table_for(l, scale);
            if (!ds4_gpu_stream_expert_cache_begin_selected_load(&t, ids,
                                                                 N_SELECTED)) {
                return 0;
            }
        }
    }
    return 1;
}

/* Two passes over the same selections, so pass 2 finds what pass 1 loaded.
 * Teardown prints a hit/miss line only if the cache was used at all, which
 * is how "was the cache used" becomes visible from outside. */
static char *run_scenario(const char *env, uint32_t budget, uint64_t slab,
                          int glm, uint64_t scale, int *loads_ok,
                          uint32_t *configured) {
    cap_begin();
    configure(env, budget, slab, glm);
    if (configured) *configured = ds4_gpu_stream_expert_cache_configured_count();
    *loads_ok = load_passes(scale, 2, N_LAYER);
    ds4_gpu_set_ssd_streaming(false);   /* flush: prints the hit/miss line */
    return cap_end();
}

static void scenario_active(void) {
    int ok = 0;
    uint32_t configured = 0;
    char *log = run_scenario(NULL, 64, SLAB_BYTES, 0, 1, &ok, &configured);
    expect(ok, "ample RAM: every selected load succeeds", log);
    expect(strstr(log, "GiB pinned") != NULL,
           "ample RAM: reports how much it pinned", log);
    expect(strstr(log, "hits,") != NULL,
           "ample RAM: the cache was consulted", log);
    expect(strstr(log, "using SSD reads") == NULL,
           "ample RAM: does not claim a fallback", log);
    expect(configured == 64,
           "ample RAM: the planner budget is reported back", log);
    free(log);
}

static void scenario_cache_smaller_than_one_call(void) {
    /* Fewer slots than one layer selects: take_slot() runs out of unprotected
     * victims and the remaining experts fall back to an uncached read. */
    int ok = 0;
    char *log = run_scenario(NULL, 2, SLAB_BYTES, 0, 1, &ok, NULL);
    expect(ok, "cache smaller than one call: loads still succeed", log);
    expect(strstr(log, "GiB pinned") != NULL,
           "cache smaller than one call: the cache still comes up", log);
    free(log);
}

static void scenario_env_budget(void) {
    int ok = 0;
    char *log = run_scenario("1", 64, SLAB_BYTES, 0, 1, &ok, NULL);
    expect(ok, "env budget: loads succeed", log);
    expect(strstr(log, "GiB pinned") != NULL,
           "env budget: honored and reported", log);
    free(log);
}

static void scenario_opt_out(void) {
    int ok = 0;
    char *log = run_scenario("0", 64, SLAB_BYTES, 0, 1, &ok, NULL);
    expect(ok, "opt-out: loads succeed on the plain SSD path", log);
    expect(strstr(log, "GiB pinned") == NULL,
           "opt-out: nothing is pinned", log);
    expect(strstr(log, "hits,") == NULL,
           "opt-out: the cache is never consulted", log);
    free(log);
}

static void scenario_invalid_env(void) {
    int ok = 0;
    char *log = run_scenario("not-a-size", 64, SLAB_BYTES, 0, 1, &ok, NULL);
    expect(ok, "invalid env: loads succeed", log);
    expect(strstr(log, "ignoring invalid DS4_CUDA_HOST_EXPERT_CACHE_GB") != NULL,
           "invalid env: says it is ignoring the value", log);
    expect(strstr(log, "GiB pinned") != NULL,
           "invalid env: falls back to the planner budget", log);
    free(log);
}

static void scenario_ram_below_one_expert(void) {
    /* No machine can hold a single expert of this shape: the cache must
     * refuse in one line, name the SSD fallback, and never retry per load. */
    int ok = 0;
    char *log = run_scenario(NULL, 64, UNFITTABLE_SLAB, 0, 1, &ok, NULL);
    expect(ok, "RAM below one expert: loads still succeed", log);
    expect(strstr(log, "using SSD reads") != NULL,
           "RAM below one expert: explains the fallback", log);
    expect(count_occurrences(log, "using SSD reads") == 1,
           "RAM below one expert: the refusal is sticky, not per load", log);
    expect(strstr(log, "GiB pinned") == NULL,
           "RAM below one expert: nothing is pinned", log);
    free(log);
}

static void scenario_boosted_layer_bypass(void) {
    /* Mixed-precision layers whose experts exceed the slab size class keep the
     * plain SSD path, as the engine's startup log promises. */
    int ok = 0;
    char *log = run_scenario(NULL, 64, SLAB_BYTES, 0, MAX_SCALE, &ok, NULL);
    expect(ok, "boosted layer: loads succeed", log);
    expect(strstr(log, "GiB pinned") != NULL,
           "boosted layer: the arena still exists for the other layers", log);
    expect(strstr(log, "hits,") == NULL,
           "boosted layer: bypasses the cache entirely", log);
    free(log);
}

static void scenario_glm(void) {
    int ok = 0;
    char *log = run_scenario(NULL, 64, SLAB_BYTES, 1, 1, &ok, NULL);
    expect(ok, "GLM: loads succeed", log);
    expect(strstr(log, "GiB pinned") == NULL,
           "GLM: the cache is not allocated", log);
    expect(strstr(log, "hits,") == NULL,
           "GLM: the cache is not consulted", log);
    free(log);
}

static void scenario_vram_too_small(void) {
    /* No GPU can hold this layer's selected experts.  The load must name the
     * failed allocation and return 0 so the engine can react, not abort. */
    size_t vram_free = 0, vram_total = 0;
    (void)cudaMemGetInfo(&vram_free, &vram_total);
    const uint64_t floor_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    const uint64_t huge =
        4ull * ((uint64_t)vram_total > floor_bytes ? (uint64_t)vram_total
                                                   : floor_bytes);

    cap_begin();
    configure(NULL, 64, SLAB_BYTES, 0);
    ds4_gpu_stream_expert_table t = table_for(0, huge / GATE_BYTES);
    /* ranges_valid() only checks arithmetic, and the VRAM allocation fails
     * long before anything dereferences the map, so the small model buffer
     * stays safe here. */
    t.model_size = UINT64_MAX / 4u;
    int32_t ids[N_SELECTED];
    for (int i = 0; i < N_SELECTED; i++) ids[i] = i;
    const int rc =
        ds4_gpu_stream_expert_cache_begin_selected_load(&t, ids, N_SELECTED);
    ds4_gpu_set_ssd_streaming(false);
    char *log = cap_end();
    expect(rc == 0, "VRAM too small: the load reports failure", log);
    expect(strstr(log, "allocation failed") != NULL,
           "VRAM too small: names the failed allocation", log);
    free(log);

    /* And the engine can carry on afterwards with a layer that does fit. */
    cap_begin();
    configure(NULL, 64, SLAB_BYTES, 0);
    const int ok = load_passes(1, 1, 1);
    ds4_gpu_set_ssd_streaming(false);
    char *log2 = cap_end();
    expect(ok, "VRAM too small: a fitting layer still loads afterwards", log2);
    free(log2);
}

static void scenario_churn(void) {
    /* Far more (layer, expert) pairs than slots, so nearly every load evicts.
     * This is where a real streaming decode spends all of its time. */
    enum { STEPS = 400 };
    int ok = 1;
    uint32_t seed = 12345u;
    int32_t ids[N_SELECTED];

    cap_begin();
    configure(NULL, 8, SLAB_BYTES, 0);
    for (int step = 0; step < STEPS && ok; step++) {
        for (int i = 0; i < N_SELECTED; i++) {
            seed = seed * 1103515245u + 12345u;
            ids[i] = (int32_t)((seed >> 16) % N_TOTAL);
        }
        const ds4_gpu_stream_expert_table t =
            table_for((uint32_t)step % N_LAYER, 1);
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(&t, ids,
                                                             N_SELECTED);
    }
    ds4_gpu_set_ssd_streaming(false);
    char *log = cap_end();
    expect(ok, "churn: every evicting load succeeds", log);
    expect(strstr(log, "hits,") != NULL, "churn: the cache stayed engaged", log);
    free(log);
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_cuda_host_expert_cache: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping (no CUDA device)\n");
        return 0;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "FAIL: ds4_gpu_init\n");
        return 1;
    }
    g_model = (char *)malloc((size_t)MODEL_BYTES);
    if (!g_model) { perror("malloc"); return 1; }
    /* Distinct bytes per position, so a mis-addressed copy would be visible
     * to anything that reads these back. */
    for (uint64_t i = 0; i < MODEL_BYTES; i++) g_model[i] = (char)(i * 31u + 7u);

    scenario_active();
    scenario_cache_smaller_than_one_call();
    scenario_env_budget();
    scenario_opt_out();
    scenario_invalid_env();
    scenario_ram_below_one_expert();
    scenario_boosted_layer_bypass();
    scenario_glm();
    scenario_vram_too_small();
    scenario_churn();

    unsetenv("DS4_CUDA_HOST_EXPERT_CACHE_GB");
    ds4_gpu_cleanup();
    free(g_model);
    (void)unlink(g_cap_path);

    fprintf(stderr,
            "test_cuda_host_expert_cache: %d/%d checks passed (%d failed)\n",
            checks - failures, checks, failures);
    return failures != 0;
}
