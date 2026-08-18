/*
 * wordle_gemini.c
 *
 * Optimal Wordle Solver (Easy Mode)
 *
 * Features:
 * 1. 128-bit Double Hashing in TT and Partition Deduplication (zero collision risk).
 * 2. Fast 1-Ply Greedy Aspiration Seeding for root beta bounds.
 * 3. Fused single-pass candidate loop (dedup + histogram + variance + lb1/ub1).
 * 4. Node-level lb1 fail-soft cutoffs and ub1==lb1 exact resolutions.
 * 5. Inlined 64-bit Introsort and 1-cycle register lower bound branch pruning.
 * 6. Easy-mode disjoint-union bound propagation.
 * 7. Correct per-bucket node count reporting in parallel mode.
 * 8. Decision tree JSON export compatible with strategies.py.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>
#include <strings.h>
#include <stdatomic.h>
#include <math.h>
#include <sys/types.h>
#include <assert.h>
#include <stdarg.h>

#define WORD_LEN 5
#define NUM_SCORES 243
#define EXACT_MATCH 242
#define MAX_SOLVER_DEPTH 32
#define TT_MAX_PROBES 16
#define PTHREAD_STACK_SIZE (8 * 1024 * 1024)

/* -------------------------------------------------------------
 * Logging control
 *
 * The library defaults to WORDLE_LOG_QUIET (silent): it prints nothing unless
 * a caller opts in by raising the level (e.g. wordle_log_set_level(WORDLE_LOG_INFO))
 * and, optionally, redirecting output with wordle_log_set_stream(). The CLI
 * raises the level to WORDLE_LOG_INFO before doing any work.
 * ------------------------------------------------------------- */

enum {
    WORDLE_LOG_QUIET = 0,
    WORDLE_LOG_ERROR = 1,
    WORDLE_LOG_INFO  = 2,
    WORDLE_LOG_DEBUG = 3,
};

static int wordle_log_level = WORDLE_LOG_QUIET;
static FILE *wordle_log_stream = NULL;

static void
wordle_logf(int level, bool is_err, const char *fmt, ...)
{
    FILE *fp;
    va_list ap;

    if (level > wordle_log_level) {
        return;
    }
    fp = (wordle_log_stream != NULL) ? wordle_log_stream : (is_err ? stderr : stdout);
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fflush(fp);
}

void
wordle_log_set_level(int level)
{
    wordle_log_level = level;
}

void
wordle_log_set_stream(FILE *stream)
{
    wordle_log_stream = stream;
}

int
wordle_log_get_level(void)
{
    return wordle_log_level;
}

#define WORDLE_ERROR(...) wordle_logf(WORDLE_LOG_ERROR, true, __VA_ARGS__)
#define WORDLE_INFO(...)  wordle_logf(WORDLE_LOG_INFO, false, __VA_ARGS__)
#define WORDLE_DEBUG(...) wordle_logf(WORDLE_LOG_DEBUG, false, __VA_ARGS__)

typedef struct {
    char word[WORD_LEN + 1];
} Word;

/* -------------------------------------------------------------
 * 128-Bit Lock-Free Shared Transposition Table
 * ------------------------------------------------------------- */

#define TT_EMPTY_HASH UINT64_MAX

typedef struct {
    _Atomic uint64_t hash1;
    _Atomic uint64_t hash2;
    _Atomic uint32_t size;
    _Atomic uint32_t exact_cost;
    _Atomic uint32_t proven_lower_bound;
    _Atomic uint32_t best_guess;
} SharedTTEntry;

typedef struct {
    SharedTTEntry *entries;
    uint64_t mask;
} SharedTT;

typedef struct {
    uint64_t hash1;
    uint64_t hash2;
    uint32_t size;
    uint32_t exact_cost;
    uint32_t proven_lower_bound;
    uint32_t best_guess;
} TTEntry;

typedef struct {
    TTEntry *entries;
    uint64_t mask;
} TT;

static int shared_tt_init(SharedTT *stt, uint64_t capacity_pow2);
static void shared_tt_free(SharedTT *stt);

typedef struct {
    Word *targets;
    uint32_t num_targets;

    Word *guesses;
    uint32_t num_guesses;

    /* score_matrix[g * num_targets + t] */
    uint8_t *score_matrix;
    /* score_matrix_transposed[t * num_guesses + g] */
    uint8_t *score_matrix_transposed;

    /* 128-bit Zobrist hashes */
    uint64_t *zobrist1;
    uint64_t *zobrist2;

    uint32_t *lower_bound; /* [num_targets + 1] */

    /* Shared global lock-free transposition table across threads */
    SharedTT shared_tt;
    uint64_t shared_tt_cap;
    uint64_t local_tt_cap;

    uint32_t max_candidates; /* Top-N candidate exploration limit per node (default: 100) */
} GameData;

static inline uint8_t
compute_score(const char *restrict guess, const char *restrict target)
{
    uint8_t counts[26] = {0};
    bool is_green[WORD_LEN];
    uint8_t score = 0;
    int i;

    for (i = 0; i < WORD_LEN; i++) {
        if (guess[i] == target[i]) {
            is_green[i] = true;
        } else {
            is_green[i] = false;
            counts[target[i] - 'a']++;
        }
    }

    for (i = 0; i < WORD_LEN; i++) {
        if (is_green[i]) {
            score = (uint8_t)(score * 3 + 2);
        } else if (counts[guess[i] - 'a'] > 0) {
            counts[guess[i] - 'a']--;
            score = (uint8_t)(score * 3 + 1);
        } else {
            score = (uint8_t)(score * 3 + 0);
        }
    }
    return score;
}

static uint64_t
splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t
next_pow2(uint64_t n)
{
    uint64_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

static int
load_wordlist(const char *filepath, GameData *game)
{
    FILE *fp;
    uint32_t target_cap = 1024;
    uint32_t guess_cap = 4096;
    char line[256];
    bool in_extra = false;

    fp = fopen(filepath, "r");
    if (!fp) {
        WORDLE_ERROR("Error: Unable to open word list file '%s'\n", filepath);
        return -1;
    }

    game->targets = malloc(target_cap * sizeof(Word));
    game->guesses = malloc(guess_cap * sizeof(Word));
    if (!game->targets || !game->guesses) {
        WORDLE_ERROR("Fatal: Out of memory allocating word lists\n");
        free(game->targets);
        free(game->guesses);
        game->targets = NULL;
        game->guesses = NULL;
        fclose(fp);
        return -1;
    }
    game->num_targets = 0;
    game->num_guesses = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        char word[64];
        int matched;
        size_t i;

        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            p++;
        }
        if (*p == '\0') {
            in_extra = true;
            continue;
        }

        matched = sscanf(p, "%63s", word);
        if (matched < 1) {
            continue;
        }

        if (strlen(word) != WORD_LEN) {
            continue;
        }
        for (i = 0; i < WORD_LEN; i++) {
            word[i] = (char)tolower((unsigned char)word[i]);
        }

        if (!in_extra) {
            if (game->num_targets >= target_cap) {
                Word *new_targets;
                target_cap *= 2;
                new_targets = realloc(game->targets, target_cap * sizeof(Word));
                if (!new_targets) {
                    WORDLE_ERROR("Fatal: Out of memory expanding target words\n");
                    free(game->targets);
                    free(game->guesses);
                    game->targets = NULL;
                    game->guesses = NULL;
                    fclose(fp);
                    return -1;
                }
                game->targets = new_targets;
            }
            strcpy(game->targets[game->num_targets].word, word);
            game->num_targets++;
        }

        if (game->num_guesses >= guess_cap) {
            Word *new_guesses;
            guess_cap *= 2;
            new_guesses = realloc(game->guesses, guess_cap * sizeof(Word));
            if (!new_guesses) {
                WORDLE_ERROR("Fatal: Out of memory expanding guess words\n");
                free(game->targets);
                free(game->guesses);
                game->targets = NULL;
                game->guesses = NULL;
                fclose(fp);
                return -1;
            }
            game->guesses = new_guesses;
        }
        strcpy(game->guesses[game->num_guesses].word, word);
        game->num_guesses++;
    }

    fclose(fp);
    WORDLE_INFO("Loaded %u target words, %u total valid guess words from '%s'\n",
           game->num_targets, game->num_guesses, filepath);
    return 0;
}

typedef struct {
    GameData *game;
    size_t start_g;
    size_t end_g;
} MatrixWorkerArg;

static void *
score_matrix_worker(void *arg)
{
    MatrixWorkerArg *m = (MatrixWorkerArg *)arg;
    GameData *game = m->game;
    size_t T = game->num_targets;
    size_t g;

    for (g = m->start_g; g < m->end_g; g++) {
        const char *gw = game->guesses[g].word;
        uint8_t *row = game->score_matrix + g * T;
        size_t t;
        for (t = 0; t < T; t++) {
            row[t] = compute_score(gw, game->targets[t].word);
        }
    }
    return NULL;
}

typedef struct {
    GameData *game;
    size_t start_t;
    size_t end_t;
} TransposeWorkerArg;

static void *
transpose_matrix_worker(void *arg)
{
    TransposeWorkerArg *m = (TransposeWorkerArg *)arg;
    GameData *game = m->game;
    size_t G = game->num_guesses;
    size_t T = game->num_targets;
    size_t t;
    size_t g;

    for (t = m->start_t; t < m->end_t; t++) {
        uint8_t *dst_row = game->score_matrix_transposed + t * G;
        for (g = 0; g < G; g++) {
            dst_row[g] = game->score_matrix[g * T + t];
        }
    }
    return NULL;
}

static int
compute_lower_bound_table(GameData *game)
{
    uint32_t n = game->num_targets;
    uint32_t k;

    game->lower_bound = malloc((n + 1) * sizeof(uint32_t));
    if (!game->lower_bound) {
        WORDLE_ERROR("Fatal: Out of memory allocating lower bound table\n");
        return -1;
    }
    game->lower_bound[0] = 0;
    for (k = 1; k <= n; k++) {
        if (k == 1) {
            game->lower_bound[k] = 1;
        } else if (k <= 243) {
            game->lower_bound[k] = 2 * k - 1;
        } else {
            game->lower_bound[k] = 1 + 2 * 242 + 3 * (k - 243);
        }
    }
    return 0;
}

static uint64_t
get_system_ram_bytes(void)
{
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (uint64_t)pages * (uint64_t)page_size;
    }
#endif
    return (uint64_t)8 * 1024 * 1024 * 1024ULL; /* Fallback: 8 GB */
}

static int
init_game_data(GameData *game, int num_threads, uint64_t max_memory_mb)
{
    size_t total_cells = (size_t)game->num_guesses * game->num_targets;
    pthread_t *threads = NULL;
    MatrixWorkerArg *args = NULL;
    TransposeWorkerArg *t_args = NULL;
    pthread_attr_t attr;
    bool attr_inited = false;
    size_t chunk;
    size_t t_chunk;
    int i;
    uint64_t seed1;
    uint64_t seed2;
    uint32_t t;
    uint64_t total_sys_ram;
    uint64_t max_bytes;
    size_t matrix_mem;
    uint64_t tt_budget;
    uint64_t shared_bytes;
    uint64_t local_bytes_per_thread;
    double shared_mb;
    double local_mb;
    double total_est_mb;

    game->score_matrix = malloc(total_cells * sizeof(uint8_t));
    if (!game->score_matrix) {
        WORDLE_ERROR("Fatal: Out of memory allocating score matrix\n");
        goto fail;
    }

    if (num_threads < 1) {
        num_threads = 1;
    }
    pthread_attr_init(&attr);
    attr_inited = true;
    pthread_attr_setstacksize(&attr, PTHREAD_STACK_SIZE);

    threads = malloc(num_threads * sizeof(pthread_t));
    args = malloc(num_threads * sizeof(MatrixWorkerArg));
    if (!threads || !args) {
        WORDLE_ERROR("Fatal: Out of memory allocating thread structures\n");
        goto fail;
    }
    chunk = (game->num_guesses + num_threads - 1) / num_threads;

    for (i = 0; i < num_threads; i++) {
        args[i].game = game;
        args[i].start_g = (size_t)i * chunk;
        args[i].end_g = (size_t)(i + 1) * chunk;
        if (args[i].start_g > game->num_guesses) {
            args[i].start_g = game->num_guesses;
        }
        if (args[i].end_g > game->num_guesses) {
            args[i].end_g = game->num_guesses;
        }
        pthread_create(&threads[i], &attr, score_matrix_worker, &args[i]);
    }
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(args);
    args = NULL;

    game->score_matrix_transposed = malloc(total_cells * sizeof(uint8_t));
    if (!game->score_matrix_transposed) {
        WORDLE_ERROR("Fatal: Out of memory allocating transposed score matrix\n");
        goto fail;
    }
    t_args = malloc(num_threads * sizeof(TransposeWorkerArg));
    if (!t_args) {
        WORDLE_ERROR("Fatal: Out of memory allocating transpose arguments\n");
        goto fail;
    }
    t_chunk = (game->num_targets + num_threads - 1) / num_threads;
    for (i = 0; i < num_threads; i++) {
        t_args[i].game = game;
        t_args[i].start_t = (size_t)i * t_chunk;
        t_args[i].end_t = (size_t)(i + 1) * t_chunk;
        if (t_args[i].start_t > game->num_targets) {
            t_args[i].start_t = game->num_targets;
        }
        if (t_args[i].end_t > game->num_targets) {
            t_args[i].end_t = game->num_targets;
        }
        pthread_create(&threads[i], &attr, transpose_matrix_worker, &t_args[i]);
    }
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    threads = NULL;
    free(t_args);
    t_args = NULL;
    pthread_attr_destroy(&attr);
    attr_inited = false;

    seed1 = 0x853c49e6748fea9bULL;
    seed2 = 0xda3e39cb94b95bdbULL;
    game->zobrist1 = malloc(game->num_targets * sizeof(uint64_t));
    game->zobrist2 = malloc(game->num_targets * sizeof(uint64_t));
    if (!game->zobrist1 || !game->zobrist2) {
        WORDLE_ERROR("Fatal: Out of memory allocating Zobrist tables\n");
        goto fail;
    }
    for (t = 0; t < game->num_targets; t++) {
        game->zobrist1[t] = splitmix64(&seed1);
        game->zobrist2[t] = splitmix64(&seed2);
    }

    if (compute_lower_bound_table(game) != 0) {
        goto fail;
    }

    /* --- Dynamic Laptop-Friendly Memory Auto-Tuning --- */
    total_sys_ram = get_system_ram_bytes();
    if (max_memory_mb > 0) {
        max_bytes = max_memory_mb * 1024 * 1024ULL;
    } else {
        /* Safe default: use ~6.25% of physical RAM (1/16th), bounded between 256MB and 1024MB. */
        max_bytes = total_sys_ram / 16;
        if (max_bytes > 1024 * 1024 * 1024ULL) {
            max_bytes = 1024 * 1024 * 1024ULL;
        }
        if (max_bytes < 256 * 1024 * 1024ULL) {
            max_bytes = 256 * 1024 * 1024ULL;
        }
    }

    matrix_mem = total_cells * 2 * sizeof(uint8_t);
    tt_budget = (max_bytes > matrix_mem + 32 * 1024 * 1024ULL)
                    ? (max_bytes - matrix_mem - 32 * 1024 * 1024ULL)
                    : (128 * 1024 * 1024ULL);

    shared_bytes = (uint64_t)(tt_budget * 0.65);
    local_bytes_per_thread = (uint64_t)((tt_budget * 0.35) / num_threads);

    game->shared_tt_cap = next_pow2(shared_bytes / sizeof(SharedTTEntry));
    if (game->shared_tt_cap > (1u << 25)) {
        game->shared_tt_cap = 1u << 25;
    }
    if (game->shared_tt_cap < (1u << 18)) {
        game->shared_tt_cap = 1u << 18;
    }

    game->local_tt_cap = next_pow2(local_bytes_per_thread / sizeof(TTEntry));
    if (game->local_tt_cap > (1u << 21)) {
        game->local_tt_cap = 1u << 21;
    }
    if (game->local_tt_cap < (1u << 16)) {
        game->local_tt_cap = 1u << 16;
    }

    shared_mb = (double)(game->shared_tt_cap * sizeof(SharedTTEntry)) / (1024.0 * 1024.0);
    local_mb = (double)(game->local_tt_cap * sizeof(TTEntry)) / (1024.0 * 1024.0);
    total_est_mb = (double)matrix_mem / (1024.0 * 1024.0) + shared_mb + local_mb * num_threads;

    WORDLE_INFO("System RAM: %.1f GB | Solver Memory Budget: %.1f MB (%.1f%% of RAM)\n",
           (double)total_sys_ram / (1024.0 * 1024.0 * 1024.0),
           total_est_mb, (total_est_mb * 1024.0 * 1024.0 * 100.0) / (double)total_sys_ram);
    WORDLE_INFO("Transposition Tables: Shared L2 = %.1f MB (%.2fM slots), Local L1 = %.1f MB/thread (%.0fK slots)\n",
           shared_mb, (double)game->shared_tt_cap / 1e6, local_mb, (double)game->local_tt_cap / 1e3);

    if (shared_tt_init(&game->shared_tt, game->shared_tt_cap) != 0) {
        goto fail;
    }

    return 0;

fail:
    free(threads);
    free(args);
    free(t_args);
    if (attr_inited) {
        pthread_attr_destroy(&attr);
    }
    return -1;
}

static void
free_game_data(GameData *game)
{
    shared_tt_free(&game->shared_tt);
    free(game->targets);
    free(game->guesses);
    free(game->score_matrix);
    free(game->score_matrix_transposed);
    free(game->zobrist1);
    free(game->zobrist2);
    free(game->lower_bound);
}

/* -------------------------------------------------------------
 * 128-Bit Transposition Table (Thread-Local L1 + Global Shared L2)
 * ------------------------------------------------------------- */

static int
tt_init(TT *tt, uint64_t capacity_pow2)
{
    uint64_t i;
    tt->entries = malloc(capacity_pow2 * sizeof(TTEntry));
    if (!tt->entries) {
        WORDLE_ERROR("Fatal: Out of memory allocating local transposition table\n");
        return -1;
    }
    tt->mask = capacity_pow2 - 1;
    for (i = 0; i < capacity_pow2; i++) {
        tt->entries[i].hash1 = TT_EMPTY_HASH;
        tt->entries[i].hash2 = TT_EMPTY_HASH;
    }
    return 0;
}

static void
tt_free(TT *tt)
{
    free(tt->entries);
}

#define TT_MAX_PROBES 16

static inline TTEntry *
tt_find(TT *tt, uint64_t h1, uint64_t h2, uint32_t size)
{
    uint64_t idx = (h1 ^ h2) & tt->mask;
    uint64_t probes;

    for (probes = 0; probes < TT_MAX_PROBES; probes++) {
        TTEntry *e = &tt->entries[(idx + probes) & tt->mask];
        if (e->hash1 == TT_EMPTY_HASH && e->hash2 == TT_EMPTY_HASH) {
            return NULL;
        }
        if (e->hash1 == h1 && e->hash2 == h2 && e->size == size) {
            return e;
        }
    }
    return NULL;
}

static inline TTEntry *
tt_find_or_claim(TT *tt, uint64_t h1, uint64_t h2, uint32_t size)
{
    uint64_t idx = (h1 ^ h2) & tt->mask;
    TTEntry *victim = NULL;
    uint64_t probes;

    for (probes = 0; probes < TT_MAX_PROBES; probes++) {
        TTEntry *e = &tt->entries[(idx + probes) & tt->mask];
        if (e->hash1 == TT_EMPTY_HASH && e->hash2 == TT_EMPTY_HASH) {
            e->hash1 = h1;
            e->hash2 = h2;
            e->size = size;
            e->exact_cost = UINT32_MAX;
            e->proven_lower_bound = 0;
            e->best_guess = UINT32_MAX;
            return e;
        }
        if (e->hash1 == h1 && e->hash2 == h2 && e->size == size) {
            return e;
        }
        if (!victim) {
            victim = e;
        }
    }
    if (victim) {
        victim->hash1 = h1;
        victim->hash2 = h2;
        victim->size = size;
        victim->exact_cost = UINT32_MAX;
        victim->proven_lower_bound = 0;
        victim->best_guess = UINT32_MAX;
        return victim;
    }
    return NULL;
}

static int
shared_tt_init(SharedTT *stt, uint64_t capacity_pow2)
{
    uint64_t i;
    stt->entries = calloc(capacity_pow2, sizeof(SharedTTEntry));
    if (!stt->entries) {
        WORDLE_ERROR("Fatal: Out of memory allocating shared transposition table\n");
        return -1;
    }
    stt->mask = capacity_pow2 - 1;
    for (i = 0; i < capacity_pow2; i++) {
        atomic_init(&stt->entries[i].hash1, TT_EMPTY_HASH);
        atomic_init(&stt->entries[i].hash2, TT_EMPTY_HASH);
        atomic_init(&stt->entries[i].size, 0);
        atomic_init(&stt->entries[i].exact_cost, UINT32_MAX);
        atomic_init(&stt->entries[i].proven_lower_bound, 0);
        atomic_init(&stt->entries[i].best_guess, UINT32_MAX);
    }
    return 0;
}

static void
shared_tt_free(SharedTT *stt)
{
    if (stt->entries) {
        free(stt->entries);
        stt->entries = NULL;
    }
}

static inline bool
shared_tt_find(SharedTT *stt, uint64_t h1, uint64_t h2, uint32_t size,
               uint32_t *out_exact, uint32_t *out_lb, uint32_t *out_guess)
{
    uint64_t idx;
    uint64_t probe;

    if (!stt || !stt->entries) {
        return false;
    }
    idx = (h1 ^ h2) & stt->mask;

    for (probe = 0; probe < TT_MAX_PROBES; probe++) {
        SharedTTEntry *e = &stt->entries[(idx + probe) & stt->mask];
        uint64_t eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        if (eh1 == TT_EMPTY_HASH) {
            return false;
        }
        if (eh1 == h1) {
            uint64_t eh2 = atomic_load_explicit(&e->hash2, memory_order_acquire);
            uint32_t esz = atomic_load_explicit(&e->size, memory_order_acquire);
            if (eh2 == h2 && esz == size && size > 0) {
                if (out_exact) {
                    *out_exact = atomic_load_explicit(&e->exact_cost, memory_order_acquire);
                }
                if (out_lb) {
                    *out_lb = atomic_load_explicit(&e->proven_lower_bound, memory_order_acquire);
                }
                if (out_guess) {
                    *out_guess = atomic_load_explicit(&e->best_guess, memory_order_acquire);
                }
                return true;
            }
        }
    }
    return false;
}

static inline void
shared_tt_store(SharedTT *stt, uint64_t h1, uint64_t h2, uint32_t size,
                uint32_t exact_cost, uint32_t proven_lb, uint32_t best_guess)
{
    uint64_t idx;
    uint64_t probe;

    if (!stt || !stt->entries || size == 0) {
        return;
    }
    idx = (h1 ^ h2) & stt->mask;

    for (probe = 0; probe < TT_MAX_PROBES; probe++) {
        SharedTTEntry *e = &stt->entries[(idx + probe) & stt->mask];
        uint64_t eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        if (eh1 == TT_EMPTY_HASH) {
            uint64_t expected = TT_EMPTY_HASH;
            if (atomic_compare_exchange_strong_explicit(&e->hash1, &expected, h1,
                                                       memory_order_acq_rel, memory_order_acquire)) {
                atomic_store_explicit(&e->hash2, h2, memory_order_release);
                atomic_store_explicit(&e->exact_cost, exact_cost, memory_order_release);
                atomic_store_explicit(&e->proven_lower_bound, proven_lb, memory_order_release);
                atomic_store_explicit(&e->best_guess, best_guess, memory_order_release);
                atomic_store_explicit(&e->size, size, memory_order_release); /* Published last */
                return;
            }
            eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        }
        if (eh1 == h1) {
            uint64_t eh2 = atomic_load_explicit(&e->hash2, memory_order_acquire);
            uint32_t esz = atomic_load_explicit(&e->size, memory_order_acquire);
            if (eh2 == h2 && esz == size && size > 0) {
                if (exact_cost != UINT32_MAX) {
                    atomic_store_explicit(&e->exact_cost, exact_cost, memory_order_release);
                }
                if (proven_lb > 0) {
                    uint32_t cur_lb = atomic_load_explicit(&e->proven_lower_bound, memory_order_relaxed);
                    while (proven_lb > cur_lb && !atomic_compare_exchange_weak_explicit(
                               &e->proven_lower_bound, &cur_lb, proven_lb,
                               memory_order_release, memory_order_relaxed)) {
                    }
                }
                if (best_guess != UINT32_MAX) {
                    atomic_store_explicit(&e->best_guess, best_guess, memory_order_release);
                }
                return;
            }
        }
    }
}

/* -------------------------------------------------------------
 * Fast Inlined 64-bit Introsort (Zero Function Pointers)
 * ------------------------------------------------------------- */

static inline void
sort64_asc(uint64_t *a, size_t n)
{
    size_t i;

    while (n > 16) {
        size_t mid = n / 2;
        uint64_t pivot;
        size_t i_part;
        size_t j_part;

        if (a[0] > a[mid]) {
            uint64_t t = a[0]; a[0] = a[mid]; a[mid] = t;
        }
        if (a[0] > a[n - 1]) {
            uint64_t t = a[0]; a[0] = a[n - 1]; a[n - 1] = t;
        }
        if (a[mid] > a[n - 1]) {
            uint64_t t = a[mid]; a[mid] = a[n - 1]; a[n - 1] = t;
        }

        pivot = a[mid];
        i_part = 0;
        j_part = n - 1;
        while (1) {
            while (a[i_part] < pivot) {
                i_part++;
            }
            while (a[j_part] > pivot) {
                j_part--;
            }
            if (i_part >= j_part) {
                break;
            }
            {
                uint64_t t = a[i_part];
                a[i_part] = a[j_part];
                a[j_part] = t;
            }
            i_part++;
            j_part--;
        }
        if (i_part < n - i_part) {
            sort64_asc(a, i_part);
            a += i_part;
            n -= i_part;
        } else {
            sort64_asc(a + i_part, n - i_part);
            n = i_part;
        }
    }

    for (i = 1; i < n; i++) {
        uint64_t key = a[i];
        size_t j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = key;
    }
}

/* Sorts the k smallest elements of a[0..n) into a[0..k) in ascending order. */
static void
sort64_asc_top(uint64_t *a, size_t n, size_t k)
{
    size_t lo;
    size_t hi;

    if (k >= n) {
        sort64_asc(a, n);
        return;
    }
    if (k == 0) {
        return;
    }

    lo = 0;
    hi = n;
    while (hi - lo > 16) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t pivot;
        size_t i;
        size_t j;

        if (a[lo] > a[mid]) {
            uint64_t t = a[lo]; a[lo] = a[mid]; a[mid] = t;
        }
        if (a[lo] > a[hi - 1]) {
            uint64_t t = a[lo]; a[lo] = a[hi - 1]; a[hi - 1] = t;
        }
        if (a[mid] > a[hi - 1]) {
            uint64_t t = a[mid]; a[mid] = a[hi - 1]; a[hi - 1] = t;
        }

        pivot = a[mid];
        i = lo;
        j = hi - 1;
        while (1) {
            while (a[i] < pivot) {
                i++;
            }
            while (a[j] > pivot) {
                j--;
            }
            if (i >= j) {
                break;
            }
            {
                uint64_t t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
            i++;
            j--;
        }
        if (k - 1 < i) {
            hi = i;
        } else {
            lo = i;
        }
    }
    sort64_asc(a, lo);
    sort64_asc(a + lo, hi - lo);
}

typedef struct {
    GameData *game;
    TT tt;
    uint64_t *candidate_keys; /* MAX_SOLVER_DEPTH layers of num_guesses uint64_t keys */
    uint32_t max_candidates;
    uint64_t nodes_visited;
} Solver;

static int
solver_init(Solver *s, GameData *game)
{
    uint64_t tt_cap;

    s->game = game;
    tt_cap = game->local_tt_cap > 0 ? game->local_tt_cap : (1u << 19);
    if (tt_init(&s->tt, tt_cap) != 0) {
        return -1;
    }

    s->candidate_keys = malloc((size_t)MAX_SOLVER_DEPTH * (size_t)game->num_guesses * sizeof(uint64_t));
    if (!s->candidate_keys) {
        WORDLE_ERROR("Fatal: Out of memory allocating candidate keys buffer\n");
        tt_free(&s->tt);
        return -1;
    }
    s->max_candidates = game->max_candidates > 0 ? game->max_candidates : game->num_guesses;
    s->nodes_visited = 0;
    return 0;
}

static void
solver_free(Solver *s)
{
    tt_free(&s->tt);
    free(s->candidate_keys);
}

static inline TTEntry *
solver_tt_find(Solver *solver, uint64_t h1, uint64_t h2, uint32_t size)
{
    TTEntry *entry = tt_find(&solver->tt, h1, h2, size);
    if (entry) {
        return entry;
    }
    if (solver->game && solver->game->shared_tt.entries) {
        uint32_t s_exact = UINT32_MAX;
        uint32_t s_lb = 0;
        uint32_t s_guess = UINT32_MAX;
        if (shared_tt_find(&solver->game->shared_tt, h1, h2, size, &s_exact, &s_lb, &s_guess)) {
            TTEntry *e = tt_find_or_claim(&solver->tt, h1, h2, size);
            if (e) {
                e->exact_cost = s_exact;
                e->proven_lower_bound = s_lb;
                e->best_guess = s_guess;
                return e;
            }
        }
    }
    return NULL;
}

static inline void
solver_tt_store_exact(Solver *solver, uint64_t h1, uint64_t h2, uint32_t size,
                      uint32_t exact_cost, uint32_t best_guess)
{
    TTEntry *e = tt_find_or_claim(&solver->tt, h1, h2, size);
    if (e) {
        e->exact_cost = exact_cost;
        if (best_guess != UINT32_MAX) {
            e->best_guess = best_guess;
        }
    }
    if (solver->game && solver->game->shared_tt.entries) {
        shared_tt_store(&solver->game->shared_tt, h1, h2, size, exact_cost, exact_cost, best_guess);
    }
}

static inline void
solver_tt_store_lb(Solver *solver, uint64_t h1, uint64_t h2, uint32_t size,
                   uint32_t lb, uint32_t best_guess)
{
    TTEntry *e = tt_find_or_claim(&solver->tt, h1, h2, size);
    if (e) {
        if (lb > e->proven_lower_bound) {
            e->proven_lower_bound = lb;
        }
        if (best_guess != UINT32_MAX && e->best_guess == UINT32_MAX) {
            e->best_guess = best_guess;
        }
    }
    if (solver->game && solver->game->shared_tt.entries) {
        shared_tt_store(&solver->game->shared_tt, h1, h2, size, UINT32_MAX, lb, best_guess);
    }
}

typedef struct {
    uint16_t score;
    uint32_t size;
    uint32_t offset;
    uint64_t hash1;
    uint64_t hash2;
} BucketInfo;

static int
compare_bucket_size_desc(const void *a, const void *b)
{
    const BucketInfo *ba = (const BucketInfo *)a;
    const BucketInfo *bb = (const BucketInfo *)b;
    return (ba->size > bb->size) ? -1 : ((ba->size < bb->size) ? 1 : 0);
}

static int
compare_bucket_size_asc(const void *a, const void *b)
{
    const BucketInfo *ba = (const BucketInfo *)a;
    const BucketInfo *bb = (const BucketInfo *)b;
    return (ba->size < bb->size) ? -1 : ((ba->size > bb->size) ? 1 : 0);
}

/* -------------------------------------------------------------
 * Fast 1-Ply Greedy Heuristic (Upper Bound Seeder)
 * ------------------------------------------------------------- */

static uint32_t
solve_greedy_tree(GameData *game, const uint32_t *targets, uint32_t count)
{
    uint32_t best_g = UINT32_MAX;
    uint32_t min_sum_sq = UINT32_MAX;
    uint32_t max_active = 0;
    uint32_t g;
    const uint8_t *row;
    uint32_t hist[NUM_SCORES] = {0};
    uint32_t offsets[NUM_SCORES + 1];
    uint32_t cur[NUM_SCORES];
    uint32_t *part;
    uint32_t total;
    int s;
    uint32_t i;

    if (count == 0) {
        return 0;
    }
    if (count == 1) {
        return 1;
    }
    if (count == 2) {
        return 3;
    }

    for (g = 0; g < game->num_guesses; g++) {
        const uint8_t *grow = game->score_matrix + (size_t)g * game->num_targets;
        uint32_t ghist[NUM_SCORES] = {0};
        uint32_t sum_sq = 0;
        uint32_t active = 0;

        for (i = 0; i < count; i++) {
            ghist[grow[targets[i]]]++;
        }
        for (s = 0; s < NUM_SCORES; s++) {
            if (ghist[s] > 0) {
                active++;
                sum_sq += ghist[s] * ghist[s];
            }
        }
        if (active <= 1) {
            continue;
        }

        if (sum_sq < min_sum_sq || (sum_sq == min_sum_sq && active > max_active)) {
            min_sum_sq = sum_sq;
            max_active = active;
            best_g = g;
        }
    }

    if (best_g == UINT32_MAX) {
        best_g = targets[0];
    }

    row = game->score_matrix + (size_t)best_g * game->num_targets;
    for (i = 0; i < count; i++) {
        hist[row[targets[i]]]++;
    }

    offsets[0] = 0;
    for (s = 0; s < NUM_SCORES; s++) {
        offsets[s + 1] = offsets[s] + hist[s];
    }
    memcpy(cur, offsets, sizeof(cur));

    part = malloc(count * sizeof(uint32_t));
    for (i = 0; i < count; i++) {
        uint8_t sc = row[targets[i]];
        part[cur[sc]++] = targets[i];
    }

    total = count;
    for (s = 0; s < NUM_SCORES; s++) {
        if (s != EXACT_MATCH && hist[s] > 0) {
            total += solve_greedy_tree(game, &part[offsets[s]], hist[s]);
        }
    }
    free(part);
    return total;
}

static uint32_t
compute_opener_greedy_upper_bound(GameData *game, uint32_t opener_idx)
{
    uint32_t count = game->num_targets;
    const uint8_t *row = game->score_matrix + (size_t)opener_idx * count;
    uint32_t hist[NUM_SCORES] = {0};
    uint32_t offsets[NUM_SCORES + 1];
    uint32_t cur[NUM_SCORES];
    uint32_t *part;
    uint32_t total;
    int s;
    uint32_t t;

    for (t = 0; t < count; t++) {
        hist[row[t]]++;
    }
    offsets[0] = 0;
    for (s = 0; s < NUM_SCORES; s++) {
        offsets[s + 1] = offsets[s] + hist[s];
    }
    memcpy(cur, offsets, sizeof(cur));

    part = malloc(count * sizeof(uint32_t));
    for (t = 0; t < count; t++) {
        uint8_t sc = row[t];
        part[cur[sc]++] = t;
    }

    total = count;
    for (s = 0; s < NUM_SCORES; s++) {
        if (s != EXACT_MATCH && hist[s] > 0) {
            total += solve_greedy_tree(game, &part[offsets[s]], hist[s]);
        }
    }
    free(part);
    return total;
}

/* -------------------------------------------------------------
 * Core Branch-and-Bound Solver
 * ------------------------------------------------------------- */

static uint32_t
solve_subset(Solver *solver, const uint32_t *targets, uint32_t count,
             uint64_t h1, uint64_t h2, uint32_t beta, uint32_t *out_guess, uint32_t depth)
{
    GameData *game;
    uint32_t num_targets;
    uint32_t num_guesses;
    const uint8_t *matrix;
    uint32_t node_lb;
    TTEntry *entry;
    uint32_t suggested_guess;
    uint32_t good_target;
    uint32_t t_counts[NUM_SCORES] = {0};
    uint16_t t_active[count + 1];
    uint64_t *candidate_keys;
    uint32_t global_lb1;
    uint32_t global_ub1;
    uint32_t best_exact_g;
    uint8_t counts[NUM_SCORES] = {0};
    uint32_t limit;
    uint32_t current_best;
    uint32_t best_g;
    bool found_improvement;
    uint32_t local_partition[count];
    uint32_t offsets[NUM_SCORES + 1];
    BucketInfo buckets[NUM_SCORES];
    uint32_t hist[NUM_SCORES] = {0};
    uint64_t bucket_h1[NUM_SCORES];
    uint64_t bucket_h2[NUM_SCORES];
    uint32_t u_costs[NUM_SCORES];
    uint16_t active_scores[count + 1];
    uint32_t c;
    uint32_t i;

    solver->nodes_visited++;
    game = solver->game;

    if (count == 0) {
        return 0;
    }
    if (count == 1) {
        if (out_guess) {
            *out_guess = targets[0];
        }
        return 1;
    }
    if (count == 2) {
        if (out_guess) {
            *out_guess = targets[0];
        }
        return 3;
    }

    num_targets = game->num_targets;
    num_guesses = game->num_guesses;
    matrix = game->score_matrix;

    node_lb = game->lower_bound[count];
    entry = solver_tt_find(solver, h1, h2, count);
    suggested_guess = UINT32_MAX;

    if (entry) {
        if (entry->exact_cost != UINT32_MAX) {
            if (out_guess) {
                *out_guess = entry->best_guess;
            }
            return entry->exact_cost;
        }
        if (entry->proven_lower_bound > node_lb) {
            node_lb = entry->proven_lower_bound;
        }
        suggested_guess = entry->best_guess;
    }

    if (node_lb >= beta) {
        solver_tt_store_lb(solver, h1, h2, count, node_lb, suggested_guess);
        return node_lb;
    }

    /* ---- O(|H|^2) Target-Only Instant Resolution Pre-Check ---- */
    if (count <= 243) {
        good_target = UINT32_MAX;
        for (i = 0; i < count; i++) {
            uint32_t t = targets[i];
            const uint8_t *row = matrix + (size_t)t * num_targets;
            uint32_t bad = 0;
            uint32_t n_act = 0;
            uint32_t j;
            uint32_t k;

            for (j = 0; j < count; j++) {
                uint8_t sc = row[targets[j]];
                if (t_counts[sc] == 0) {
                    t_active[n_act++] = sc;
                }
                t_counts[sc]++;
                if (t_counts[sc] >= 2) {
                    bad++;
                }
            }
            for (k = 0; k < n_act; k++) {
                t_counts[t_active[k]] = 0;
            }

            if (bad == 0) {
                uint32_t cost = 2 * count - 1;
                solver_tt_store_exact(solver, h1, h2, count, cost, t);
                if (out_guess) {
                    *out_guess = t;
                }
                return cost;
            }
            if (bad == 1 && good_target == UINT32_MAX) {
                good_target = t;
            }
        }

        if (good_target != UINT32_MAX) {
            uint32_t cost = 2 * count;
            if (cost < beta) {
                solver_tt_store_exact(solver, h1, h2, count, cost, good_target);
                if (out_guess) {
                    *out_guess = good_target;
                }
                return cost;
            } else {
                solver_tt_store_lb(solver, h1, h2, count, cost, good_target);
                if (out_guess) {
                    *out_guess = good_target;
                }
                return cost;
            }
        } else {
            if (2 * count > node_lb) {
                node_lb = 2 * count;
                if (node_lb >= beta) {
                    solver_tt_store_lb(solver, h1, h2, count, node_lb, suggested_guess);
                    if (out_guess) {
                        *out_guess = (suggested_guess != UINT32_MAX) ? suggested_guess : targets[0];
                    }
                    return node_lb;
                }
            }
        }
    }

    /* ---- Fast Move Ordering & lb1 Computation ---- */
    assert(depth < MAX_SOLVER_DEPTH);
    candidate_keys = solver->candidate_keys + (size_t)depth * num_guesses;
    global_lb1 = UINT32_MAX;
    global_ub1 = UINT32_MAX;
    best_exact_g = UINT32_MAX;

    if (count <= 8) {
        const uint8_t *col0 = game->score_matrix_transposed + (size_t)targets[0] * num_guesses;
        const uint8_t *col1 = game->score_matrix_transposed + (size_t)targets[1] * num_guesses;
        const uint8_t *col2 = (count > 2) ? game->score_matrix_transposed + (size_t)targets[2] * num_guesses : NULL;
        const uint8_t *col3 = (count > 3) ? game->score_matrix_transposed + (size_t)targets[3] * num_guesses : NULL;
        const uint8_t *col4 = (count > 4) ? game->score_matrix_transposed + (size_t)targets[4] * num_guesses : NULL;
        const uint8_t *col5 = (count > 5) ? game->score_matrix_transposed + (size_t)targets[5] * num_guesses : NULL;
        const uint8_t *col6 = (count > 6) ? game->score_matrix_transposed + (size_t)targets[6] * num_guesses : NULL;
        const uint8_t *col7 = (count > 7) ? game->score_matrix_transposed + (size_t)targets[7] * num_guesses : NULL;
        uint32_t g;

        for (g = 0; g < num_guesses; g++) {
            uint32_t s2 = 0;
            uint32_t guess_lb = count;
            bool max_bucket_le_2 = true;
            bool in_set;
            uint32_t rank_score;
            uint8_t sc0 = col0[g]; uint8_t c0 = ++counts[sc0]; s2 += 2 * c0 - 1; guess_lb += 2 - (c0 == 1); max_bucket_le_2 &= (c0 <= 2);
            uint8_t sc1 = col1[g]; uint8_t c1 = ++counts[sc1]; s2 += 2 * c1 - 1; guess_lb += 2 - (c1 == 1); max_bucket_le_2 &= (c1 <= 2);
            uint8_t sc2 = 0, sc3 = 0, sc4 = 0, sc5 = 0, sc6 = 0, sc7 = 0;

            if (count > 2) { sc2 = col2[g]; uint8_t c2 = ++counts[sc2]; s2 += 2 * c2 - 1; guess_lb += 2 - (c2 == 1); max_bucket_le_2 &= (c2 <= 2); }
            if (count > 3) { sc3 = col3[g]; uint8_t c3 = ++counts[sc3]; s2 += 2 * c3 - 1; guess_lb += 2 - (c3 == 1); max_bucket_le_2 &= (c3 <= 2); }
            if (count > 4) { sc4 = col4[g]; uint8_t c4 = ++counts[sc4]; s2 += 2 * c4 - 1; guess_lb += 2 - (c4 == 1); max_bucket_le_2 &= (c4 <= 2); }
            if (count > 5) { sc5 = col5[g]; uint8_t c5 = ++counts[sc5]; s2 += 2 * c5 - 1; guess_lb += 2 - (c5 == 1); max_bucket_le_2 &= (c5 <= 2); }
            if (count > 6) { sc6 = col6[g]; uint8_t c6 = ++counts[sc6]; s2 += 2 * c6 - 1; guess_lb += 2 - (c6 == 1); max_bucket_le_2 &= (c6 <= 2); }
            if (count > 7) { sc7 = col7[g]; uint8_t c7 = ++counts[sc7]; s2 += 2 * c7 - 1; guess_lb += 2 - (c7 == 1); max_bucket_le_2 &= (c7 <= 2); }

            in_set = (counts[EXACT_MATCH] > 0);
            guess_lb -= (in_set ? 1 : 0);

            counts[sc0] = 0; counts[sc1] = 0;
            if (count > 2) counts[sc2] = 0;
            if (count > 3) counts[sc3] = 0;
            if (count > 4) counts[sc4] = 0;
            if (count > 5) counts[sc5] = 0;
            if (count > 6) counts[sc6] = 0;
            if (count > 7) counts[sc7] = 0;

            if (guess_lb < global_lb1) {
                global_lb1 = guess_lb;
            }
            if (max_bucket_le_2 && guess_lb < global_ub1) {
                global_ub1 = guess_lb;
                best_exact_g = g;
            }

            if (s2 == count) {
                if (sc0 == EXACT_MATCH || sc1 == EXACT_MATCH || (count > 2 && sc2 == EXACT_MATCH) ||
                    (count > 3 && sc3 == EXACT_MATCH) || (count > 4 && sc4 == EXACT_MATCH) ||
                    (count > 5 && sc5 == EXACT_MATCH) || (count > 6 && sc6 == EXACT_MATCH) ||
                    (count > 7 && sc7 == EXACT_MATCH)) {
                    uint32_t cost = 2 * count - 1;
                    solver_tt_store_exact(solver, h1, h2, count, cost, g);
                    if (out_guess) {
                        *out_guess = g;
                    }
                    return cost;
                }
                if (2 * count < beta) {
                    solver_tt_store_exact(solver, h1, h2, count, 2 * count, g);
                    if (out_guess) {
                        *out_guess = g;
                    }
                    return 2 * count;
                }
            }

            rank_score = 2 * s2 + count * guess_lb - (in_set ? 2 : 0);
            candidate_keys[g] = ((uint64_t)rank_score << 32) | ((uint64_t)(guess_lb & 0xFFFF) << 16) | (uint64_t)g;
        }
    } else {
        const uint8_t *cols[count];
        uint16_t big_counts[NUM_SCORES] = {0};
        const bool can_abort = (beta <= 3 * count);
        uint32_t g;

        for (i = 0; i < count; i++) {
            cols[i] = game->score_matrix_transposed + (size_t)targets[i] * num_guesses;
        }

        for (g = 0; g < num_guesses; g++) {
            uint32_t s2 = 0;
            uint32_t guess_lb = count;
            bool max_bucket_le_2 = true;
            bool aborted = false;
            bool in_set;
            uint32_t rank_score;
            uint32_t touched = 0;
            uint32_t r;

            for (i = 0; i < count; i++) {
                uint8_t sc = cols[i][g];
                uint16_t cval = ++big_counts[sc];
                s2 += 2 * cval - 1;
                guess_lb += (cval == 1) ? 1 : (cval <= 243 ? 2 : 3);
                if (cval > 2) {
                    max_bucket_le_2 = false;
                }
                touched++;
                if (can_abort && guess_lb >= beta + (big_counts[EXACT_MATCH] ? 1 : 0)) {
                    aborted = true;
                    break;
                }
            }

            in_set = (big_counts[EXACT_MATCH] > 0);
            if (!aborted) {
                guess_lb -= big_counts[EXACT_MATCH];
            }

            for (r = 0; r < touched; r++) {
                big_counts[cols[r][g]] = 0;
            }

            if (aborted) {
                candidate_keys[g] = UINT64_MAX;
                continue;
            }

            if (guess_lb < global_lb1) {
                global_lb1 = guess_lb;
            }
            if (max_bucket_le_2 && guess_lb < global_ub1) {
                global_ub1 = guess_lb;
                best_exact_g = g;
            }

            rank_score = 2 * s2 + count * guess_lb - (in_set ? 2 : 0);
            candidate_keys[g] = ((uint64_t)rank_score << 32) | ((uint64_t)(guess_lb & 0xFFFF) << 16) | (uint64_t)g;
        }
    }

    /* Sound fail-soft cutoff if lb1 >= beta */
    if (global_lb1 == UINT32_MAX) {
        global_lb1 = beta;
    }
    if (global_lb1 >= beta) {
        solver_tt_store_lb(solver, h1, h2, count, global_lb1, UINT32_MAX);
        return global_lb1;
    }

    /* Exact resolution if ub1 == lb1 */
    if (global_ub1 == global_lb1 && best_exact_g != UINT32_MAX && global_lb1 < beta) {
        solver_tt_store_exact(solver, h1, h2, count, global_lb1, best_exact_g);
        if (out_guess) {
            *out_guess = best_exact_g;
        }
        return global_lb1;
    }

    /* Compact candidate keys: drop any whose analytic lb already meets beta. */
    {
        uint32_t m = 0;
        for (i = 0; i < num_guesses; i++) {
            uint32_t glb = (uint32_t)((candidate_keys[i] >> 16) & 0xFFFF);
            if (glb < beta) {
                candidate_keys[m++] = candidate_keys[i];
            }
        }
        if (m == 0) {
            solver_tt_store_lb(solver, h1, h2, count, beta, suggested_guess);
            if (out_guess) {
                *out_guess = (suggested_guess != UINT32_MAX) ? suggested_guess : targets[0];
            }
            return beta;
        }
        limit = solver->max_candidates < m ? solver->max_candidates : m;
        sort64_asc_top(candidate_keys, m, limit);

        if (suggested_guess != UINT32_MAX) {
            uint32_t r;
            for (r = 0; r < m; r++) {
                if ((uint32_t)(candidate_keys[r] & 0xFFFF) == suggested_guess) {
                    uint64_t tmp = candidate_keys[0];
                    candidate_keys[0] = candidate_keys[r];
                    candidate_keys[r] = tmp;
                    break;
                }
            }
        }
    }

    /* ---- Main Branch-and-Bound Loop ---- */
    current_best = beta;
    best_g = (uint32_t)(candidate_keys[0] & 0xFFFF);
    found_improvement = false;

    for (c = 0; c < limit; c++) {
        uint32_t clb = (uint32_t)((candidate_keys[c] >> 16) & 0xFFFF);
        uint32_t g;
        const uint8_t *row;
        uint32_t active_buckets = 0;
        uint32_t guess_lb = count;
        bool has_exact;
        uint32_t b;
        uint32_t tier2_total_lb;
        uint32_t resolved_cost;
        uint32_t num_unresolved;
        BucketInfo unresolved_buckets[NUM_SCORES];
        uint32_t cur_offsets[NUM_SCORES];
        uint32_t u;
        uint32_t running_cost;
        uint32_t remaining_lb;
        bool pruned;

        if (clb >= current_best) {
            continue;
        }

        g = (uint32_t)(candidate_keys[c] & 0xFFFF);
        row = matrix + (size_t)g * num_targets;

        for (i = 0; i < count; i++) {
            uint32_t tid = targets[i];
            uint8_t sc = row[tid];
            if (hist[sc] == 0) {
                active_scores[active_buckets++] = sc;
                bucket_h1[sc] = game->zobrist1[tid];
                bucket_h2[sc] = game->zobrist2[tid];
            } else {
                bucket_h1[sc] ^= game->zobrist1[tid];
                bucket_h2[sc] ^= game->zobrist2[tid];
            }
            hist[sc]++;
        }

        has_exact = (hist[EXACT_MATCH] > 0);
        for (b = 0; b < active_buckets; b++) {
            uint16_t s = active_scores[b];
            uint32_t sz = hist[s];
            if (s != EXACT_MATCH) {
                guess_lb += game->lower_bound[sz];
            }
            buckets[b].score = s;
            buckets[b].size = sz;
            buckets[b].hash1 = bucket_h1[s];
            buckets[b].hash2 = bucket_h2[s];
        }

        if (active_buckets == 1) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            continue;
        }

        /* Perfect split shortcut (2n - 1) */
        if (active_buckets == count && has_exact) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            current_best = guess_lb;
            best_g = g;
            found_improvement = true;
            break;
        }

        /* Non-candidate singleton split shortcut (2n) */
        if (active_buckets == count && !has_exact) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            if (2 * count < current_best) {
                current_best = 2 * count;
                best_g = g;
                found_improvement = true;
                if (current_best <= node_lb) {
                    break;
                }
            }
            continue;
        }

        if (guess_lb >= current_best) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            continue;
        }

        /* ---- Tier 2: Deferred in-register TT probe without local_partition or qsort ---- */
        tier2_total_lb = count;
        resolved_cost = count;
        num_unresolved = 0;

        for (b = 0; b < active_buckets; b++) {
            uint16_t s = buckets[b].score;
            uint32_t sz;
            uint64_t bh1;
            uint64_t bh2;
            uint32_t base_lb;
            TTEntry *tentry;

            if (s == EXACT_MATCH) {
                continue;
            }
            sz = buckets[b].size;
            if (sz <= 2) {
                uint32_t c_cost = (sz == 1) ? 1 : 3;
                tier2_total_lb += c_cost;
                resolved_cost += c_cost;
                continue;
            }

            bh1 = buckets[b].hash1;
            bh2 = buckets[b].hash2;

            base_lb = game->lower_bound[sz];
            tentry = solver_tt_find(solver, bh1, bh2, sz);
            if (tentry) {
                if (tentry->exact_cost != UINT32_MAX) {
                    tier2_total_lb += tentry->exact_cost;
                    resolved_cost += tentry->exact_cost;
                } else if (tentry->proven_lower_bound > base_lb) {
                    tier2_total_lb += tentry->proven_lower_bound;
                    unresolved_buckets[num_unresolved++] = buckets[b];
                } else {
                    tier2_total_lb += base_lb;
                    unresolved_buckets[num_unresolved++] = buckets[b];
                }
            } else {
                tier2_total_lb += base_lb;
                unresolved_buckets[num_unresolved++] = buckets[b];
            }
        }

        if (tier2_total_lb >= current_best) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            continue;
        }

        if (num_unresolved == 0) {
            for (b = 0; b < active_buckets; b++) {
                hist[active_scores[b]] = 0;
            }
            if (resolved_cost < current_best) {
                current_best = resolved_cost;
                best_g = g;
                found_improvement = true;
                if (current_best <= node_lb) {
                    break;
                }
            }
            continue;
        }

        /* ---- Build local_partition ONLY for surviving unresolved candidate ---- */
        offsets[0] = 0;
        for (i = 0; i < NUM_SCORES; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }
        memcpy(cur_offsets, offsets, sizeof(cur_offsets));
        for (i = 0; i < count; i++) {
            uint8_t sc = row[targets[i]];
            local_partition[cur_offsets[sc]++] = targets[i];
        }
        for (b = 0; b < active_buckets; b++) {
            hist[active_scores[b]] = 0;
        }

        for (u = 0; u < num_unresolved; u++) {
            unresolved_buckets[u].offset = offsets[unresolved_buckets[u].score];
        }
        qsort(unresolved_buckets, num_unresolved, sizeof(BucketInfo), compare_bucket_size_asc);

        /* ---- Tier 3: Ordered Fail-Soft Recursion on Unresolved Buckets ---- */
        running_cost = resolved_cost;
        remaining_lb = 0;
        for (u = 0; u < num_unresolved; u++) {
            remaining_lb += game->lower_bound[unresolved_buckets[u].size];
            u_costs[u] = 0;
        }

        pruned = false;
        for (u = 0; u < num_unresolved; u++) {
            uint32_t sz = unresolved_buckets[u].size;
            uint32_t bucket_beta;
            uint32_t bucket_cost;

            remaining_lb -= game->lower_bound[sz];
            if (running_cost + remaining_lb >= current_best) {
                pruned = true;
                break;
            }

            bucket_beta = current_best - running_cost - remaining_lb;
            bucket_cost = solve_subset(solver, &local_partition[unresolved_buckets[u].offset], sz,
                                       unresolved_buckets[u].hash1, unresolved_buckets[u].hash2,
                                       bucket_beta, NULL, depth + 1);
            if (bucket_cost >= bucket_beta) {
                pruned = true;
                break;
            }
            u_costs[u] = bucket_cost;
            running_cost += bucket_cost;
        }

        if (pruned && num_unresolved >= 2) {
            uint32_t u1;
            uint32_t u2;
            for (u1 = 0; u1 < num_unresolved; u1++) {
                if (u_costs[u1] == 0) {
                    continue;
                }
                for (u2 = u1 + 1; u2 < num_unresolved; u2++) {
                    uint64_t mh1;
                    uint64_t mh2;
                    uint32_t msize;
                    uint32_t mlb;

                    if (u_costs[u2] == 0) {
                        continue;
                    }
                    mh1 = unresolved_buckets[u1].hash1 ^ unresolved_buckets[u2].hash1;
                    mh2 = unresolved_buckets[u1].hash2 ^ unresolved_buckets[u2].hash2;
                    msize = unresolved_buckets[u1].size + unresolved_buckets[u2].size;
                    mlb = u_costs[u1] + u_costs[u2];
                    solver_tt_store_lb(solver, mh1, mh2, msize, mlb, UINT32_MAX);
                }
            }
        }

        if (!pruned && running_cost < current_best) {
            current_best = running_cost;
            best_g = g;
            found_improvement = true;
            if (current_best <= node_lb) {
                break;
            }
        }
    }

    if (found_improvement) {
        solver_tt_store_exact(solver, h1, h2, count, current_best, best_g);
        if (out_guess) {
            *out_guess = best_g;
        }
        return current_best;
    } else {
        solver_tt_store_lb(solver, h1, h2, count, beta, best_g);
        if (out_guess) {
            *out_guess = best_g;
        }
        return beta;
    }
}

/* -------------------------------------------------------------
 * Parallel Evaluation
 * ------------------------------------------------------------- */

typedef struct {
    uint32_t opener_idx;
    uint32_t exact_total_cost;
    double avg_guesses;
    double time_sec;
    uint64_t nodes;
    bool is_exact;
} OpenerResult;

static void
partition_root(GameData *game, uint32_t opener_idx, uint32_t *local_partition,
               BucketInfo *buckets, uint32_t *out_active_buckets)
{
    uint32_t count = game->num_targets;
    const uint8_t *row = game->score_matrix + (size_t)opener_idx * game->num_targets;
    uint32_t hist[NUM_SCORES] = {0};
    uint32_t offsets[NUM_SCORES + 1];
    uint32_t cur[NUM_SCORES];
    uint32_t active = 0;
    uint32_t t;
    int s;

    for (t = 0; t < count; t++) {
        hist[row[t]]++;
    }
    offsets[0] = 0;
    for (s = 0; s < NUM_SCORES; s++) {
        offsets[s + 1] = offsets[s] + hist[s];
    }
    memcpy(cur, offsets, sizeof(cur));
    for (t = 0; t < count; t++) {
        uint8_t sc = row[t];
        local_partition[cur[sc]++] = t;
    }

    for (s = 0; s < NUM_SCORES; s++) {
        if (hist[s] > 0) {
            uint32_t off = offsets[s];
            uint64_t bh1 = 0;
            uint64_t bh2 = 0;
            uint32_t j;
            for (j = 0; j < hist[s]; j++) {
                uint32_t tid = local_partition[off + j];
                bh1 ^= game->zobrist1[tid];
                bh2 ^= game->zobrist2[tid];
            }
            buckets[active++] = (BucketInfo){
                .score = (uint16_t)s,
                .size = hist[s],
                .offset = off,
                .hash1 = bh1,
                .hash2 = bh2
            };
        }
    }
    *out_active_buckets = active;
}

typedef struct {
    GameData *game;
    const uint32_t *local_partition;
    BucketInfo *buckets;
    uint32_t num_buckets;
    const uint32_t *bucket_betas;
    atomic_size_t next_idx;
    atomic_bool failed;
    uint32_t *out_costs;
    uint32_t *out_guesses;
    uint64_t *out_nodes;
} BucketPool;

static void *
bucket_worker(void *arg)
{
    BucketPool *pool = (BucketPool *)arg;
    Solver solver;

    if (solver_init(&solver, pool->game) != 0) {
        atomic_store(&pool->failed, true);
        return NULL;
    }
    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        BucketInfo *bkt;
        uint32_t best_guess;
        uint64_t start_nodes;
        uint32_t beta;
        uint32_t cost;

        if (idx >= pool->num_buckets) {
            break;
        }
        bkt = &pool->buckets[idx];
        if (bkt->score == EXACT_MATCH) {
            continue;
        }

        start_nodes = solver.nodes_visited;
        beta = pool->bucket_betas ? pool->bucket_betas[idx] : UINT32_MAX;
        cost = solve_subset(&solver, &pool->local_partition[bkt->offset], bkt->size,
                            bkt->hash1, bkt->hash2, beta, &best_guess, 0);
        pool->out_costs[idx] = cost;
        pool->out_guesses[idx] = best_guess;
        pool->out_nodes[idx] = solver.nodes_visited - start_nodes;
    }
    solver_free(&solver);
    return NULL;
}

static OpenerResult
evaluate_opener_parallel(GameData *game, uint32_t opener_idx, int num_threads)
{
    struct timespec t0, t1;
    uint32_t greedy_upper_bound;
    uint32_t count;
    uint32_t *local_partition;
    BucketInfo *buckets;
    uint32_t active_buckets;
    uint32_t total_other_lb = 0;
    uint32_t *bucket_betas;
    uint32_t b;
    uint32_t total_cost;
    uint64_t total_nodes = 0;
    bool failed = false;
    OpenerResult res;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    greedy_upper_bound = compute_opener_greedy_upper_bound(game, opener_idx);

    count = game->num_targets;
    local_partition = malloc(count * sizeof(uint32_t));
    buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    for (b = 0; b < active_buckets; b++) {
        if (buckets[b].score != EXACT_MATCH) {
            total_other_lb += game->lower_bound[buckets[b].size];
        }
    }

    bucket_betas = malloc(active_buckets * sizeof(uint32_t));
    for (b = 0; b < active_buckets; b++) {
        uint32_t other_lb;
        if (buckets[b].score == EXACT_MATCH) {
            bucket_betas[b] = 0;
            continue;
        }
        other_lb = total_other_lb - game->lower_bound[buckets[b].size];
        if (greedy_upper_bound > count + other_lb) {
            bucket_betas[b] = greedy_upper_bound - count - other_lb + 1;
        } else {
            bucket_betas[b] = game->lower_bound[buckets[b].size] + 1;
        }
    }

    if (num_threads < 1) {
        num_threads = 1;
    }
    total_cost = count;

    if (active_buckets > 0) {
        BucketPool pool = {
            .game = game,
            .local_partition = local_partition,
            .buckets = buckets,
            .num_buckets = active_buckets,
            .bucket_betas = bucket_betas
        };
        pthread_t *threads;
        pthread_attr_t attr;
        int i;

        atomic_init(&pool.next_idx, 0);
        atomic_init(&pool.failed, false);
        pool.out_costs = calloc(active_buckets, sizeof(uint32_t));
        pool.out_guesses = calloc(active_buckets, sizeof(uint32_t));
        pool.out_nodes = calloc(active_buckets, sizeof(uint64_t));

        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, PTHREAD_STACK_SIZE);

        threads = malloc(num_threads * sizeof(pthread_t));
        for (i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], &attr, bucket_worker, &pool);
        }
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        free(threads);
        pthread_attr_destroy(&attr);

        if (atomic_load(&pool.failed)) {
            failed = true;
        }

        for (b = 0; b < active_buckets; b++) {
            if (pool.buckets[b].score == EXACT_MATCH) {
                continue;
            }
            if (pool.out_costs[b] >= bucket_betas[b]) {
                failed = true;
            }
            total_cost += pool.out_costs[b];
            total_nodes += pool.out_nodes[b];
        }
        free(pool.out_costs);
        free(pool.out_guesses);
        free(pool.out_nodes);
    }

    free(bucket_betas);
    free(local_partition);
    free(buckets);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    res.opener_idx = opener_idx;
    res.exact_total_cost = failed ? UINT32_MAX : total_cost;
    res.avg_guesses = failed ? 99.0 : ((double)total_cost / (double)count);
    res.is_exact = !failed;
    res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    res.nodes = total_nodes;
    return res;
}

static OpenerResult
evaluate_opener_sequential(Solver *solver, uint32_t opener_idx,
                           atomic_uint_fast32_t *global_best_cost)
{
    GameData *game = solver->game;
    struct timespec t0, t1;
    uint32_t count;
    uint32_t *local_partition;
    BucketInfo *buckets;
    uint32_t active_buckets;
    uint32_t root_lb;
    uint32_t b;
    OpenerResult res;
    uint32_t ceiling;
    uint32_t running_cost;
    uint32_t remaining_lb;
    bool pruned;
    uint64_t start_nodes;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    count = game->num_targets;
    local_partition = malloc(count * sizeof(uint32_t));
    buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);

    root_lb = count;
    for (b = 0; b < active_buckets; b++) {
        if (buckets[b].score != EXACT_MATCH) {
            root_lb += game->lower_bound[buckets[b].size];
        }
    }

    res.opener_idx = opener_idx;
    ceiling = atomic_load(global_best_cost);
    if (ceiling != UINT32_MAX && root_lb > ceiling) {
        free(local_partition);
        free(buckets);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        res.exact_total_cost = UINT32_MAX;
        res.avg_guesses = 99.0;
        res.is_exact = false;
        res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        res.nodes = 0;
        return res;
    }

    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_asc);

    running_cost = count;
    remaining_lb = root_lb - count;
    pruned = false;
    start_nodes = solver->nodes_visited;

    for (b = 0; b < active_buckets; b++) {
        uint32_t bucket_beta;
        uint32_t cost;

        if (buckets[b].score == EXACT_MATCH) {
            continue;
        }
        remaining_lb -= game->lower_bound[buckets[b].size];
        ceiling = atomic_load(global_best_cost);
        if (ceiling != UINT32_MAX && running_cost + remaining_lb >= ceiling) {
            pruned = true;
            break;
        }
        bucket_beta = (ceiling == UINT32_MAX) ? UINT32_MAX : (ceiling - running_cost - remaining_lb);
        cost = solve_subset(solver, &local_partition[buckets[b].offset], buckets[b].size,
                            buckets[b].hash1, buckets[b].hash2, bucket_beta, NULL, 0);
        if (cost >= bucket_beta) {
            pruned = true;
            break;
        }
        running_cost += cost;
    }

    free(local_partition);
    free(buckets);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    res.exact_total_cost = pruned ? UINT32_MAX : running_cost;
    res.avg_guesses = pruned ? 99.0 : (double)running_cost / (double)count;
    res.is_exact = !pruned;
    res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    res.nodes = solver->nodes_visited - start_nodes;
    return res;
}

/* -------------------------------------------------------------
 * Decision Tree & JSON Export
 * ------------------------------------------------------------- */

typedef struct TreeNode {
    char guess[WORD_LEN + 1];
    uint32_t num_targets;
    bool is_leaf;
    struct TreeNode *children[NUM_SCORES];
} TreeNode;

static void free_tree(TreeNode *node);

static TreeNode *
make_leaf_from_word(const char *word)
{
    TreeNode *n = calloc(1, sizeof(TreeNode));
    if (!n) {
        WORDLE_ERROR("Fatal: Out of memory allocating TreeNode\n");
        return NULL;
    }
    n->is_leaf = true;
    n->num_targets = 1;
    strncpy(n->guess, word, WORD_LEN);
    n->guess[WORD_LEN] = '\0';
    return n;
}

static TreeNode *
build_subtree_node(Solver *solver, const uint32_t *targets, uint32_t count, uint64_t h1, uint64_t h2);

static TreeNode *
build_subtree_node_with_guess(Solver *solver, const uint32_t *targets, uint32_t count,
                             uint32_t best_g)
{
    GameData *game = solver->game;
    TreeNode *node;
    const uint8_t *row;
    uint32_t hist[NUM_SCORES] = {0};
    uint32_t offsets[NUM_SCORES + 1];
    uint32_t *local_partition;
    uint32_t cur[NUM_SCORES];
    int s;
    uint32_t i;

    node = calloc(1, sizeof(TreeNode));
    if (!node) {
        WORDLE_ERROR("Fatal: Out of memory allocating TreeNode\n");
        return NULL;
    }
    node->num_targets = count;
    strcpy(node->guess, game->guesses[best_g].word);

    row = game->score_matrix + (size_t)best_g * game->num_targets;
    for (i = 0; i < count; i++) {
        hist[row[targets[i]]]++;
    }

    offsets[0] = 0;
    for (s = 0; s < NUM_SCORES; s++) {
        offsets[s + 1] = offsets[s] + hist[s];
    }

    local_partition = malloc(count * sizeof(uint32_t));
    if (!local_partition) {
        WORDLE_ERROR("Fatal: Out of memory allocating local partition\n");
        free_tree(node);
        return NULL;
    }
    memcpy(cur, offsets, sizeof(cur));
    for (i = 0; i < count; i++) {
        uint8_t sc = row[targets[i]];
        local_partition[cur[sc]++] = targets[i];
    }

    for (s = 0; s < NUM_SCORES; s++) {
        if (hist[s] == 0) {
            continue;
        }
        if (s == EXACT_MATCH) {
            node->children[EXACT_MATCH] = make_leaf_from_word(game->guesses[best_g].word);
        } else {
            uint32_t off = offsets[s];
            uint64_t bh1 = 0;
            uint64_t bh2 = 0;
            uint32_t j;
            for (j = 0; j < hist[s]; j++) {
                uint32_t tid = local_partition[off + j];
                bh1 ^= game->zobrist1[tid];
                bh2 ^= game->zobrist2[tid];
            }
            node->children[s] = build_subtree_node(solver, &local_partition[off], hist[s], bh1, bh2);
        }
        if (!node->children[s]) {
            free_tree(node);
            free(local_partition);
            return NULL;
        }
    }

    free(local_partition);
    return node;
}

static TreeNode *
build_subtree_node(Solver *solver, const uint32_t *targets, uint32_t count, uint64_t h1, uint64_t h2)
{
    uint32_t best_g;

    if (count == 0) {
        return NULL;
    }
    if (count == 1) {
        return make_leaf_from_word(solver->game->targets[targets[0]].word);
    }
    solve_subset(solver, targets, count, h1, h2, UINT32_MAX, &best_g, 0);
    return build_subtree_node_with_guess(solver, targets, count, best_g);
}

typedef struct {
    GameData *game;
    const uint32_t *local_partition;
    BucketInfo *buckets;
    uint32_t num_buckets;
    atomic_size_t next_idx;
    atomic_bool failed;
    TreeNode **out_nodes;
} TreeBucketPool;

static void *
tree_bucket_worker(void *arg)
{
    TreeBucketPool *pool = (TreeBucketPool *)arg;
    Solver solver;

    if (solver_init(&solver, pool->game) != 0) {
        atomic_store(&pool->failed, true);
        return NULL;
    }
    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        BucketInfo *bkt;
        TreeNode *n;

        if (idx >= pool->num_buckets) {
            break;
        }
        bkt = &pool->buckets[idx];
        if (bkt->score == EXACT_MATCH) {
            continue;
        }
        n = build_subtree_node(&solver, &pool->local_partition[bkt->offset],
                               bkt->size, bkt->hash1, bkt->hash2);
        if (!n) {
            atomic_store(&pool->failed, true);
            break;
        }
        pool->out_nodes[idx] = n;
    }
    solver_free(&solver);
    return NULL;
}

static TreeNode *
build_solution_tree(GameData *game, uint32_t opener_idx, int num_threads)
{
    uint32_t count = game->num_targets;
    uint32_t *local_partition;
    BucketInfo *buckets;
    uint32_t active_buckets;
    TreeNode *root;
    uint32_t b;

    local_partition = malloc(count * sizeof(uint32_t));
    buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    if (!local_partition || !buckets) {
        WORDLE_ERROR("Fatal: Out of memory allocating partition buffers\n");
        free(local_partition);
        free(buckets);
        return NULL;
    }

    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    if (num_threads < 1) {
        num_threads = 1;
    }
    root = calloc(1, sizeof(TreeNode));
    if (!root) {
        WORDLE_ERROR("Fatal: Out of memory allocating root TreeNode\n");
        free(local_partition);
        free(buckets);
        return NULL;
    }
    root->num_targets = count;
    strcpy(root->guess, game->guesses[opener_idx].word);

    if (active_buckets > 0) {
        TreeBucketPool pool = {
            .game = game,
            .local_partition = local_partition,
            .buckets = buckets,
            .num_buckets = active_buckets
        };
        pthread_t *threads;
        pthread_attr_t attr;
        int i;

        atomic_init(&pool.next_idx, 0);
        atomic_init(&pool.failed, false);
        pool.out_nodes = calloc(active_buckets, sizeof(TreeNode *));
        if (!pool.out_nodes) {
            WORDLE_ERROR("Fatal: Out of memory allocating tree bucket results\n");
            free(local_partition);
            free(buckets);
            free_tree(root);
            return NULL;
        }

        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, PTHREAD_STACK_SIZE);

        threads = malloc(num_threads * sizeof(pthread_t));
        if (!threads) {
            WORDLE_ERROR("Fatal: Out of memory allocating tree worker threads\n");
            pthread_attr_destroy(&attr);
            free(pool.out_nodes);
            free(local_partition);
            free(buckets);
            free_tree(root);
            return NULL;
        }
        for (i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], &attr, tree_bucket_worker, &pool);
        }
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        free(threads);
        pthread_attr_destroy(&attr);

        if (atomic_load(&pool.failed)) {
            WORDLE_ERROR("Fatal: Out of memory during tree construction\n");
            free(pool.out_nodes);
            free(local_partition);
            free(buckets);
            free_tree(root);
            return NULL;
        }

        for (b = 0; b < active_buckets; b++) {
            if (pool.buckets[b].score == EXACT_MATCH) {
                root->children[EXACT_MATCH] = make_leaf_from_word(game->guesses[opener_idx].word);
            } else {
                root->children[pool.buckets[b].score] = pool.out_nodes[b];
            }
        }
        free(pool.out_nodes);
    }

    free(local_partition);
    free(buckets);
    return root;
}

static void
free_tree(TreeNode *node)
{
    int s;
    if (!node) {
        return;
    }
    for (s = 0; s < NUM_SCORES; s++) {
        free_tree(node->children[s]);
    }
    free(node);
}

static void
write_node_json(const TreeNode *node, FILE *fp, int indent)
{
    char pad[128];
    int p = (indent > 60) ? 60 : indent;
    int i;
    bool first = true;
    int s;

    for (i = 0; i < p; i++) {
        pad[i] = ' ';
    }
    pad[p] = '\0';

    if (node->is_leaf) {
        fprintf(fp, "%s{\"guess\": \"%s\", \"num_targets\": %u, \"leaf\": true}",
                pad, node->guess, node->num_targets);
        return;
    }

    fprintf(fp, "%s{\n", pad);
    fprintf(fp, "%s  \"guess\": \"%s\",\n", pad, node->guess);
    fprintf(fp, "%s  \"num_targets\": %u,\n", pad, node->num_targets);
    fprintf(fp, "%s  \"branches\": {\n", pad);

    for (s = 0; s < NUM_SCORES; s++) {
        if (node->children[s]) {
            if (!first) {
                fprintf(fp, ",\n");
            }
            fprintf(fp, "%s    \"%d\":\n", pad, s);
            write_node_json(node->children[s], fp, indent + 6);
            first = false;
        }
    }
    fprintf(fp, "\n%s  }\n", pad);
    fprintf(fp, "%s}", pad);
}

static int
dump_tree_to_json(const TreeNode *root, const char *filepath, uint32_t num_targets,
                  uint32_t num_guesses, uint32_t exact_total_cost)
{
    FILE *fp;
    double avg_score;

    fp = fopen(filepath, "w");
    if (!fp) {
        WORDLE_ERROR("Error: Could not open '%s' for writing\n", filepath);
        return -1;
    }
    avg_score = (double)exact_total_cost / (double)num_targets;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": 1,\n");
    fprintf(fp, "  \"opener\": \"%s\",\n", root->guess);
    fprintf(fp, "  \"num_targets\": %u,\n", num_targets);
    fprintf(fp, "  \"num_guesses\": %u,\n", num_guesses);
    fprintf(fp, "  \"exact_total_guesses\": %u,\n", exact_total_cost);
    fprintf(fp, "  \"exact_avg_score\": %.17g,\n", avg_score);
    fprintf(fp, "  \"tree\":\n");
    write_node_json(root, fp, 4);
    fprintf(fp, "\n}\n");
    fclose(fp);
    WORDLE_INFO("Successfully dumped complete optimal solution tree to '%s'\n", filepath);
    return 0;
}

/* -------------------------------------------------------------
 * Opener Pool for --top / --all
 * ------------------------------------------------------------- */

typedef struct {
    uint32_t guess_idx;
    uint32_t sum_sq;
} HeuristicCandidate;

static int
compare_heuristic_asc(const void *a, const void *b)
{
    const HeuristicCandidate *ha = (const HeuristicCandidate *)a;
    const HeuristicCandidate *hb = (const HeuristicCandidate *)b;
    return (ha->sum_sq < hb->sum_sq) ? -1 : (ha->sum_sq > hb->sum_sq ? 1 : 0);
}

static int
compare_opener_results_asc(const void *a, const void *b)
{
    const OpenerResult *ra = (const OpenerResult *)a;
    const OpenerResult *rb = (const OpenerResult *)b;
    if (ra->exact_total_cost != rb->exact_total_cost) {
        return (ra->exact_total_cost < rb->exact_total_cost) ? -1 : 1;
    }
    if (ra->avg_guesses < rb->avg_guesses) {
        return -1;
    }
    if (ra->avg_guesses > rb->avg_guesses) {
        return 1;
    }
    if (ra->opener_idx != rb->opener_idx) {
        return (ra->opener_idx < rb->opener_idx) ? -1 : 1;
    }
    return 0;
}

typedef struct {
    GameData *game;
    const uint32_t *opener_indices;
    size_t num_openers;
    atomic_size_t next_idx;
    atomic_size_t completed;
    atomic_bool failed;
    OpenerResult *results;
    atomic_uint_fast32_t global_best_cost;
    pthread_mutex_t print_mutex;
    bool quiet;
    struct timespec start;
    FILE *log_fp;
    const char *save_tree_prefix;
    int num_threads;
} OpenerWorkPool;

static void *
opener_worker(void *arg)
{
    OpenerWorkPool *pool = (OpenerWorkPool *)arg;
    Solver solver;

    if (solver_init(&solver, pool->game) != 0) {
        atomic_store(&pool->failed, true);
        return NULL;
    }

    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        uint32_t g_idx;
        OpenerResult res;
        size_t completed;
        uint32_t prev_best;
        bool is_new_best;
        struct timespec now;
        double el;
        double wps;

        if (idx >= pool->num_openers) {
            break;
        }

        g_idx = pool->opener_indices[idx];
        res = evaluate_opener_sequential(&solver, g_idx, &pool->global_best_cost);
        pool->results[idx] = res;

        completed = atomic_fetch_add(&pool->completed, 1) + 1;

        pthread_mutex_lock(&pool->print_mutex);
        prev_best = atomic_load(&pool->global_best_cost);
        is_new_best = res.is_exact && res.exact_total_cost < prev_best;
        if (is_new_best) {
            atomic_store(&pool->global_best_cost, res.exact_total_cost);
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        el = (now.tv_sec - pool->start.tv_sec) + (now.tv_nsec - pool->start.tv_nsec) * 1e-9;
        wps = (el > 0.001) ? ((double)completed / el) : 0.0;

        if (pool->log_fp) {
            fprintf(pool->log_fp,
                    "{\"completed\": %zu, \"total\": %zu, \"word\": \"%s\", \"exact_total\": %u, \"avg_guesses\": %.5f, \"is_exact\": %s, \"time_sec\": %.4f, \"elapsed_sec\": %.2f, \"words_per_sec\": %.2f, \"nodes\": %llu, \"is_new_best\": %s}\n",
                    completed, pool->num_openers, pool->game->guesses[g_idx].word,
                    res.exact_total_cost, res.avg_guesses, res.is_exact ? "true" : "false",
                    res.time_sec, el, wps, (unsigned long long)res.nodes, is_new_best ? "true" : "false");
            fflush(pool->log_fp);
        }

        if (pool->save_tree_prefix && is_new_best) {
            char tree_path[512];
            TreeNode *root;
            snprintf(tree_path, sizeof(tree_path), "%s_%s.json", pool->save_tree_prefix, pool->game->guesses[g_idx].word);
            root = build_solution_tree(pool->game, g_idx, pool->num_threads);
            if (root) {
                dump_tree_to_json(root, tree_path, pool->game->num_targets, pool->game->num_guesses, res.exact_total_cost);
                free_tree(root);
                WORDLE_INFO("  -> [CHECKPOINT] Saved new best strategy tree to '%s'\n", tree_path);
            }
        }

        if (pool->quiet) {
            double pct = (double)completed * 100.0 / (double)pool->num_openers;
            uint32_t cur_best = atomic_load(&pool->global_best_cost);
            double best_avg = (cur_best == UINT32_MAX) ? 0.0 : (double)cur_best / (double)pool->game->num_targets;
            WORDLE_INFO("\r[%6zu/%6zu] (%5.1f%%) | Elapsed: %7.1fs | Current Best: %.5f avg",
                   completed, pool->num_openers, pct, el, best_avg);
            fflush(stdout);
        } else {
            if (res.is_exact) {
                WORDLE_INFO("[%5zu/%5zu] Opener: %-5s | Exact Avg: %.5f (%u total) | Time: %6.2fs | Nodes: %llu %s\n",
                       completed, pool->num_openers, pool->game->guesses[g_idx].word,
                       res.avg_guesses, res.exact_total_cost, res.time_sec,
                       (unsigned long long)res.nodes, is_new_best ? " <-- NEW BEST" : "");
            } else {
                WORDLE_INFO("[%5zu/%5zu] Opener: %-5s | Status: PRUNED (>= %u)        | Time: %6.2fs | Nodes: %llu\n",
                       completed, pool->num_openers, pool->game->guesses[g_idx].word,
                       prev_best, res.time_sec, (unsigned long long)res.nodes);
            }
            fflush(stdout);
        }
        pthread_mutex_unlock(&pool->print_mutex);
    }

    solver_free(&solver);
    return NULL;
}

static void
print_usage(const char *prog)
{
    WORDLE_INFO("Wordle Exact Solver (wordle_gemini)\n\n");
    WORDLE_INFO("Usage:\n");
    WORDLE_INFO("  %s [options]\n\n", prog);
    WORDLE_INFO("Options:\n");
    WORDLE_INFO("  --wordlist <path>     Path to words.txt (default: words.txt)\n");
    WORDLE_INFO("  --opener <word>       Evaluate a single opening word to exact optimality\n");
    WORDLE_INFO("  --subset <path|seq>   Solve an arbitrary candidate subset to exact optimality\n");
    WORDLE_INFO("                          path: one word per line ('-' = stdin)\n");
    WORDLE_INFO("                          seq:  word.score[.word.score...] (0=gray, 1=yellow, 2=green)\n");
    WORDLE_INFO("  --top <N>             Heuristically pre-rank openers, then exactly solve the top N\n");
    WORDLE_INFO("  --all                 Exactly solve every possible opening word\n");
    WORDLE_INFO("  --threads <N>         Number of worker threads (default: hardware concurrency)\n");
    WORDLE_INFO("  --max-memory <MB>     Maximum memory budget for caches (default: auto)\n");
    WORDLE_INFO("  --log <path>          Path to append real-time JSONL results\n");
    WORDLE_INFO("  --save-tree <prefix>  Checkpoint tree filename prefix whenever a new best opener is found\n");
    WORDLE_INFO("  --tree, --dump-tree <path> Dump optimal solution tree to JSON\n");
    WORDLE_INFO("  --quiet, -q           Compact progress output\n");
    WORDLE_INFO("  --help                Display this help message\n\n");
}

uint32_t
wordle_find_target(const GameData *game, const char *word)
{
    uint32_t t;
    for (t = 0; t < game->num_targets; t++) {
        if (strcasecmp(game->targets[t].word, word) == 0) {
            return t;
        }
    }
    return UINT32_MAX;
}

/* Load a candidate subset (one 5-letter target word per line) and map it to
 * target indices. Returns a malloc'd index array (caller frees) and sets
 * *out_count, *out_h1, *out_h2 (128-bit Zobrist of the subset). NULL on error. */
static uint32_t *
load_subset(const GameData *game, const char *path, uint32_t *out_count,
            uint64_t *out_h1, uint64_t *out_h2)
{
    FILE *fp;
    uint32_t cap = 64;
    uint32_t *indices;
    uint32_t count = 0;
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    char line[256];

    fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "r");
    if (!fp) {
        WORDLE_ERROR("Error: cannot open subset file '%s'\n", path);
        return NULL;
    }

    indices = malloc(cap * sizeof(uint32_t));
    if (!indices) {
        WORDLE_ERROR("Fatal: Out of memory allocating subset indices\n");
        if (fp != stdin) fclose(fp);
        return NULL;
    }

    while (fgets(line, sizeof(line), fp)) {
        char word[64];
        uint32_t t;
        uint32_t i;
        int matched;

        matched = sscanf(line, "%63s", word);
        if (matched < 1) {
            continue;
        }
        if (strlen(word) != WORD_LEN) {
            WORDLE_ERROR("Error: subset word '%s' is not %d letters\n", word, WORD_LEN);
            free(indices);
            if (fp != stdin) fclose(fp);
            return NULL;
        }
        t = wordle_find_target(game, word);
        if (t == UINT32_MAX) {
            WORDLE_ERROR("Error: subset word '%s' is not a valid target word\n", word);
            free(indices);
            if (fp != stdin) fclose(fp);
            return NULL;
        }
        for (i = 0; i < count; i++) {
            if (indices[i] == t) {
                WORDLE_ERROR("Error: duplicate subset word '%s'\n", word);
                free(indices);
                if (fp != stdin) fclose(fp);
                return NULL;
            }
        }
        if (count >= cap) {
            cap *= 2;
            indices = realloc(indices, cap * sizeof(uint32_t));
            if (!indices) {
                WORDLE_ERROR("Fatal: Out of memory expanding subset indices\n");
                if (fp != stdin) fclose(fp);
                return NULL;
            }
        }
        indices[count++] = t;
        h1 ^= game->zobrist1[t];
        h2 ^= game->zobrist2[t];
    }

    if (fp != stdin) {
        fclose(fp);
    }
    *out_count = count;
    *out_h1 = h1;
    *out_h2 = h2;
    return indices;
}

uint32_t
wordle_find_guess(const GameData *game, const char *word)
{
    uint32_t g;
    for (g = 0; g < game->num_guesses; g++) {
        if (strcasecmp(game->guesses[g].word, word) == 0) {
            return g;
        }
    }
    return UINT32_MAX;
}

/* True iff `s` looks like it was meant as a "word.score..." sequence: it
 * contains a '.' and the first '.'-separated token is exactly 5 letters.
 * Full validation happens in parse_sequence_subset, which reports precise
 * errors (so a malformed sequence gets a sequence error, not a "can't open
 * file" error). */
static bool
looks_like_sequence(const char *s)
{
    const char *dot = strchr(s, '.');
    const char *p;
    if (!dot) {
        return false;
    }
    if ((size_t)(dot - s) != WORD_LEN) {
        return false;
    }
    for (p = s; p < dot; p++) {
        if (!isalpha((unsigned char)*p)) {
            return false;
        }
    }
    return true;
}

/* Parse "word.score[.word.score...]" into the candidate subset it leaves:
 * start from the full target list, then for each (guess, score) pair keep
 * only targets scoring `score` against `guess`. Returns a malloc'd index
 * array (caller frees) and sets *out_count, *out_h1, *out_h2 (128-bit
 * Zobrist of the subset). NULL on error. */
static uint32_t *
parse_sequence_subset(GameData *game, const char *seq, uint32_t *out_count,
                      uint64_t *out_h1, uint64_t *out_h2)
{
    char *buf;
    char *saveptr = NULL;
    char *tok;
    char *word = NULL;
    uint32_t *subset;
    uint32_t count;
    uint32_t cap;
    uint32_t t;
    uint32_t i;
    uint32_t m;
    int part = 0;
    uint64_t h1 = 0;
    uint64_t h2 = 0;

    count = game->num_targets;
    cap = game->num_targets;
    subset = malloc(cap * sizeof(uint32_t));
    if (!subset) {
        WORDLE_ERROR("Fatal: Out of memory allocating sequence subset\n");
        return NULL;
    }
    for (t = 0; t < count; t++) {
        subset[t] = t;
    }

    buf = strdup(seq);
    if (!buf) {
        WORDLE_ERROR("Fatal: Out of memory duplicating sequence\n");
        free(subset);
        return NULL;
    }

    for (tok = strtok_r(buf, ".", &saveptr); tok; tok = strtok_r(NULL, ".", &saveptr)) {
        if (part % 2 == 0) {
            word = tok;
        } else {
            uint32_t sc = 0;
            uint32_t g;
            const uint8_t *row;

            if (strlen(tok) != WORD_LEN) {
                WORDLE_ERROR("Error: score '%s' is not %d digits\n", tok, WORD_LEN);
                free(buf); free(subset); return NULL;
            }
            for (i = 0; i < WORD_LEN; i++) {
                if (tok[i] < '0' || tok[i] > '2') {
                    WORDLE_ERROR("Error: score '%s' must contain only 0/1/2 digits\n", tok);
                    free(buf); free(subset); return NULL;
                }
                sc = sc * 3 + (uint32_t)(tok[i] - '0');
            }

            g = wordle_find_guess(game, word);
            if (g == UINT32_MAX) {
                WORDLE_ERROR("Error: guess word '%s' not found in word list\n", word);
                free(buf); free(subset); return NULL;
            }

            row = game->score_matrix + (size_t)g * game->num_targets;
            m = 0;
            for (i = 0; i < count; i++) {
                if (row[subset[i]] == sc) {
                    subset[m++] = subset[i];
                }
            }
            count = m;
            word = NULL;
        }
        part++;
    }
    free(buf);

    if (part % 2 != 0) {
        WORDLE_ERROR("Error: sequence must be word.score[.word.score...] pairs\n");
        free(subset);
        return NULL;
    }

    for (t = 0; t < count; t++) {
        h1 ^= game->zobrist1[subset[t]];
        h2 ^= game->zobrist2[subset[t]];
    }
    *out_count = count;
    *out_h1 = h1;
    *out_h2 = h2;
    return subset;
}

/* -------------------------------------------------------------
 * Library API
 *
 * The CLI below uses the same entry points, so a shared-library build can
 * expose exactly this surface. GameData is read-only during a solve, the
 * shared transposition table is lock-free, and wordle_subset_solve allocates
 * its own Solver state, so concurrent solves are safe (each uses one core
 * and its own local TT). Defaults to WORDLE_LOG_QUIET: no stdout/stderr output
 * unless the caller raises the log level.
 * ------------------------------------------------------------- */

GameData *
wordle_init(const char *wordlist_path, int num_threads, uint64_t max_memory_mb)
{
    GameData *game = calloc(1, sizeof(GameData));
    if (!game) {
        WORDLE_ERROR("Fatal: Out of memory allocating GameData\n");
        return NULL;
    }
    if (load_wordlist(wordlist_path, game) != 0) {
        free_game_data(game);
        free(game);
        return NULL;
    }
    game->max_candidates = UINT32_MAX;
    if (init_game_data(game, num_threads, max_memory_mb) != 0) {
        free_game_data(game);
        free(game);
        return NULL;
    }
    return game;
}

void
wordle_free(GameData *game)
{
    if (game) {
        free_game_data(game);
        free(game);
    }
}

uint32_t
wordle_num_targets(const GameData *game)
{
    return game->num_targets;
}

uint32_t
wordle_num_guesses(const GameData *game)
{
    return game->num_guesses;
}

const char *
wordle_target_word(const GameData *game, uint32_t t)
{
    return game->targets[t].word;
}

const char *
wordle_guess_word(const GameData *game, uint32_t g)
{
    return game->guesses[g].word;
}

void
wordle_subset_hash(const GameData *game, const uint32_t *targets, uint32_t count,
               uint64_t *out_h1, uint64_t *out_h2)
{
    uint64_t h1 = 0;
    uint64_t h2 = 0;
    uint32_t i;
    for (i = 0; i < count; i++) {
        h1 ^= game->zobrist1[targets[i]];
        h2 ^= game->zobrist2[targets[i]];
    }
    *out_h1 = h1;
    *out_h2 = h2;
}

/* Solve `targets` (target indices) to exact optimality. Returns the exact
 * total cost, or UINT32_MAX on error (best_guess is then UINT32_MAX). */
uint32_t
wordle_subset_solve(GameData *game, const uint32_t *targets, uint32_t count,
                uint64_t h1, uint64_t h2, uint32_t *best_guess)
{
    Solver solver;
    uint32_t cost;

    if (!game || !targets || count == 0) {
        if (best_guess) {
            *best_guess = UINT32_MAX;
        }
        return UINT32_MAX;
    }
    if (solver_init(&solver, game) != 0) {
        if (best_guess) {
            *best_guess = UINT32_MAX;
        }
        return UINT32_MAX;
    }
    solver.max_candidates = UINT32_MAX;
    cost = solve_subset(&solver, targets, count, h1, h2, UINT32_MAX, best_guess, 0);
    solver_free(&solver);
    return cost;
}

int
main(int argc, char **argv)
{
    const char *wordlist_path = "words.txt";
    const char *single_opener = NULL;
    const char *tree_dump_path = NULL;
    const char *subset_path = NULL;
    int top_n = 10;
    bool search_all = false;
    bool quiet = false;
    int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    uint32_t max_candidates = 100;
    const char *log_path = NULL;
    const char *save_tree_prefix = NULL;
    uint64_t max_memory_mb = 0;
    int i;
    GameData game = {0};
    struct timespec t0, t1;
    HeuristicCandidate *cands;
    uint32_t g;
    uint32_t t;
    size_t count_to_eval;
    uint32_t *openers_to_eval;
    uint32_t initial_seed;
    FILE *log_fp = NULL;
    OpenerWorkPool pool;
    pthread_t *threads;
    uint32_t best_total;
    uint32_t best_opener_idx;
    double best_avg;

    if (num_threads < 1) {
        num_threads = 4;
    }
    wordle_log_set_level(WORDLE_LOG_INFO);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wordlist") == 0 && i + 1 < argc) {
            wordlist_path = argv[++i];
        } else if (strcmp(argv[i], "--opener") == 0 && i + 1 < argc) {
            single_opener = argv[++i];
        } else if (strcmp(argv[i], "--subset") == 0 && i + 1 < argc) {
            subset_path = argv[++i];
        } else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc) {
            top_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all") == 0) {
            search_all = true;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--max-memory") == 0 || strcmp(argv[i], "--tt-mem") == 0) && i + 1 < argc) {
            max_memory_mb = (uint64_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "--candidates") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            max_candidates = (uint32_t)atoi(argv[++i]);
            if (max_candidates == 0) {
                max_candidates = UINT32_MAX;
            }
        } else if (strcmp(argv[i], "--exhaustive") == 0) {
            max_candidates = UINT32_MAX;
        } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            log_path = argv[++i];
        } else if (strcmp(argv[i], "--save-tree") == 0 && i + 1 < argc) {
            save_tree_prefix = argv[++i];
        } else if (strcmp(argv[i], "--tree") == 0 || strcmp(argv[i], "--dump-tree") == 0) {
            if (i + 1 < argc) {
                tree_dump_path = argv[++i];
            }
        } else if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    WORDLE_INFO("=================================================================\n");
    WORDLE_INFO("      WORDLE OPTIMAL FULL-TREE SOLVER (wordle_gemini)\n");
    WORDLE_INFO("=================================================================\n");

    if (load_wordlist(wordlist_path, &game) != 0) {
        return 1;
    }
    game.max_candidates = max_candidates;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    WORDLE_INFO("Precomputing %u x %u score matrix using %d threads...\n", game.num_guesses, game.num_targets, num_threads);
    if (init_game_data(&game, num_threads, max_memory_mb) != 0) {
        free_game_data(&game);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    WORDLE_INFO("Ready in %.3f seconds (%.1f MB matrix).\n\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9,
           (double)((size_t)game.num_guesses * game.num_targets) / (1024.0 * 1024.0));

    if (subset_path) {
        uint32_t *subset;
        uint32_t count;
        uint64_t h1;
        uint64_t h2;
        Solver solver;
        uint32_t best_g = UINT32_MAX;
        uint32_t cost;
        struct timespec st0, st1;
        double solve_sec;

        subset = (strcmp(subset_path, "-") != 0 && looks_like_sequence(subset_path))
                     ? parse_sequence_subset(&game, subset_path, &count, &h1, &h2)
                     : load_subset(&game, subset_path, &count, &h1, &h2);
        if (!subset) {
            free_game_data(&game);
            return 1;
        }
        if (count == 0) {
            WORDLE_ERROR("Error: subset is empty\n");
            free(subset);
            free_game_data(&game);
            return 1;
        }

        if (solver_init(&solver, &game) != 0) {
            free(subset);
            free_game_data(&game);
            return 1;
        }
        solver.max_candidates = UINT32_MAX;

        WORDLE_INFO("Solving subset of %u target word(s) to exact optimality...\n", count);
        clock_gettime(CLOCK_MONOTONIC, &st0);
        cost = solve_subset(&solver, subset, count, h1, h2, UINT32_MAX, &best_g, 0);
        clock_gettime(CLOCK_MONOTONIC, &st1);
        solve_sec = (st1.tv_sec - st0.tv_sec) + (st1.tv_nsec - st0.tv_nsec) * 1e-9;

        WORDLE_INFO("{\n");
        WORDLE_INFO("  \"mode\": \"subset\",\n");
        WORDLE_INFO("  \"num_targets\": %u,\n", count);
        if (best_g != UINT32_MAX) {
            WORDLE_INFO("  \"best_guess\": \"%s\",\n", game.guesses[best_g].word);
        } else {
            WORDLE_INFO("  \"best_guess\": null,\n");
        }
        WORDLE_INFO("  \"exact_cost\": %u,\n", cost);
        WORDLE_INFO("  \"avg_score\": %.17g,\n", (double)cost / (double)count);
        WORDLE_INFO("  \"nodes\": %llu,\n", (unsigned long long)solver.nodes_visited);
        WORDLE_INFO("  \"time_sec\": %.4f\n", solve_sec);
        WORDLE_INFO("}\n");

        if (tree_dump_path) {
            TreeNode *root;
            if (count == 1) {
                root = make_leaf_from_word(game.targets[subset[0]].word);
            } else {
                root = build_subtree_node_with_guess(&solver, subset, count, best_g);
            }
            if (root) {
                dump_tree_to_json(root, tree_dump_path, count, game.num_guesses, cost);
                free_tree(root);
            }
        }

        solver_free(&solver);
        free(subset);
        free_game_data(&game);
        return 0;
    }

    if (single_opener) {
        uint32_t opener_idx = UINT32_MAX;
        OpenerResult res;

        for (g = 0; g < game.num_guesses; g++) {
            if (strcasecmp(game.guesses[g].word, single_opener) == 0) {
                opener_idx = g;
                break;
            }
        }
        if (opener_idx == UINT32_MAX) {
            WORDLE_ERROR("Error: Opening word '%s' not found in word list.\n", single_opener);
            return 1;
        }

        WORDLE_INFO("Evaluating opener: '%s' to exact mathematical optimality...\n", game.guesses[opener_idx].word);
        res = evaluate_opener_parallel(&game, opener_idx, num_threads);

        if (res.is_exact) {
            WORDLE_INFO("\n======================= EXACT RESULT =======================\n");
            WORDLE_INFO("  Opener:               %-5s\n", game.guesses[opener_idx].word);
            WORDLE_INFO("  Total Target Words:   %u\n", game.num_targets);
            WORDLE_INFO("  Exact Total Guesses:  %u\n", res.exact_total_cost);
            WORDLE_INFO("  Exact Average Score:  %.5f guesses/game\n", res.avg_guesses);
            WORDLE_INFO("  Computation Time:     %.3f seconds\n", res.time_sec);
            WORDLE_INFO("  Tree Nodes Visited:   %llu\n", (unsigned long long)res.nodes);
            WORDLE_INFO("============================================================\n");
        } else {
            WORDLE_INFO("\n====================== PRUNED RESULT =======================\n");
            WORDLE_INFO("  Opener:               %-5s (Pruned)\n", game.guesses[opener_idx].word);
            WORDLE_INFO("  Total Target Words:   %u\n", game.num_targets);
            WORDLE_INFO("  Status:               Exceeds Aspiration Bound\n");
            WORDLE_INFO("  Computation Time:     %.3f seconds\n", res.time_sec);
            WORDLE_INFO("  Tree Nodes Visited:   %llu\n", (unsigned long long)res.nodes);
            WORDLE_INFO("============================================================\n");
        }

        if (log_path) {
            FILE *sl_fp = fopen(log_path, "a");
            if (sl_fp) {
                fprintf(sl_fp,
                        "{\"completed\": 1, \"total\": 1, \"word\": \"%s\", \"exact_total\": %u, \"avg_guesses\": %.5f, \"is_exact\": %s, \"time_sec\": %.4f, \"nodes\": %llu, \"is_new_best\": %s}\n",
                        game.guesses[opener_idx].word, res.exact_total_cost, res.avg_guesses,
                        res.is_exact ? "true" : "false", res.time_sec, (unsigned long long)res.nodes,
                        res.is_exact ? "true" : "false");
                fclose(sl_fp);
            }
        }

        if (tree_dump_path) {
            TreeNode *root;
            WORDLE_INFO("\nBuilding full decision tree for opener '%s'...\n", game.guesses[opener_idx].word);
            root = build_solution_tree(&game, opener_idx, num_threads);
            if (root) {
                dump_tree_to_json(root, tree_dump_path, game.num_targets, game.num_guesses, res.exact_total_cost);
                free_tree(root);
            }
        }

        free_game_data(&game);
        return 0;
    }

    WORDLE_INFO("Ranking opening guesses by partition variance (heuristic pre-filter)...\n");
    cands = malloc(game.num_guesses * sizeof(HeuristicCandidate));
    for (g = 0; g < game.num_guesses; g++) {
        uint32_t chist[NUM_SCORES] = {0};
        const uint8_t *row = game.score_matrix + (size_t)g * game.num_targets;
        uint32_t sum_sq = 0;
        int s;
        for (t = 0; t < game.num_targets; t++) {
            chist[row[t]]++;
        }
        for (s = 0; s < NUM_SCORES; s++) {
            sum_sq += chist[s] * chist[s];
        }
        cands[g] = (HeuristicCandidate){ .guess_idx = g, .sum_sq = sum_sq };
    }
    qsort(cands, game.num_guesses, sizeof(HeuristicCandidate), compare_heuristic_asc);

    count_to_eval = search_all ? game.num_guesses : (size_t)top_n;
    if (count_to_eval > game.num_guesses) {
        count_to_eval = game.num_guesses;
    }

    openers_to_eval = malloc(count_to_eval * sizeof(uint32_t));
    for (i = 0; i < (int)count_to_eval; i++) {
        openers_to_eval[i] = cands[i].guess_idx;
    }
    free(cands);

    initial_seed = compute_opener_greedy_upper_bound(&game, openers_to_eval[0]);
    WORDLE_INFO("Initial aspiration seed for top opener '%s': %u total guesses (%.4f avg)\n",
           game.guesses[openers_to_eval[0]].word, initial_seed, (double)initial_seed / game.num_targets);

    WORDLE_INFO("Evaluating %zu opener(s) in parallel using %d threads%s...\n\n",
           count_to_eval, num_threads, quiet ? " (quiet mode)" : "");

    if (log_path) {
        log_fp = fopen(log_path, "a");
        if (!log_fp) {
            WORDLE_ERROR("Warning: Unable to open log file '%s' for writing\n", log_path);
        }
    }

    pool.game = &game;
    pool.opener_indices = openers_to_eval;
    pool.num_openers = count_to_eval;
    pool.quiet = quiet;
    pool.log_fp = log_fp;
    pool.save_tree_prefix = save_tree_prefix;
    pool.num_threads = num_threads;

    clock_gettime(CLOCK_MONOTONIC, &pool.start);
    atomic_init(&pool.next_idx, 0);
    atomic_init(&pool.completed, 0);
    atomic_init(&pool.failed, false);
    atomic_init(&pool.global_best_cost, initial_seed);
    pthread_mutex_init(&pool.print_mutex, NULL);
    pool.results = calloc(count_to_eval, sizeof(OpenerResult));

    {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, PTHREAD_STACK_SIZE);

        threads = malloc(num_threads * sizeof(pthread_t));
        for (i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], &attr, opener_worker, &pool);
        }
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        free(threads);
        pthread_attr_destroy(&attr);
    }
    pthread_mutex_destroy(&pool.print_mutex);
    if (log_fp) {
        fclose(log_fp);
    }
    if (atomic_load(&pool.failed)) {
        WORDLE_ERROR("Fatal: Out of memory during opener evaluation\n");
        free(openers_to_eval);
        free(pool.results);
        free_game_data(&game);
        return 1;
    }
    if (quiet) {
        WORDLE_INFO("\n");
    }

    qsort(pool.results, count_to_eval, sizeof(OpenerResult), compare_opener_results_asc);

    best_total = pool.results[0].exact_total_cost;
    best_opener_idx = pool.results[0].opener_idx;
    best_avg = pool.results[0].avg_guesses;

    if (best_total == UINT32_MAX) {
        best_total = initial_seed;
        best_opener_idx = openers_to_eval[0];
        best_avg = (double)best_total / (double)game.num_targets;
    }

    WORDLE_INFO("\n======================== TOP RESULTS ========================\n");
    WORDLE_INFO(" Rank | Opener | Exact Total | Exact Average | Time\n");
    WORDLE_INFO("------+--------+-------------+---------------+----------\n");
    for (i = 0; i < (int)count_to_eval && i < 20; i++) {
        if (pool.results[i].is_exact) {
            WORDLE_INFO(" %4d | %-6s | %11u | %11.5f | %6.2fs\n",
                   i + 1, game.guesses[pool.results[i].opener_idx].word,
                   pool.results[i].exact_total_cost, pool.results[i].avg_guesses, pool.results[i].time_sec);
        } else {
            WORDLE_INFO(" %4d | %-6s |    (pruned) |      (pruned) | %6.2fs\n",
                   i + 1, game.guesses[pool.results[i].opener_idx].word,
                   pool.results[i].time_sec);
        }
    }
    WORDLE_INFO("=============================================================\n");
    WORDLE_INFO("%s OPENER: '%s' with exact average score: %.5f (%u total guesses)\n",
           search_all ? "GLOBAL OPTIMAL" : "BEST OF TOP-N",
           game.guesses[best_opener_idx].word, best_avg, best_total);
    WORDLE_INFO("=============================================================\n");

    if (tree_dump_path) {
        TreeNode *root;
        WORDLE_INFO("\nBuilding full decision tree for winning opener '%s'...\n", game.guesses[best_opener_idx].word);
        root = build_solution_tree(&game, best_opener_idx, num_threads);
        if (root) {
            dump_tree_to_json(root, tree_dump_path, game.num_targets, game.num_guesses, best_total);
            free_tree(root);
        }
    }

    free(openers_to_eval);
    free(pool.results);
    free_game_data(&game);
    return 0;
}
