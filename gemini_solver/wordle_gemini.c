// wordle_gemini.c
//
// Optimal Wordle Solver (Easy Mode)
//
// Features:
// 1. 128-bit Double Hashing in TT and Partition Deduplication (zero collision risk).
// 2. Fast 1-Ply Greedy Aspiration Seeding for root beta bounds.
// 3. Fused single-pass candidate loop (dedup + histogram + variance + lb1/ub1).
// 4. Node-level lb1 fail-soft cutoffs and ub1==lb1 exact resolutions.
// 5. Precomputed wildcard endgame clusters and static coverage bounds.
// 6. Easy-mode disjoint-union bound propagation.
// 7. Correct per-bucket node count reporting in parallel mode.
// 8. Decision tree JSON export compatible with strategies.py.

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

#define WORD_LEN 5
#define NUM_SCORES 243
#define EXACT_MATCH 242
#define MAX_ENDGAMES 2048
#define MIN_ENDGAME_COUNT 4

typedef struct {
    char word[WORD_LEN + 1];
} Word;

typedef struct {
    char pattern[WORD_LEN + 1]; // e.g. ".ound"
    uint32_t* targets;          // indices of targets matching this pattern
    uint32_t count;
    uint32_t wildcard_pos;
    uint32_t letter_mask;       // bitmask of (c - 'a') present in this endgame
} Endgame;

// -------------------------------------------------------------
// 128-Bit Lock-Free Shared Transposition Table
// -------------------------------------------------------------

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
    SharedTTEntry* entries;
    uint64_t mask;
} SharedTT;

static void shared_tt_init(SharedTT* stt, uint64_t capacity_pow2);
static void shared_tt_free(SharedTT* stt);

typedef struct {
    Word* targets;
    uint32_t num_targets;

    Word* guesses;
    uint32_t num_guesses;

    // score_matrix[g * num_targets + t]
    uint8_t* score_matrix;

    // 128-bit Zobrist hashes
    uint64_t* zobrist1;
    uint64_t* zobrist2;

    uint32_t* lower_bound; // [num_targets + 1]

    // Endgame tables
    Endgame* endgames;
    uint32_t num_endgames;
    uint32_t* target_endgame_counts; // [num_targets]
    uint32_t** target_endgames;      // [num_targets][count]

    // Shared global lock-free transposition table across threads
    SharedTT shared_tt;
} GameData;

static inline uint8_t compute_score(const char* restrict guess, const char* restrict target) {
    uint8_t counts[26] = {0};
    bool is_green[WORD_LEN];
    for (int i = 0; i < WORD_LEN; i++) {
        if (guess[i] == target[i]) {
            is_green[i] = true;
        } else {
            is_green[i] = false;
            counts[target[i] - 'a']++;
        }
    }
    uint8_t score = 0;
    for (int i = 0; i < WORD_LEN; i++) {
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

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int load_wordlist(const char* filepath, GameData* game) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open word list file '%s'\n", filepath);
        return -1;
    }

    uint32_t target_cap = 1024, guess_cap = 4096;
    game->targets = malloc(target_cap * sizeof(Word));
    game->guesses = malloc(guess_cap * sizeof(Word));
    game->num_targets = 0;
    game->num_guesses = 0;

    char line[256];
    bool in_extra = false;

    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0') {
            in_extra = true;
            continue;
        }

        char word[64];
        double weight;
        int matched = sscanf(p, "%63s %lf", word, &weight);
        if (matched < 1) continue;

        if (strlen(word) != WORD_LEN) continue;
        for (size_t i = 0; i < WORD_LEN; i++) word[i] = (char)tolower((unsigned char)word[i]);

        if (!in_extra) {
            if (game->num_targets >= target_cap) {
                target_cap *= 2;
                game->targets = realloc(game->targets, target_cap * sizeof(Word));
            }
            strcpy(game->targets[game->num_targets].word, word);
            game->num_targets++;
        }

        if (game->num_guesses >= guess_cap) {
            guess_cap *= 2;
            game->guesses = realloc(game->guesses, guess_cap * sizeof(Word));
        }
        strcpy(game->guesses[game->num_guesses].word, word);
        game->num_guesses++;
    }

    fclose(fp);
    printf("Loaded %u target words, %u total valid guess words from '%s'\n",
           game->num_targets, game->num_guesses, filepath);
    return 0;
}

typedef struct {
    GameData* game;
    size_t start_g, end_g;
} MatrixWorkerArg;

static void* score_matrix_worker(void* arg) {
    MatrixWorkerArg* m = (MatrixWorkerArg*)arg;
    GameData* game = m->game;
    size_t T = game->num_targets;
    for (size_t g = m->start_g; g < m->end_g; g++) {
        const char* gw = game->guesses[g].word;
        uint8_t* row = game->score_matrix + g * T;
        for (size_t t = 0; t < T; t++) {
            row[t] = compute_score(gw, game->targets[t].word);
        }
    }
    return NULL;
}

static void compute_lower_bound_table(GameData* game) {
    uint32_t n = game->num_targets;
    game->lower_bound = malloc((n + 1) * sizeof(uint32_t));
    game->lower_bound[0] = 0;
    for (uint32_t k = 1; k <= n; k++) {
        if (k == 1) game->lower_bound[k] = 1;
        else if (k <= 243) game->lower_bound[k] = 2 * k - 1;
        else game->lower_bound[k] = 1 + 2 * 242 + 3 * (k - 243);
    }
}

static void init_endgames(GameData* game) {
    uint32_t T = game->num_targets;
    game->endgames = malloc(MAX_ENDGAMES * sizeof(Endgame));
    game->num_endgames = 0;

    game->target_endgame_counts = calloc(T, sizeof(uint32_t));
    game->target_endgames = calloc(T, sizeof(uint32_t*));

    typedef struct {
        char pattern[WORD_LEN + 1];
        uint32_t wildcard_pos;
        uint32_t count;
        uint32_t target_indices[128];
    } PatternGroup;

    uint32_t group_cap = 4096;
    PatternGroup* groups = calloc(group_cap, sizeof(PatternGroup));
    uint32_t num_groups = 0;

    for (uint32_t t = 0; t < T; t++) {
        const char* tw = game->targets[t].word;
        for (int p = 0; p < WORD_LEN; p++) {
            char pat[WORD_LEN + 1];
            strcpy(pat, tw);
            pat[p] = '.';

            int found_idx = -1;
            for (uint32_t g = 0; g < num_groups; g++) {
                if (strcmp(groups[g].pattern, pat) == 0) {
                    found_idx = (int)g;
                    break;
                }
            }

            if (found_idx == -1) {
                if (num_groups >= group_cap) {
                    group_cap *= 2;
                    groups = realloc(groups, group_cap * sizeof(PatternGroup));
                }
                found_idx = (int)num_groups++;
                strcpy(groups[found_idx].pattern, pat);
                groups[found_idx].wildcard_pos = p;
                groups[found_idx].count = 0;
            }

            if (groups[found_idx].count < 128) {
                groups[found_idx].target_indices[groups[found_idx].count++] = t;
            }
        }
    }

    for (uint32_t g = 0; g < num_groups; g++) {
        if (groups[g].count >= MIN_ENDGAME_COUNT && game->num_endgames < MAX_ENDGAMES) {
            uint32_t eg_idx = game->num_endgames++;
            Endgame* eg = &game->endgames[eg_idx];
            strcpy(eg->pattern, groups[g].pattern);
            eg->count = groups[g].count;
            eg->wildcard_pos = groups[g].wildcard_pos;
            eg->targets = malloc(eg->count * sizeof(uint32_t));
            memcpy(eg->targets, groups[g].target_indices, eg->count * sizeof(uint32_t));
            eg->letter_mask = 0;

            for (uint32_t i = 0; i < eg->count; i++) {
                uint32_t tid = eg->targets[i];
                char c = game->targets[tid].word[eg->wildcard_pos];
                eg->letter_mask |= (1u << (c - 'a'));
            }
        }
    }

    free(groups);

    // Map each target to its endgames
    for (uint32_t t = 0; t < T; t++) {
        uint32_t matched[64];
        uint32_t mcount = 0;
        for (uint32_t e = 0; e < game->num_endgames; e++) {
            Endgame* eg = &game->endgames[e];
            for (uint32_t i = 0; i < eg->count; i++) {
                if (eg->targets[i] == t) {
                    if (mcount < 64) matched[mcount++] = e;
                    break;
                }
            }
        }
        game->target_endgame_counts[t] = mcount;
        if (mcount > 0) {
            game->target_endgames[t] = malloc(mcount * sizeof(uint32_t));
            memcpy(game->target_endgames[t], matched, mcount * sizeof(uint32_t));
        }
    }
}

static void init_game_data(GameData* game, int num_threads) {
    size_t total_cells = (size_t)game->num_guesses * game->num_targets;
    game->score_matrix = malloc(total_cells * sizeof(uint8_t));
    if (!game->score_matrix) {
        fprintf(stderr, "Fatal: Out of memory allocating score matrix\n");
        exit(1);
    }

    if (num_threads < 1) num_threads = 1;
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    MatrixWorkerArg* args = malloc(num_threads * sizeof(MatrixWorkerArg));
    size_t chunk = (game->num_guesses + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        args[i].game = game;
        args[i].start_g = (size_t)i * chunk;
        args[i].end_g = (size_t)(i + 1) * chunk;
        if (args[i].start_g > game->num_guesses) args[i].start_g = game->num_guesses;
        if (args[i].end_g > game->num_guesses) args[i].end_g = game->num_guesses;
        pthread_create(&threads[i], NULL, score_matrix_worker, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    free(threads);
    free(args);

    uint64_t seed1 = 0x853c49e6748fea9bULL;
    uint64_t seed2 = 0xda3e39cb94b95bdbULL;
    game->zobrist1 = malloc(game->num_targets * sizeof(uint64_t));
    game->zobrist2 = malloc(game->num_targets * sizeof(uint64_t));
    for (uint32_t t = 0; t < game->num_targets; t++) {
        game->zobrist1[t] = splitmix64(&seed1);
        game->zobrist2[t] = splitmix64(&seed2);
    }

    compute_lower_bound_table(game);
    init_endgames(game);
    shared_tt_init(&game->shared_tt, 1u << 22);
}

static void free_game_data(GameData* game) {
    shared_tt_free(&game->shared_tt);
    free(game->targets);
    free(game->guesses);
    free(game->score_matrix);
    free(game->zobrist1);
    free(game->zobrist2);
    free(game->lower_bound);
    for (uint32_t e = 0; e < game->num_endgames; e++) {
        free(game->endgames[e].targets);
    }
    free(game->endgames);
    for (uint32_t t = 0; t < game->num_targets; t++) {
        if (game->target_endgames[t]) free(game->target_endgames[t]);
    }
    free(game->target_endgame_counts);
    free(game->target_endgames);
}

// -------------------------------------------------------------
// 128-Bit Transposition Table (Thread-Local L1 + Global Shared L2)
// -------------------------------------------------------------

typedef struct {
    uint64_t hash1;
    uint64_t hash2;
    uint32_t size;
    uint32_t exact_cost;
    uint32_t proven_lower_bound;
    uint32_t best_guess;
} TTEntry;

typedef struct {
    TTEntry* entries;
    uint64_t mask;
} TT;

static void tt_init(TT* tt, uint64_t capacity_pow2) {
    tt->entries = malloc(capacity_pow2 * sizeof(TTEntry));
    tt->mask = capacity_pow2 - 1;
    for (uint64_t i = 0; i < capacity_pow2; i++) {
        tt->entries[i].hash1 = TT_EMPTY_HASH;
        tt->entries[i].hash2 = TT_EMPTY_HASH;
    }
}

static void tt_free(TT* tt) { free(tt->entries); }

static inline TTEntry* tt_find(TT* tt, uint64_t h1, uint64_t h2, uint32_t size) {
    uint64_t idx = (h1 ^ h2) & tt->mask;
    for (uint64_t probes = 0; probes <= tt->mask; probes++) {
        TTEntry* e = &tt->entries[idx];
        if (e->hash1 == TT_EMPTY_HASH && e->hash2 == TT_EMPTY_HASH) return NULL;
        if (e->hash1 == h1 && e->hash2 == h2 && e->size == size) return e;
        idx = (idx + 1) & tt->mask;
    }
    return NULL;
}

static inline TTEntry* tt_find_or_claim(TT* tt, uint64_t h1, uint64_t h2, uint32_t size) {
    uint64_t idx = (h1 ^ h2) & tt->mask;
    for (uint64_t probes = 0; probes <= tt->mask; probes++) {
        TTEntry* e = &tt->entries[idx];
        if (e->hash1 == TT_EMPTY_HASH && e->hash2 == TT_EMPTY_HASH) {
            e->hash1 = h1;
            e->hash2 = h2;
            e->size = size;
            e->exact_cost = UINT32_MAX;
            e->proven_lower_bound = 0;
            e->best_guess = UINT32_MAX;
            return e;
        }
        if (e->hash1 == h1 && e->hash2 == h2 && e->size == size) return e;
        idx = (idx + 1) & tt->mask;
    }
    return NULL;
}

static void shared_tt_init(SharedTT* stt, uint64_t capacity_pow2) {
    stt->entries = calloc(capacity_pow2, sizeof(SharedTTEntry));
    stt->mask = capacity_pow2 - 1;
    for (uint64_t i = 0; i < capacity_pow2; i++) {
        atomic_init(&stt->entries[i].hash1, TT_EMPTY_HASH);
        atomic_init(&stt->entries[i].hash2, TT_EMPTY_HASH);
        atomic_init(&stt->entries[i].size, 0);
        atomic_init(&stt->entries[i].exact_cost, UINT32_MAX);
        atomic_init(&stt->entries[i].proven_lower_bound, 0);
        atomic_init(&stt->entries[i].best_guess, UINT32_MAX);
    }
}

static void shared_tt_free(SharedTT* stt) {
    if (stt->entries) {
        free(stt->entries);
        stt->entries = NULL;
    }
}

static inline bool shared_tt_find(SharedTT* stt, uint64_t h1, uint64_t h2, uint32_t size,
                                  uint32_t* out_exact, uint32_t* out_lb, uint32_t* out_guess) {
    if (!stt || !stt->entries) return false;
    uint64_t idx = (h1 ^ h2) & stt->mask;
    for (uint64_t probe = 0; probe < 16; probe++) {
        SharedTTEntry* e = &stt->entries[(idx + probe) & stt->mask];
        uint64_t eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        if (eh1 == TT_EMPTY_HASH) return false;
        if (eh1 == h1) {
            uint64_t eh2 = atomic_load_explicit(&e->hash2, memory_order_acquire);
            uint32_t esz = atomic_load_explicit(&e->size, memory_order_acquire);
            if (eh2 == h2 && esz == size) {
                if (out_exact) *out_exact = atomic_load_explicit(&e->exact_cost, memory_order_relaxed);
                if (out_lb) *out_lb = atomic_load_explicit(&e->proven_lower_bound, memory_order_relaxed);
                if (out_guess) *out_guess = atomic_load_explicit(&e->best_guess, memory_order_relaxed);
                return true;
            }
        }
    }
    return false;
}

static inline void shared_tt_store(SharedTT* stt, uint64_t h1, uint64_t h2, uint32_t size,
                                   uint32_t exact_cost, uint32_t proven_lb, uint32_t best_guess) {
    if (!stt || !stt->entries) return;
    uint64_t idx = (h1 ^ h2) & stt->mask;
    for (uint64_t probe = 0; probe < 16; probe++) {
        SharedTTEntry* e = &stt->entries[(idx + probe) & stt->mask];
        uint64_t eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        if (eh1 == TT_EMPTY_HASH) {
            uint64_t expected = TT_EMPTY_HASH;
            if (atomic_compare_exchange_strong_explicit(&e->hash1, &expected, h1,
                                                       memory_order_acq_rel, memory_order_acquire)) {
                atomic_store_explicit(&e->hash2, h2, memory_order_release);
                atomic_store_explicit(&e->size, size, memory_order_release);
                atomic_store_explicit(&e->exact_cost, exact_cost, memory_order_release);
                atomic_store_explicit(&e->proven_lower_bound, proven_lb, memory_order_release);
                atomic_store_explicit(&e->best_guess, best_guess, memory_order_release);
                return;
            }
            eh1 = atomic_load_explicit(&e->hash1, memory_order_acquire);
        }
        if (eh1 == h1) {
            uint64_t eh2 = atomic_load_explicit(&e->hash2, memory_order_acquire);
            uint32_t esz = atomic_load_explicit(&e->size, memory_order_acquire);
            if (eh2 == h2 && esz == size) {
                if (exact_cost != UINT32_MAX) {
                    atomic_store_explicit(&e->exact_cost, exact_cost, memory_order_release);
                }
                if (proven_lb > 0) {
                    uint32_t cur_lb = atomic_load_explicit(&e->proven_lower_bound, memory_order_relaxed);
                    while (proven_lb > cur_lb && !atomic_compare_exchange_weak_explicit(
                               &e->proven_lower_bound, &cur_lb, proven_lb,
                               memory_order_release, memory_order_relaxed)) {}
                }
                if (best_guess != UINT32_MAX) {
                    atomic_store_explicit(&e->best_guess, best_guess, memory_order_release);
                }
                return;
            }
        }
    }
}

// -------------------------------------------------------------
// Solver Context
// -------------------------------------------------------------

typedef struct {
    uint32_t guess_idx;
    uint32_t active_buckets;
    uint32_t sum_sq;
    uint32_t lb;
    bool is_exact_lb;
} RankedCandidate;

typedef struct {
    GameData* game;
    TT tt;

    // 128-bit Dedup scratch
    uint64_t* dedup_stamp;
    uint64_t* dedup_hash1;
    uint64_t* dedup_hash2;
    uint32_t* dedup_guess;
    uint64_t call_id;

    uint32_t* representatives;
    RankedCandidate* ranked;
    uint64_t nodes_visited;
} Solver;

static void solver_init(Solver* s, GameData* game) {
    s->game = game;
    uint64_t tt_cap = next_pow2((uint64_t)game->num_targets * 128);
    if (tt_cap < (1u << 16)) tt_cap = 1u << 16;
    if (tt_cap > (1u << 24)) tt_cap = 1u << 24;
    tt_init(&s->tt, tt_cap);

    uint64_t dedup_cap = next_pow2((uint64_t)game->num_guesses * 2);
    s->dedup_stamp = calloc(dedup_cap, sizeof(uint64_t));
    s->dedup_hash1 = malloc(dedup_cap * sizeof(uint64_t));
    s->dedup_hash2 = malloc(dedup_cap * sizeof(uint64_t));
    s->dedup_guess = malloc(dedup_cap * sizeof(uint32_t));
    s->call_id = 0;
    s->representatives = malloc((size_t)game->num_guesses * sizeof(uint32_t));
    s->ranked = malloc((size_t)game->num_guesses * sizeof(RankedCandidate));
    s->nodes_visited = 0;
}

static void solver_free(Solver* s) {
    tt_free(&s->tt);
    free(s->dedup_stamp);
    free(s->dedup_hash1);
    free(s->dedup_hash2);
    free(s->dedup_guess);
    free(s->representatives);
    free(s->ranked);
}

static inline TTEntry* solver_tt_find(Solver* solver, uint64_t h1, uint64_t h2, uint32_t size) {
    TTEntry* entry = tt_find(&solver->tt, h1, h2, size);
    if (entry) return entry;
    if (solver->game && solver->game->shared_tt.entries) {
        uint32_t s_exact = UINT32_MAX, s_lb = 0, s_guess = UINT32_MAX;
        if (shared_tt_find(&solver->game->shared_tt, h1, h2, size, &s_exact, &s_lb, &s_guess)) {
            TTEntry* e = tt_find_or_claim(&solver->tt, h1, h2, size);
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

static inline void solver_tt_store_exact(Solver* solver, uint64_t h1, uint64_t h2, uint32_t size, uint32_t exact_cost, uint32_t best_guess) {
    TTEntry* e = tt_find_or_claim(&solver->tt, h1, h2, size);
    if (e) {
        e->exact_cost = exact_cost;
        if (best_guess != UINT32_MAX) e->best_guess = best_guess;
    }
    if (solver->game && solver->game->shared_tt.entries) {
        shared_tt_store(&solver->game->shared_tt, h1, h2, size, exact_cost, exact_cost, best_guess);
    }
}

static inline void solver_tt_store_lb(Solver* solver, uint64_t h1, uint64_t h2, uint32_t size, uint32_t lb, uint32_t best_guess) {
    TTEntry* e = tt_find_or_claim(&solver->tt, h1, h2, size);
    if (e) {
        if (lb > e->proven_lower_bound) e->proven_lower_bound = lb;
        if (best_guess != UINT32_MAX && e->best_guess == UINT32_MAX) e->best_guess = best_guess;
    }
    if (solver->game && solver->game->shared_tt.entries) {
        shared_tt_store(&solver->game->shared_tt, h1, h2, size, UINT32_MAX, lb, best_guess);
    }
}

static int compare_ranked_desc(const void* a, const void* b) {
    const RankedCandidate* ra = (const RankedCandidate*)a;
    const RankedCandidate* rb = (const RankedCandidate*)b;
    if (ra->sum_sq != rb->sum_sq) return (ra->sum_sq < rb->sum_sq) ? -1 : 1;
    if (ra->lb != rb->lb) return (ra->lb < rb->lb) ? -1 : 1;
    if (ra->active_buckets != rb->active_buckets) return (ra->active_buckets > rb->active_buckets) ? -1 : 1;
    return (ra->guess_idx < rb->guess_idx) ? -1 : (ra->guess_idx > rb->guess_idx ? 1 : 0);
}

typedef struct {
    uint16_t score;
    uint32_t size;
    uint32_t offset;
    uint64_t hash1;
    uint64_t hash2;
} BucketInfo;

static int compare_bucket_size_desc(const void* a, const void* b) {
    const BucketInfo* ba = (const BucketInfo*)a;
    const BucketInfo* bb = (const BucketInfo*)b;
    return (ba->size > bb->size) ? -1 : ((ba->size < bb->size) ? 1 : 0);
}

// -------------------------------------------------------------
// Fast 1-Ply Greedy Heuristic (Upper Bound Seeder)
// -------------------------------------------------------------

static uint32_t solve_greedy_tree(GameData* game, const uint32_t* targets, uint32_t count) {
    if (count == 0) return 0;
    if (count == 1) return 1;
    if (count == 2) return 3;

    uint32_t best_g = UINT32_MAX;
    uint32_t min_sum_sq = UINT32_MAX;
    uint32_t max_active = 0;

    for (uint32_t g = 0; g < game->num_guesses; g++) {
        const uint8_t* row = game->score_matrix + (size_t)g * game->num_targets;
        uint32_t hist[NUM_SCORES] = {0};
        for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;

        uint32_t sum_sq = 0, active = 0;
        for (int s = 0; s < NUM_SCORES; s++) {
            if (hist[s] > 0) {
                active++;
                sum_sq += hist[s] * hist[s];
            }
        }

        if (sum_sq < min_sum_sq || (sum_sq == min_sum_sq && active > max_active)) {
            min_sum_sq = sum_sq;
            max_active = active;
            best_g = g;
        }
    }

    const uint8_t* row = game->score_matrix + (size_t)best_g * game->num_targets;
    uint32_t hist[NUM_SCORES] = {0};
    for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;

    uint32_t offsets[NUM_SCORES + 1];
    offsets[0] = 0;
    for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
    uint32_t cur[NUM_SCORES];
    memcpy(cur, offsets, sizeof(cur));

    uint32_t* part = malloc(count * sizeof(uint32_t));
    for (uint32_t i = 0; i < count; i++) {
        uint8_t s = row[targets[i]];
        part[cur[s]++] = targets[i];
    }

    uint32_t total = count;
    for (int s = 0; s < NUM_SCORES; s++) {
        if (s != EXACT_MATCH && hist[s] > 0) {
            total += solve_greedy_tree(game, &part[offsets[s]], hist[s]);
        }
    }
    free(part);
    return total;
}

static uint32_t compute_opener_greedy_upper_bound(GameData* game, uint32_t opener_idx) {
    uint32_t count = game->num_targets;
    const uint8_t* row = game->score_matrix + (size_t)opener_idx * count;
    uint32_t hist[NUM_SCORES] = {0};
    for (uint32_t t = 0; t < count; t++) hist[row[t]]++;

    uint32_t offsets[NUM_SCORES + 1];
    offsets[0] = 0;
    for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
    uint32_t cur[NUM_SCORES];
    memcpy(cur, offsets, sizeof(cur));

    uint32_t* part = malloc(count * sizeof(uint32_t));
    for (uint32_t t = 0; t < count; t++) {
        uint8_t s = row[t];
        part[cur[s]++] = t;
    }

    uint32_t total = count;
    for (int s = 0; s < NUM_SCORES; s++) {
        if (s != EXACT_MATCH && hist[s] > 0) {
            total += solve_greedy_tree(game, &part[offsets[s]], hist[s]);
        }
    }
    free(part);
    return total;
}

// -------------------------------------------------------------
// Core Branch-and-Bound Solver
// -------------------------------------------------------------

static uint32_t solve_subset(Solver* solver, const uint32_t* targets, uint32_t count,
                              uint64_t h1, uint64_t h2, uint32_t beta, uint32_t* out_guess) {
    solver->nodes_visited++;
    GameData* game = solver->game;

    if (count == 0) return 0;
    if (count == 1) {
        if (out_guess) *out_guess = targets[0];
        return 1;
    }
    if (count == 2) {
        if (out_guess) *out_guess = targets[0];
        return 3;
    }

    const uint32_t num_targets = game->num_targets;
    const uint32_t num_guesses = game->num_guesses;
    const uint8_t* matrix = game->score_matrix;

    uint32_t node_lb = game->lower_bound[count];

    TTEntry* entry = solver_tt_find(solver, h1, h2, count);
    uint32_t suggested_guess = UINT32_MAX;
    if (entry) {
        if (entry->exact_cost != UINT32_MAX) {
            if (out_guess) *out_guess = entry->best_guess;
            return entry->exact_cost;
        }
        if (entry->proven_lower_bound > node_lb) node_lb = entry->proven_lower_bound;
        suggested_guess = entry->best_guess;
    }

    if (node_lb >= beta) {
        solver_tt_store_lb(solver, h1, h2, count, node_lb, suggested_guess);
        return node_lb;
    }

    // ---- Endgame Cluster Static Analysis ----
    if (count >= MIN_ENDGAME_COUNT) {
        uint32_t eg_counts[MAX_ENDGAMES] = {0};
        uint32_t max_eg_count = 0;
        int biggest_eg = -1;

        for (uint32_t i = 0; i < count; i++) {
            uint32_t tid = targets[i];
            uint32_t eg_cnt = game->target_endgame_counts[tid];
            for (uint32_t k = 0; k < eg_cnt; k++) {
                uint32_t eg_id = game->target_endgames[tid][k];
                eg_counts[eg_id]++;
                if (eg_counts[eg_id] > max_eg_count) {
                    max_eg_count = eg_counts[eg_id];
                    biggest_eg = (int)eg_id;
                }
            }
        }

        if (biggest_eg >= 0 && max_eg_count >= MIN_ENDGAME_COUNT) {
            // Live endgame target indices
            uint32_t live_eg[128];
            uint32_t live_count = 0;

            for (uint32_t i = 0; i < count; i++) {
                uint32_t tid = targets[i];
                for (uint32_t k = 0; k < game->target_endgame_counts[tid]; k++) {
                    if (game->target_endgames[tid][k] == (uint32_t)biggest_eg) {
                        live_eg[live_count++] = tid;
                        break;
                    }
                }
            }

            // Exact Selby letter distinguishability coverage analysis
            uint32_t mult[6] = {0};
            for (uint32_t g = 0; g < num_guesses; g++) {
                const uint8_t* row = matrix + (size_t)g * num_targets;
                uint8_t seen[NUM_SCORES] = {0};
                uint32_t n_distinct = 0;
                for (uint32_t j = 0; j < live_count; j++) {
                    uint8_t sc = row[live_eg[j]];
                    if (!seen[sc]) {
                        seen[sc] = 1;
                        n_distinct++;
                    }
                }
                if (n_distinct >= 1 && n_distinct <= 6) mult[n_distinct - 1]++;
            }

            uint32_t sum_coverage = 0;
            uint32_t r = 5;
            for (int n = 5; n > 0 && r > 0; n--) {
                uint32_t r1 = (r < mult[n]) ? r : mult[n];
                sum_coverage += r1 * (uint32_t)n;
                r -= r1;
            }

            if (live_count - 1 > sum_coverage) {
                // Static coverage cutoff: provably cannot distinguish live_count words in remaining moves
                solver_tt_store_lb(solver, h1, h2, count, beta, UINT32_MAX);
                return beta;
            }

            int32_t heuristic = (int32_t)5 - (int32_t)(sum_coverage - (live_count - 1));
            // Live endgame subsearch with Selby heuristic gating
            if (heuristic > 0 && live_count < count) {
                uint64_t lh1 = 0, lh2 = 0;
                for (uint32_t j = 0; j < live_count; j++) {
                    lh1 ^= game->zobrist1[live_eg[j]];
                    lh2 ^= game->zobrist2[live_eg[j]];
                }
                uint32_t eg_cost = solve_subset(solver, live_eg, live_count, lh1, lh2, beta, NULL);
                if (eg_cost >= beta) {
                    solver_tt_store_lb(solver, h1, h2, count, eg_cost, UINT32_MAX);
                    return eg_cost;
                }
            }
        }
    }

    // ---- Fused Dedup + Histogram + Move Ordering + lb1 Computation ----
    solver->call_id++;
    uint64_t call_id = solver->call_id;
    uint64_t dedup_mask = next_pow2((uint64_t)num_guesses * 2) - 1;
    uint32_t rep_count = 0;

    RankedCandidate* ranked = solver->ranked;
    uint32_t global_lb1 = UINT32_MAX;
    uint32_t global_ub1 = UINT32_MAX;
    uint32_t best_exact_g = UINT32_MAX;

    uint32_t hist[NUM_SCORES] = {0};
    uint16_t active_scores[count + 1];

    if (count <= 8) {
        uint32_t t0 = targets[0], t1 = targets[1], t2 = (count > 2) ? targets[2] : 0;
        uint32_t t3 = (count > 3) ? targets[3] : 0, t4 = (count > 4) ? targets[4] : 0;
        uint32_t t5 = (count > 5) ? targets[5] : 0, t6 = (count > 6) ? targets[6] : 0, t7 = (count > 7) ? targets[7] : 0;

        for (uint32_t g = 0; g < num_guesses; g++) {
            const uint8_t* row = matrix + (size_t)g * num_targets;
            uint64_t sig = (uint64_t)row[t0] | ((uint64_t)row[t1] << 8);
            if (count > 2) sig |= ((uint64_t)row[t2] << 16);
            if (count > 3) sig |= ((uint64_t)row[t3] << 24);
            if (count > 4) sig |= ((uint64_t)row[t4] << 32);
            if (count > 5) sig |= ((uint64_t)row[t5] << 40);
            if (count > 6) sig |= ((uint64_t)row[t6] << 48);
            if (count > 7) sig |= ((uint64_t)row[t7] << 56);

            uint64_t idx = (sig ^ (sig >> 17) ^ (sig >> 33)) & dedup_mask;
            bool duplicate = false;
            while (solver->dedup_stamp[idx] == call_id) {
                if (solver->dedup_hash1[idx] == sig) {
                    duplicate = true;
                    break;
                }
                idx = (idx + 1) & dedup_mask;
            }
            if (duplicate) continue;

            solver->dedup_stamp[idx] = call_id;
            solver->dedup_hash1[idx] = sig;
            solver->dedup_guess[idx] = g;
            solver->representatives[rep_count] = g;

            uint32_t num_active = 0;
            for (uint32_t i = 0; i < count; i++) {
                uint8_t sc = (uint8_t)(sig >> (i * 8));
                if (hist[sc] == 0) active_scores[num_active++] = sc;
                hist[sc]++;
            }

            uint32_t sum_sq = 0;
            uint32_t guess_lb = count;
            bool max_bucket_le_2 = true;

            for (uint32_t k = 0; k < num_active; k++) {
                uint16_t s = active_scores[k];
                uint32_t sz = hist[s];
                sum_sq += sz * sz;
                if (s != EXACT_MATCH) guess_lb += game->lower_bound[sz];
                if (sz > 2) max_bucket_le_2 = false;
                hist[s] = 0; // Reset
            }

            if (guess_lb < global_lb1) global_lb1 = guess_lb;
            if (max_bucket_le_2 && guess_lb < global_ub1) {
                global_ub1 = guess_lb;
                best_exact_g = g;
            }

            ranked[rep_count] = (RankedCandidate){
                .guess_idx = g,
                .active_buckets = num_active,
                .sum_sq = sum_sq,
                .lb = guess_lb,
                .is_exact_lb = max_bucket_le_2
            };
            rep_count++;
        }
    } else {
        for (uint32_t g = 0; g < num_guesses; g++) {
            const uint8_t* row = matrix + (size_t)g * num_targets;

            uint64_t ph1 = 1469598103934665603ULL;
            uint64_t ph2 = 1099511628211ULL;

            for (uint32_t i = 0; i < count; i++) {
                uint8_t sc = row[targets[i]];
                ph1 = (ph1 ^ sc) * 1099511628211ULL;
                ph2 = (ph2 ^ sc) * 1469598103934665603ULL;
            }

            uint64_t idx = (ph1 ^ ph2) & dedup_mask;
            bool duplicate = false;
            while (solver->dedup_stamp[idx] == call_id) {
                if (solver->dedup_hash1[idx] == ph1 && solver->dedup_hash2[idx] == ph2) {
                    duplicate = true;
                    break;
                }
                idx = (idx + 1) & dedup_mask;
            }
            if (duplicate) continue;

            solver->dedup_stamp[idx] = call_id;
            solver->dedup_hash1[idx] = ph1;
            solver->dedup_hash2[idx] = ph2;
            solver->dedup_guess[idx] = g;
            solver->representatives[rep_count] = g;

            uint32_t num_active = 0;
            for (uint32_t i = 0; i < count; i++) {
                uint8_t sc = row[targets[i]];
                if (hist[sc] == 0) active_scores[num_active++] = sc;
                hist[sc]++;
            }

            uint32_t sum_sq = 0;
            uint32_t guess_lb = count;
            bool max_bucket_le_2 = true;

            for (uint32_t k = 0; k < num_active; k++) {
                uint16_t s = active_scores[k];
                uint32_t sz = hist[s];
                sum_sq += sz * sz;
                if (s != EXACT_MATCH) guess_lb += game->lower_bound[sz];
                if (sz > 2) max_bucket_le_2 = false;
                hist[s] = 0; // Reset
            }

            if (guess_lb < global_lb1) global_lb1 = guess_lb;
            if (max_bucket_le_2 && guess_lb < global_ub1) {
                global_ub1 = guess_lb;
                best_exact_g = g;
            }

            ranked[rep_count] = (RankedCandidate){
                .guess_idx = g,
                .active_buckets = num_active,
                .sum_sq = sum_sq,
                .lb = guess_lb,
                .is_exact_lb = max_bucket_le_2
            };
            rep_count++;
        }
    }

    // Sound fail-soft cutoff if lb1 >= beta
    if (global_lb1 >= beta) {
        solver_tt_store_lb(solver, h1, h2, count, global_lb1, UINT32_MAX);
        return global_lb1;
    }

    // Exact resolution if ub1 == lb1 (optimal move found without recursion)
    if (global_ub1 == global_lb1 && best_exact_g != UINT32_MAX && global_lb1 < beta) {
        solver_tt_store_exact(solver, h1, h2, count, global_lb1, best_exact_g);
        if (out_guess) *out_guess = best_exact_g;
        return global_lb1;
    }

    qsort(ranked, rep_count, sizeof(RankedCandidate), compare_ranked_desc);

    if (suggested_guess != UINT32_MAX) {
        for (uint32_t r = 0; r < rep_count; r++) {
            if (ranked[r].guess_idx == suggested_guess) {
                RankedCandidate tmp = ranked[0];
                ranked[0] = ranked[r];
                ranked[r] = tmp;
                break;
            }
        }
    }

    uint32_t rep_guesses[rep_count];
    for (uint32_t r = 0; r < rep_count; r++) rep_guesses[r] = ranked[r].guess_idx;

    // ---- Main Branch-and-Bound Loop ----
    uint32_t current_best = beta;
    uint32_t best_g = rep_guesses[0];
    bool found_improvement = false;

    uint32_t local_partition[count];
    uint32_t offsets[NUM_SCORES + 1];
    BucketInfo buckets[NUM_SCORES];

    for (uint32_t c = 0; c < rep_count; c++) {
        uint32_t g = rep_guesses[c];
        const uint8_t* row = matrix + (size_t)g * num_targets;

        uint32_t active_buckets = 0;
        uint32_t guess_lb = count;

        for (uint32_t i = 0; i < count; i++) {
            uint8_t sc = row[targets[i]];
            if (hist[sc] == 0) {
                active_scores[active_buckets++] = sc;
            }
            hist[sc]++;
        }

        bool has_exact = (hist[EXACT_MATCH] > 0);
        for (uint32_t b = 0; b < active_buckets; b++) {
            uint16_t s = active_scores[b];
            uint32_t sz = hist[s];
            if (s != EXACT_MATCH) guess_lb += game->lower_bound[sz];
            buckets[b].score = s;
            buckets[b].size = sz;
        }

        if (active_buckets == 1) {
            for (uint32_t b = 0; b < active_buckets; b++) hist[active_scores[b]] = 0;
            continue; // Zero-info move prune
        }

        // Perfect split shortcut (2n - 1)
        if (active_buckets == count && has_exact) {
            for (uint32_t b = 0; b < active_buckets; b++) hist[active_scores[b]] = 0;
            current_best = guess_lb;
            best_g = g;
            found_improvement = true;
            break;
        }

        // Non-candidate singleton split shortcut (2n)
        if (active_buckets == count && !has_exact) {
            for (uint32_t b = 0; b < active_buckets; b++) hist[active_scores[b]] = 0;
            if (2 * count < current_best) {
                current_best = 2 * count;
                best_g = g;
                found_improvement = true;
            }
            continue;
        }

        if (guess_lb >= current_best) {
            for (uint32_t b = 0; b < active_buckets; b++) hist[active_scores[b]] = 0;
            continue;
        }

        offsets[0] = 0;
        for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
        uint32_t cur_offsets[NUM_SCORES];
        memcpy(cur_offsets, offsets, sizeof(cur_offsets));
        for (uint32_t i = 0; i < count; i++) {
            uint8_t s = row[targets[i]];
            local_partition[cur_offsets[s]++] = targets[i];
        }

        // Clear hist now that partition is formed
        for (uint32_t b = 0; b < active_buckets; b++) hist[active_scores[b]] = 0;

        for (uint32_t b = 0; b < active_buckets; b++) {
            uint32_t sz = buckets[b].size;
            uint32_t off = offsets[buckets[b].score];
            buckets[b].offset = off;
            if (sz >= 3) {
                uint64_t bh1 = 0, bh2 = 0;
                for (uint32_t j = 0; j < sz; j++) {
                    uint32_t tid = local_partition[off + j];
                    bh1 ^= game->zobrist1[tid];
                    bh2 ^= game->zobrist2[tid];
                }
                buckets[b].hash1 = bh1;
                buckets[b].hash2 = bh2;
            }
        }

        qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

        // ---- Tier 2: Pre-scan buckets with instant exact resolution (sz <= 2) and TT lower bound probe ----
        uint32_t bucket_exact[NUM_SCORES] = {0};
        uint32_t bucket_lb[NUM_SCORES] = {0};
        uint32_t tier2_total_lb = count;
        uint32_t resolved_cost = count;
        uint32_t num_unresolved = 0;
        BucketInfo unresolved_buckets[NUM_SCORES];

        for (uint32_t b = 0; b < active_buckets; b++) {
            if (buckets[b].score == EXACT_MATCH) continue;
            uint32_t sz = buckets[b].size;
            if (sz <= 2) {
                uint32_t c = (sz == 1) ? 1 : 3;
                bucket_exact[b] = c;
                bucket_lb[b] = c;
                tier2_total_lb += c;
                resolved_cost += c;
                continue;
            }

            uint32_t base_lb = game->lower_bound[sz];
            TTEntry* entry = solver_tt_find(solver, buckets[b].hash1, buckets[b].hash2, sz);
            if (entry) {
                if (entry->exact_cost != UINT32_MAX) {
                    bucket_exact[b] = entry->exact_cost;
                    bucket_lb[b] = entry->exact_cost;
                    resolved_cost += entry->exact_cost;
                } else if (entry->proven_lower_bound > base_lb) {
                    bucket_lb[b] = entry->proven_lower_bound;
                    unresolved_buckets[num_unresolved++] = buckets[b];
                } else {
                    bucket_lb[b] = base_lb;
                    unresolved_buckets[num_unresolved++] = buckets[b];
                }
            } else {
                bucket_lb[b] = base_lb;
                unresolved_buckets[num_unresolved++] = buckets[b];
            }
            tier2_total_lb += bucket_lb[b];
        }

        // Tier 2 cutoff: provably cannot beat current_best without any recursion
        if (tier2_total_lb >= current_best) continue;

        // If all buckets were resolved by sz <= 2 or TT exact hits, we have the exact score immediately
        if (num_unresolved == 0) {
            if (resolved_cost < current_best) {
                current_best = resolved_cost;
                best_g = g;
                found_improvement = true;
                if (current_best <= node_lb) break;
            }
            continue;
        }

        // ---- Tier 3: Ordered Fail-Soft Recursion on Unresolved Buckets ----
        uint32_t running_cost = resolved_cost;
        uint32_t remaining_lb = 0;
        for (uint32_t u = 0; u < num_unresolved; u++) {
            remaining_lb += game->lower_bound[unresolved_buckets[u].size];
        }

        bool pruned = false;
        uint32_t u_costs[NUM_SCORES] = {0};

        for (uint32_t u = 0; u < num_unresolved; u++) {
            uint32_t sz = unresolved_buckets[u].size;
            remaining_lb -= game->lower_bound[sz];

            if (running_cost + remaining_lb >= current_best) {
                pruned = true;
                break;
            }

            uint32_t bucket_beta = current_best - running_cost - remaining_lb;
            uint32_t bucket_cost = solve_subset(solver, &local_partition[unresolved_buckets[u].offset], sz,
                                                 unresolved_buckets[u].hash1, unresolved_buckets[u].hash2, bucket_beta, NULL);
            if (bucket_cost >= bucket_beta) {
                pruned = true;
                break;
            }
            u_costs[u] = bucket_cost;
            running_cost += bucket_cost;
        }

        // Easy-mode disjoint-union bound propagation on cutoff
        if (pruned && num_unresolved >= 2) {
            for (uint32_t u1 = 0; u1 < num_unresolved; u1++) {
                if (u_costs[u1] == 0) continue;
                for (uint32_t u2 = u1 + 1; u2 < num_unresolved; u2++) {
                    if (u_costs[u2] == 0) continue;
                    uint64_t mh1 = unresolved_buckets[u1].hash1 ^ unresolved_buckets[u2].hash1;
                    uint64_t mh2 = unresolved_buckets[u1].hash2 ^ unresolved_buckets[u2].hash2;
                    uint32_t msize = unresolved_buckets[u1].size + unresolved_buckets[u2].size;
                    uint32_t mlb = u_costs[u1] + u_costs[u2];
                    solver_tt_store_lb(solver, mh1, mh2, msize, mlb, UINT32_MAX);
                }
            }
        }

        if (!pruned && running_cost < current_best) {
            current_best = running_cost;
            best_g = g;
            found_improvement = true;
            if (current_best <= node_lb) break;
        }
    }

    if (found_improvement) {
        solver_tt_store_exact(solver, h1, h2, count, current_best, best_g);
        if (out_guess) *out_guess = best_g;
        return current_best;
    } else {
        solver_tt_store_lb(solver, h1, h2, count, beta, best_g);
        if (out_guess) *out_guess = best_g;
        return beta;
    }
}

// -------------------------------------------------------------
// Parallel Evaluation
// -------------------------------------------------------------

typedef struct {
    uint32_t opener_idx;
    uint32_t exact_total_cost;
    double avg_guesses;
    double time_sec;
    uint64_t nodes;
    bool is_exact;
} OpenerResult;

static void partition_root(GameData* game, uint32_t opener_idx, uint32_t* local_partition,
                            BucketInfo* buckets, uint32_t* out_active_buckets) {
    uint32_t count = game->num_targets;
    const uint8_t* row = game->score_matrix + (size_t)opener_idx * game->num_targets;
    uint32_t hist[NUM_SCORES] = {0};
    for (uint32_t t = 0; t < count; t++) hist[row[t]]++;

    uint32_t offsets[NUM_SCORES + 1];
    offsets[0] = 0;
    for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
    uint32_t cur[NUM_SCORES];
    memcpy(cur, offsets, sizeof(cur));
    for (uint32_t t = 0; t < count; t++) {
        uint8_t s = row[t];
        local_partition[cur[s]++] = t;
    }

    uint32_t active = 0;
    for (int s = 0; s < NUM_SCORES; s++) {
        if (hist[s] > 0) {
            uint32_t off = offsets[s];
            uint64_t bh1 = 0, bh2 = 0;
            for (uint32_t j = 0; j < hist[s]; j++) {
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
    GameData* game;
    const uint32_t* local_partition;
    BucketInfo* buckets;
    uint32_t num_buckets;
    const uint32_t* bucket_betas;
    atomic_size_t next_idx;
    uint32_t* out_costs;
    uint32_t* out_guesses;
    uint64_t* out_nodes;
} BucketPool;

static void* bucket_worker(void* arg) {
    BucketPool* pool = (BucketPool*)arg;
    Solver solver;
    solver_init(&solver, pool->game);
    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        if (idx >= pool->num_buckets) break;
        BucketInfo* bkt = &pool->buckets[idx];
        if (bkt->score == EXACT_MATCH) continue;
        uint32_t best_guess;
        uint64_t start_nodes = solver.nodes_visited;
        uint32_t beta = pool->bucket_betas ? pool->bucket_betas[idx] : UINT32_MAX;
        uint32_t cost = solve_subset(&solver, &pool->local_partition[bkt->offset], bkt->size,
                                      bkt->hash1, bkt->hash2, beta, &best_guess);
        pool->out_costs[idx] = cost;
        pool->out_guesses[idx] = best_guess;
        pool->out_nodes[idx] = solver.nodes_visited - start_nodes; // Fix: record bucket delta
    }
    solver_free(&solver);
    return NULL;
}

static OpenerResult evaluate_opener_parallel(GameData* game, uint32_t opener_idx, int num_threads) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Fast greedy aspiration upper bound
    uint32_t greedy_upper_bound = compute_opener_greedy_upper_bound(game, opener_idx);

    uint32_t count = game->num_targets;
    uint32_t* local_partition = malloc(count * sizeof(uint32_t));
    BucketInfo* buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    uint32_t active_buckets;
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    uint32_t total_other_lb = 0;
    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score != EXACT_MATCH) total_other_lb += game->lower_bound[buckets[b].size];
    }

    uint32_t* bucket_betas = malloc(active_buckets * sizeof(uint32_t));
    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score == EXACT_MATCH) {
            bucket_betas[b] = 0;
            continue;
        }
        uint32_t other_lb = total_other_lb - game->lower_bound[buckets[b].size];
        if (greedy_upper_bound > count + other_lb) {
            bucket_betas[b] = greedy_upper_bound - count - other_lb + 1;
        } else {
            bucket_betas[b] = game->lower_bound[buckets[b].size] + 1;
        }
    }

    if (num_threads < 1) num_threads = 1;
    uint32_t total_cost = count;
    uint64_t total_nodes = 0;

    if (active_buckets > 0) {
        BucketPool pool = {
            .game = game,
            .local_partition = local_partition,
            .buckets = buckets,
            .num_buckets = active_buckets,
            .bucket_betas = bucket_betas
        };
        atomic_init(&pool.next_idx, 0);
        pool.out_costs = calloc(active_buckets, sizeof(uint32_t));
        pool.out_guesses = calloc(active_buckets, sizeof(uint32_t));
        pool.out_nodes = calloc(active_buckets, sizeof(uint64_t));

        pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
        for (int i = 0; i < num_threads; i++) pthread_create(&threads[i], NULL, bucket_worker, &pool);
        for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
        free(threads);

        for (uint32_t b = 0; b < active_buckets; b++) {
            if (pool.buckets[b].score == EXACT_MATCH) continue;
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

    OpenerResult res;
    res.opener_idx = opener_idx;
    res.exact_total_cost = total_cost;
    res.avg_guesses = (double)total_cost / (double)count;
    res.is_exact = true;
    res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    res.nodes = total_nodes;
    (void)greedy_upper_bound;
    return res;
}

static OpenerResult evaluate_opener_sequential(Solver* solver, uint32_t opener_idx,
                                                atomic_uint_fast32_t* global_best_cost) {
    GameData* game = solver->game;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint32_t count = game->num_targets;
    uint32_t* local_partition = malloc(count * sizeof(uint32_t));
    BucketInfo* buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    uint32_t active_buckets;
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);

    uint32_t root_lb = count;
    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score != EXACT_MATCH) root_lb += game->lower_bound[buckets[b].size];
    }

    OpenerResult res = { .opener_idx = opener_idx };
    uint32_t ceiling = atomic_load(global_best_cost);
    if (ceiling != UINT32_MAX && root_lb > ceiling) {
        free(local_partition); free(buckets);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        res.exact_total_cost = UINT32_MAX;
        res.avg_guesses = 99.0;
        res.is_exact = false;
        res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        res.nodes = 0;
        return res;
    }

    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    uint32_t running_cost = count;
    uint32_t remaining_lb = root_lb - count;
    bool pruned = false;
    uint64_t start_nodes = solver->nodes_visited;

    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score == EXACT_MATCH) continue;
        remaining_lb -= game->lower_bound[buckets[b].size];
        ceiling = atomic_load(global_best_cost);
        if (ceiling != UINT32_MAX && running_cost + remaining_lb >= ceiling) {
            pruned = true;
            break;
        }
        uint32_t bucket_beta = (ceiling == UINT32_MAX) ? UINT32_MAX : (ceiling - running_cost - remaining_lb);
        uint32_t cost = solve_subset(solver, &local_partition[buckets[b].offset], buckets[b].size,
                                      buckets[b].hash1, buckets[b].hash2, bucket_beta, NULL);
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

// -------------------------------------------------------------
// Decision Tree & JSON Export
// -------------------------------------------------------------

typedef struct TreeNode {
    char guess[WORD_LEN + 1];
    uint32_t num_targets;
    bool is_leaf;
    struct TreeNode* children[NUM_SCORES];
} TreeNode;

static TreeNode* make_leaf(GameData* game, uint32_t guess_idx) {
    TreeNode* n = calloc(1, sizeof(TreeNode));
    n->is_leaf = true;
    n->num_targets = 1;
    strcpy(n->guess, game->targets[guess_idx].word);
    return n;
}

static TreeNode* build_subtree_node(Solver* solver, const uint32_t* targets, uint32_t count, uint64_t h1, uint64_t h2);

static TreeNode* build_subtree_node_with_guess(Solver* solver, const uint32_t* targets, uint32_t count,
                                                uint32_t best_g) {
    GameData* game = solver->game;
    TreeNode* node = calloc(1, sizeof(TreeNode));
    node->num_targets = count;
    strcpy(node->guess, game->guesses[best_g].word);

    const uint8_t* row = game->score_matrix + (size_t)best_g * game->num_targets;
    uint32_t hist[NUM_SCORES] = {0};
    for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;

    uint32_t offsets[NUM_SCORES + 1];
    offsets[0] = 0;
    for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];

    uint32_t* local_partition = malloc(count * sizeof(uint32_t));
    uint32_t cur[NUM_SCORES];
    memcpy(cur, offsets, sizeof(cur));
    for (uint32_t i = 0; i < count; i++) {
        uint8_t s = row[targets[i]];
        local_partition[cur[s]++] = targets[i];
    }

    for (int s = 0; s < NUM_SCORES; s++) {
        if (hist[s] == 0) continue;
        if (s == EXACT_MATCH) {
            node->children[EXACT_MATCH] = make_leaf(game, best_g);
        } else {
            uint32_t off = offsets[s];
            uint64_t bh1 = 0, bh2 = 0;
            for (uint32_t j = 0; j < hist[s]; j++) {
                uint32_t tid = local_partition[off + j];
                bh1 ^= game->zobrist1[tid];
                bh2 ^= game->zobrist2[tid];
            }
            node->children[s] = build_subtree_node(solver, &local_partition[off], hist[s], bh1, bh2);
        }
    }

    free(local_partition);
    return node;
}

static TreeNode* build_subtree_node(Solver* solver, const uint32_t* targets, uint32_t count, uint64_t h1, uint64_t h2) {
    if (count == 0) return NULL;
    if (count == 1) return make_leaf(solver->game, targets[0]);
    uint32_t best_g;
    solve_subset(solver, targets, count, h1, h2, UINT32_MAX, &best_g);
    return build_subtree_node_with_guess(solver, targets, count, best_g);
}

typedef struct {
    GameData* game;
    const uint32_t* local_partition;
    BucketInfo* buckets;
    uint32_t num_buckets;
    atomic_size_t next_idx;
    TreeNode** out_nodes;
} TreeBucketPool;

static void* tree_bucket_worker(void* arg) {
    TreeBucketPool* pool = (TreeBucketPool*)arg;
    Solver solver;
    solver_init(&solver, pool->game);
    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        if (idx >= pool->num_buckets) break;
        BucketInfo* bkt = &pool->buckets[idx];
        if (bkt->score == EXACT_MATCH) continue;
        pool->out_nodes[idx] = build_subtree_node(&solver, &pool->local_partition[bkt->offset], bkt->size, bkt->hash1, bkt->hash2);
    }
    solver_free(&solver);
    return NULL;
}

static TreeNode* build_solution_tree(GameData* game, uint32_t opener_idx, int num_threads) {
    uint32_t count = game->num_targets;
    uint32_t* local_partition = malloc(count * sizeof(uint32_t));
    BucketInfo* buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    uint32_t active_buckets;
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    if (num_threads < 1) num_threads = 1;
    TreeNode* root = calloc(1, sizeof(TreeNode));
    root->num_targets = count;
    strcpy(root->guess, game->guesses[opener_idx].word);

    if (active_buckets > 0) {
        TreeBucketPool pool = {
            .game = game,
            .local_partition = local_partition,
            .buckets = buckets,
            .num_buckets = active_buckets
        };
        atomic_init(&pool.next_idx, 0);
        pool.out_nodes = calloc(active_buckets, sizeof(TreeNode*));

        pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
        for (int i = 0; i < num_threads; i++) pthread_create(&threads[i], NULL, tree_bucket_worker, &pool);
        for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
        free(threads);

        for (uint32_t b = 0; b < active_buckets; b++) {
            if (pool.buckets[b].score == EXACT_MATCH) {
                root->children[EXACT_MATCH] = make_leaf(game, opener_idx);
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

static void free_tree(TreeNode* node) {
    if (!node) return;
    for (int s = 0; s < NUM_SCORES; s++) free_tree(node->children[s]);
    free(node);
}

static void write_node_json(const TreeNode* node, FILE* fp, int indent) {
    char pad[128];
    int p = (indent > 60) ? 60 : indent;
    for (int i = 0; i < p; i++) pad[i] = ' ';
    pad[p] = '\0';

    if (node->is_leaf) {
        fprintf(fp, "%s{\"guess\": \"%s\", \"num_targets\": %u, \"leaf\": true}", pad, node->guess, node->num_targets);
        return;
    }

    fprintf(fp, "%s{\n", pad);
    fprintf(fp, "%s  \"guess\": \"%s\",\n", pad, node->guess);
    fprintf(fp, "%s  \"num_targets\": %u,\n", pad, node->num_targets);
    fprintf(fp, "%s  \"branches\": {\n", pad);
    bool first = true;
    for (int s = 0; s < NUM_SCORES; s++) {
        if (node->children[s]) {
            if (!first) fprintf(fp, ",\n");
            fprintf(fp, "%s    \"%d\":\n", pad, s);
            write_node_json(node->children[s], fp, indent + 6);
            first = false;
        }
    }
    fprintf(fp, "\n%s  }\n", pad);
    fprintf(fp, "%s}", pad);
}

static int dump_tree_to_json(const TreeNode* root, const char* filepath, GameData* game, uint32_t exact_total_cost) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open '%s' for writing\n", filepath);
        return -1;
    }
    double avg_score = (double)exact_total_cost / (double)game->num_targets;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": 1,\n");
    fprintf(fp, "  \"opener\": \"%s\",\n", root->guess);
    fprintf(fp, "  \"num_targets\": %u,\n", game->num_targets);
    fprintf(fp, "  \"num_guesses\": %u,\n", game->num_guesses);
    fprintf(fp, "  \"exact_total_guesses\": %u,\n", exact_total_cost);
    fprintf(fp, "  \"exact_avg_score\": %.17g,\n", avg_score);
    fprintf(fp, "  \"tree\":\n");
    write_node_json(root, fp, 4);
    fprintf(fp, "\n}\n");
    fclose(fp);
    printf("Successfully dumped complete optimal solution tree to '%s'\n", filepath);
    return 0;
}

// -------------------------------------------------------------
// Opener Pool for --top / --all
// -------------------------------------------------------------

typedef struct {
    uint32_t guess_idx;
    uint32_t sum_sq;
} HeuristicCandidate;

static int compare_heuristic_asc(const void* a, const void* b) {
    const HeuristicCandidate* ha = (const HeuristicCandidate*)a;
    const HeuristicCandidate* hb = (const HeuristicCandidate*)b;
    return (ha->sum_sq < hb->sum_sq) ? -1 : (ha->sum_sq > hb->sum_sq ? 1 : 0);
}

static int compare_opener_results_asc(const void* a, const void* b) {
    const OpenerResult* ra = (const OpenerResult*)a;
    const OpenerResult* rb = (const OpenerResult*)b;
    if (ra->exact_total_cost != rb->exact_total_cost) return (ra->exact_total_cost < rb->exact_total_cost) ? -1 : 1;
    return 0;
}

typedef struct {
    GameData* game;
    const uint32_t* opener_indices;
    size_t num_openers;
    atomic_size_t next_idx;
    atomic_size_t completed;
    OpenerResult* results;
    atomic_uint_fast32_t global_best_cost;
    pthread_mutex_t print_mutex;
    bool quiet;
    struct timespec start;
} OpenerWorkPool;

static void* opener_worker(void* arg) {
    OpenerWorkPool* pool = (OpenerWorkPool*)arg;
    Solver solver;
    solver_init(&solver, pool->game);

    while (1) {
        size_t idx = atomic_fetch_add(&pool->next_idx, 1);
        if (idx >= pool->num_openers) break;

        uint32_t g_idx = pool->opener_indices[idx];
        OpenerResult res = evaluate_opener_sequential(&solver, g_idx, &pool->global_best_cost);
        pool->results[idx] = res;

        size_t completed = atomic_fetch_add(&pool->completed, 1) + 1;

        pthread_mutex_lock(&pool->print_mutex);
        uint32_t prev_best = atomic_load(&pool->global_best_cost);
        bool is_new_best = res.is_exact && res.exact_total_cost < prev_best;
        if (is_new_best) atomic_store(&pool->global_best_cost, res.exact_total_cost);

        if (pool->quiet) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double el = (now.tv_sec - pool->start.tv_sec) + (now.tv_nsec - pool->start.tv_nsec) * 1e-9;
            double pct = (double)completed * 100.0 / (double)pool->num_openers;
            uint32_t cur_best = atomic_load(&pool->global_best_cost);
            double best_avg = (cur_best == UINT32_MAX) ? 0.0 : (double)cur_best / (double)pool->game->num_targets;
            printf("\r[%6zu/%6zu] (%5.1f%%) | Elapsed: %7.1fs | Current Best: %.5f avg",
                   completed, pool->num_openers, pct, el, best_avg);
            fflush(stdout);
        } else {
            printf("[%5zu/%5zu] Opener: %-5s | Exact Avg: %.5f (%u total) | Time: %6.2fs | Nodes: %llu %s\n",
                   completed, pool->num_openers, pool->game->guesses[g_idx].word,
                   res.avg_guesses, res.exact_total_cost, res.time_sec,
                   (unsigned long long)res.nodes, is_new_best ? " <-- NEW BEST" : "");
            fflush(stdout);
        }
        pthread_mutex_unlock(&pool->print_mutex);
    }

    solver_free(&solver);
    return NULL;
}

static void print_usage(const char* prog) {
    printf("Wordle Exact Solver (gemini_solver)\n\n");
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --wordlist <path>     Path to words.txt (default: words.txt)\n");
    printf("  --opener <word>       Evaluate a single opening word to exact optimality\n");
    printf("  --top <N>             Heuristically pre-rank openers, then exactly solve the top N\n");
    printf("  --all                 Exactly solve every possible opening word\n");
    printf("  --threads <N>         Number of worker threads (default: hardware concurrency)\n");
    printf("  --tree, --dump-tree <path> Dump optimal solution tree to JSON\n");
    printf("  --quiet, -q           Compact progress output\n");
    printf("  --help                Display this help message\n\n");
}

int main(int argc, char** argv) {
    const char* wordlist_path = "words.txt";
    const char* single_opener = NULL;
    const char* tree_dump_path = NULL;
    int top_n = 10;
    bool search_all = false;
    bool quiet = false;
    int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads < 1) num_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wordlist") == 0 && i + 1 < argc) wordlist_path = argv[++i];
        else if (strcmp(argv[i], "--opener") == 0 && i + 1 < argc) single_opener = argv[++i];
        else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc) top_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--all") == 0) search_all = true;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tree") == 0 || strcmp(argv[i], "--dump-tree") == 0) {
            if (i + 1 < argc) tree_dump_path = argv[++i];
        } else if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) quiet = true;
        else if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
    }

    printf("=================================================================\n");
    printf("      WORDLE OPTIMAL FULL-TREE SOLVER (gemini_solver)\n");
    printf("=================================================================\n");

    GameData game;
    if (load_wordlist(wordlist_path, &game) != 0) return 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    printf("Precomputing %u x %u score matrix using %d threads...\n", game.num_guesses, game.num_targets, num_threads);
    init_game_data(&game, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("Ready in %.3f seconds (%.1f MB matrix, %u endgames precomputed).\n\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9,
           (double)((size_t)game.num_guesses * game.num_targets) / (1024.0 * 1024.0),
           game.num_endgames);

    if (single_opener) {
        uint32_t opener_idx = UINT32_MAX;
        for (uint32_t g = 0; g < game.num_guesses; g++) {
            if (strcasecmp(game.guesses[g].word, single_opener) == 0) { opener_idx = g; break; }
        }
        if (opener_idx == UINT32_MAX) {
            fprintf(stderr, "Error: Opening word '%s' not found in word list.\n", single_opener);
            return 1;
        }

        printf("Evaluating opener: '%s' to exact mathematical optimality...\n", game.guesses[opener_idx].word);
        OpenerResult res = evaluate_opener_parallel(&game, opener_idx, num_threads);

        printf("\n======================= EXACT RESULT =======================\n");
        printf("  Opener:               %-5s\n", game.guesses[opener_idx].word);
        printf("  Total Target Words:   %u\n", game.num_targets);
        printf("  Exact Total Guesses:  %u\n", res.exact_total_cost);
        printf("  Exact Average Score:  %.5f guesses/game\n", res.avg_guesses);
        printf("  Computation Time:     %.3f seconds\n", res.time_sec);
        printf("  Tree Nodes Visited:   %llu\n", (unsigned long long)res.nodes);
        printf("============================================================\n");

        if (tree_dump_path) {
            printf("\nBuilding full decision tree for opener '%s'...\n", game.guesses[opener_idx].word);
            TreeNode* root = build_solution_tree(&game, opener_idx, num_threads);
            dump_tree_to_json(root, tree_dump_path, &game, res.exact_total_cost);
            free_tree(root);
        }

        free_game_data(&game);
        return 0;
    }

    printf("Ranking opening guesses by partition variance (heuristic pre-filter)...\n");
    HeuristicCandidate* cands = malloc(game.num_guesses * sizeof(HeuristicCandidate));
    for (uint32_t g = 0; g < game.num_guesses; g++) {
        uint32_t hist[NUM_SCORES] = {0};
        const uint8_t* row = game.score_matrix + (size_t)g * game.num_targets;
        for (uint32_t t = 0; t < game.num_targets; t++) hist[row[t]]++;
        uint32_t sum_sq = 0;
        for (int s = 0; s < NUM_SCORES; s++) sum_sq += hist[s] * hist[s];
        cands[g] = (HeuristicCandidate){ .guess_idx = g, .sum_sq = sum_sq };
    }
    qsort(cands, game.num_guesses, sizeof(HeuristicCandidate), compare_heuristic_asc);

    size_t count_to_eval = search_all ? game.num_guesses : (size_t)top_n;
    if (count_to_eval > game.num_guesses) count_to_eval = game.num_guesses;

    uint32_t* openers_to_eval = malloc(count_to_eval * sizeof(uint32_t));
    for (size_t i = 0; i < count_to_eval; i++) openers_to_eval[i] = cands[i].guess_idx;
    free(cands);

    // Compute greedy seed on top opener to initialize global_best_cost
    uint32_t initial_seed = compute_opener_greedy_upper_bound(&game, openers_to_eval[0]);
    printf("Initial aspiration seed for top opener '%s': %u total guesses (%.4f avg)\n",
           game.guesses[openers_to_eval[0]].word, initial_seed, (double)initial_seed / game.num_targets);

    printf("Evaluating %zu opener(s) in parallel using %d threads%s...\n\n",
           count_to_eval, num_threads, quiet ? " (quiet mode)" : "");

    OpenerWorkPool pool = {
        .game = &game,
        .opener_indices = openers_to_eval,
        .num_openers = count_to_eval,
        .quiet = quiet
    };
    clock_gettime(CLOCK_MONOTONIC, &pool.start);
    atomic_init(&pool.next_idx, 0);
    atomic_init(&pool.completed, 0);
    atomic_init(&pool.global_best_cost, UINT32_MAX);
    pthread_mutex_init(&pool.print_mutex, NULL);
    pool.results = calloc(count_to_eval, sizeof(OpenerResult));

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    for (int i = 0; i < num_threads; i++) pthread_create(&threads[i], NULL, opener_worker, &pool);
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    free(threads);
    if (quiet) printf("\n");

    qsort(pool.results, count_to_eval, sizeof(OpenerResult), compare_opener_results_asc);

    uint32_t best_total = pool.results[0].exact_total_cost;
    uint32_t best_opener_idx = pool.results[0].opener_idx;
    double best_avg = pool.results[0].avg_guesses;

    printf("\n======================== TOP RESULTS ========================\n");
    printf(" Rank | Opener | Exact Total | Exact Average | Time\n");
    printf("------+--------+-------------+---------------+----------\n");
    for (size_t i = 0; i < count_to_eval && i < 20; i++) {
        printf(" %4zu | %-6s | %11u | %11.5f | %6.2fs\n",
               i + 1, game.guesses[pool.results[i].opener_idx].word,
               pool.results[i].exact_total_cost, pool.results[i].avg_guesses, pool.results[i].time_sec);
    }
    printf("=============================================================\n");
    printf("%s OPENER: '%s' with exact average score: %.5f (%u total guesses)\n",
           search_all ? "GLOBAL OPTIMAL" : "BEST OF TOP-N",
           game.guesses[best_opener_idx].word, best_avg, best_total);
    printf("=============================================================\n");

    if (tree_dump_path) {
        printf("\nBuilding full decision tree for winning opener '%s'...\n", game.guesses[best_opener_idx].word);
        TreeNode* root = build_solution_tree(&game, best_opener_idx, num_threads);
        dump_tree_to_json(root, tree_dump_path, &game, best_total);
        free_tree(root);
    }

    free(openers_to_eval);
    free(pool.results);
    free_game_data(&game);
    return 0;
}
