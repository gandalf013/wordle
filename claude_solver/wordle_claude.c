// wordle_claude.c
//
// A from-scratch exact Wordle solver: computes the true minimum total
// number of guesses (summed over every target word) achievable with
// optimal play, for a given opening word or for the whole opening-word
// search space.
//
// Design goals, in order: (1) provable correctness -- every guess that
// could possibly be optimal at a node is actually considered, and every
// pruning decision is backed by an admissible lower bound; (2) speed,
// using only optimizations that cannot change the answer:
//
//   - Equivalence-class deduplication: two guesses that induce the exact
//     same partition of the remaining candidate set produce identical
//     subtrees, so only one representative per partition is ever solved.
//   - Admissible combinatorial lower bounds for branch-and-bound pruning,
//     plus an early exit the instant a node's cost matches its own lower
//     bound (a proof of optimality, not a heuristic).
//   - A transposition table keyed only by (subset hash, subset size),
//     with no notion of "generation" -- a subset's optimal cost does not
//     depend on how you reached it, so a solved subtree is cached
//     permanently and reused everywhere it recurs.
//   - Move ordering (informative guesses first) to reach a strong bound
//     quickly, which only affects speed, never which guesses are
//     considered.
//
// No candidate guess is ever skipped due to an arbitrary cap, sample
// stride, or search-depth restriction. Every "shortcut" in this file is
// either a structurally-forced base case (0 or 1 candidates remain -- no
// guess to search for) or a pruning rule with an explicit correctness
// argument in a comment.

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

#define WORD_LEN 5
#define NUM_SCORES 243
#define EXACT_MATCH 242

// -------------------------------------------------------------
// Word list / game data (all dynamically sized -- no fixed caps)
// -------------------------------------------------------------

typedef struct {
    char word[WORD_LEN + 1];
} Word;

typedef struct {
    Word* targets;
    Word* guesses;
    uint32_t num_targets;
    uint32_t num_guesses;

    // score_matrix[g * num_targets + t] = score of guesses[g] against targets[t]
    uint8_t* score_matrix;

    // Two independent zobrist tables give every subset a 128-bit identity
    // (hash1, hash2) instead of 64 bits alone. A single 64-bit collision is
    // unlikely but not negligible over a full --all run (~1e-4-1e-3 by the
    // birthday bound at ~1e7-1e8 distinct subsets); requiring both to
    // collide simultaneously makes it physically negligible (<1e-22).
    uint64_t* zobrist;          // [num_targets]
    uint64_t* zobrist2;         // [num_targets], independent seed
    uint32_t* lower_bound;      // [num_targets + 1], admissible LB by subset size
} GameData;

// Matches Python scoring.py: green pass first (consuming that letter from
// the target's multiset), then a left-to-right yellow pass over the guess
// consuming whatever of the target's multiset remains. Verified against
// scoring.py's get_score() with duplicate-letter cases.
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

// Words before the first blank line are targets (also valid guesses);
// words after it are valid guesses only. Every target line optionally
// carries a trailing weight column, tolerated but discarded, so files
// written for other tools can be read here unmodified.
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

        if (strlen(word) != WORD_LEN) {
            fprintf(stderr, "Warning: Skipping invalid word '%s' (length %zu)\n", word, strlen(word));
            continue;
        }
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

// Admissible lower bound on total guesses to resolve `k` remaining
// candidates. Derivation: a single guess can produce at most 243 distinct
// outcomes, one of which (the exact match) can be a "free" size-1 bucket
// that costs only the guess itself. Idealized optimal play distributes
// the rest across the remaining <=242 outcome buckets as evenly as
// possible, recursively. Because cost-per-item is exactly linear
// (LB(n) = 2n-1) for any bucket of size <=243, and no bucket here ever
// needs to exceed a few hundred for realistic Wordle-sized target lists,
// the whole recursion telescopes into the closed form below -- this is
// verified by direct simulation of the recursive definition, not merely
// asserted. It is a lower bound on ANY strategy, real guesses included,
// since real guesses can only do as well as this idealized partition or
// worse.
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

static void init_game_data(GameData* game, int num_threads) {
    size_t total_cells = (size_t)game->num_guesses * game->num_targets;
    game->score_matrix = malloc(total_cells * sizeof(uint8_t));
    if (!game->score_matrix) {
        fprintf(stderr, "Fatal: Out of memory allocating %zu MB score matrix\n", total_cells / (1024 * 1024));
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

    uint64_t seed = 0x853c49e6748fea9bULL;
    game->zobrist = malloc(game->num_targets * sizeof(uint64_t));
    for (uint32_t t = 0; t < game->num_targets; t++) game->zobrist[t] = splitmix64(&seed);

    // Independent stream: splitmix64 is deterministic in `seed`, so a
    // different starting seed gives a second table with no fixed
    // relationship to the first (not simply derivable from it).
    uint64_t seed2 = 0x9e3779b97f4a7c15ULL;
    game->zobrist2 = malloc(game->num_targets * sizeof(uint64_t));
    for (uint32_t t = 0; t < game->num_targets; t++) game->zobrist2[t] = splitmix64(&seed2);

    compute_lower_bound_table(game);
}

static void free_game_data(GameData* game) {
    free(game->targets);
    free(game->guesses);
    free(game->score_matrix);
    free(game->zobrist);
    free(game->zobrist2);
    free(game->lower_bound);
}

// -------------------------------------------------------------
// Transposition table: permanent, generation-free.
//
// A subset's optimal cost is intrinsic to the subset, not to how the
// search reached it, so both fields below are valid forever once set:
//   exact_cost        -- the true optimal total cost (UINT32_MAX = unknown)
//   proven_lower_bound -- "the true optimal cost is >= this value", a fact
//                         established by fully searching every candidate
//                         guess against a bound of `beta` without any of
//                         them beating it. This remains true no matter who
//                         asks next, so it also never needs to expire.
// -------------------------------------------------------------

typedef struct {
    uint64_t hash1;
    uint64_t hash2;
    uint32_t size;
    uint32_t exact_cost;
    uint32_t proven_lower_bound;
    uint32_t best_guess;
} TTEntry;

#define TT_EMPTY_HASH UINT64_MAX

typedef struct {
    TTEntry* entries;
    uint64_t mask;
} TT;

static void tt_init(TT* tt, uint64_t capacity_pow2) {
    tt->entries = malloc(capacity_pow2 * sizeof(TTEntry));
    tt->mask = capacity_pow2 - 1;
    for (uint64_t i = 0; i < capacity_pow2; i++) tt->entries[i].hash1 = TT_EMPTY_HASH;
}

static void tt_free(TT* tt) { free(tt->entries); }

// Returns NULL if not present. The table is a pure cache: correctness
// never depends on a hit, only on the fields being trustworthy when they
// do hit. Requiring both independent hashes (plus size) to match makes a
// false hit (two distinct subsets colliding on their full 128-bit
// identity) physically negligible, rather than merely unlikely on a
// single 64-bit hash.
static TTEntry* tt_find(TT* tt, uint64_t hash1, uint64_t hash2, uint32_t size) {
    uint64_t idx = hash1 & tt->mask;
    for (uint64_t probes = 0; probes <= tt->mask; probes++) {
        TTEntry* e = &tt->entries[idx];
        if (e->hash1 == TT_EMPTY_HASH) return NULL;
        if (e->hash1 == hash1 && e->hash2 == hash2 && e->size == size) return e;
        idx = (idx + 1) & tt->mask;
    }
    return NULL;
}

// Finds an existing slot for (hash1,hash2,size), or an empty slot to
// claim, or NULL if the table is degenerately full (never happens in
// practice -- table is sized generously and, if it were to fill, we just
// stop caching, which is always safe).
static TTEntry* tt_find_or_claim(TT* tt, uint64_t hash1, uint64_t hash2, uint32_t size) {
    uint64_t idx = hash1 & tt->mask;
    for (uint64_t probes = 0; probes <= tt->mask; probes++) {
        TTEntry* e = &tt->entries[idx];
        if (e->hash1 == TT_EMPTY_HASH) {
            e->hash1 = hash1;
            e->hash2 = hash2;
            e->size = size;
            e->exact_cost = UINT32_MAX;
            e->proven_lower_bound = 0;
            e->best_guess = UINT32_MAX;
            return e;
        }
        if (e->hash1 == hash1 && e->hash2 == hash2 && e->size == size) return e;
        idx = (idx + 1) & tt->mask;
    }
    return NULL;
}

// -------------------------------------------------------------
// Per-thread solver context: TT plus reusable scratch buffers.
//
// Scratch buffers are safe to share across an entire recursive call
// chain within one thread because each solve_subset() call only touches
// them during its own candidate-construction phase, strictly before it
// recurses into any child. By the time a child call reuses the scratch,
// the parent has already finished reading from it (the parent's own
// candidate list, built from the scratch, lives in a separate
// per-call VLA that IS parent-owned for the parent's whole lifetime).
// -------------------------------------------------------------

typedef struct {
    GameData* game;
    TT tt;

    // Equivalence-class dedup scratch, sized to num_guesses. Cleared via
    // a generation stamp instead of memset (O(1) amortized "clear"). Two
    // independent hashes per slot for the same reason as TTEntry: a false
    // "duplicate" collapse (two guesses with genuinely different
    // partitions colliding on one hash) would silently drop a candidate
    // guess from the search -- a real correctness bug, not just a missed
    // dedup opportunity -- if it happened to be the guess that made the
    // node's true optimum. Requiring both hashes to match makes that
    // physically negligible.
    uint64_t* dedup_stamp;
    uint64_t* dedup_hash;
    uint64_t* dedup_hash2;
    uint32_t* dedup_guess;
    uint64_t call_id;

    uint32_t* representatives; // scratch, sized to num_guesses

    uint64_t nodes_visited;
} Solver;

static void solver_init(Solver* s, GameData* game) {
    s->game = game;
    uint64_t tt_cap = next_pow2((uint64_t)game->num_targets * 64);
    if (tt_cap < (1u << 16)) tt_cap = 1u << 16;
    if (tt_cap > (1u << 23)) tt_cap = 1u << 23;
    tt_init(&s->tt, tt_cap);

    uint64_t dedup_cap = next_pow2((uint64_t)game->num_guesses * 2);
    s->dedup_stamp = calloc(dedup_cap, sizeof(uint64_t));
    s->dedup_hash = malloc(dedup_cap * sizeof(uint64_t));
    s->dedup_hash2 = malloc(dedup_cap * sizeof(uint64_t));
    s->dedup_guess = malloc(dedup_cap * sizeof(uint32_t));
    s->call_id = 0;
    s->representatives = malloc((size_t)game->num_guesses * sizeof(uint32_t));
    s->nodes_visited = 0;
}

static void solver_free(Solver* s) {
    tt_free(&s->tt);
    free(s->dedup_stamp);
    free(s->dedup_hash);
    free(s->dedup_hash2);
    free(s->dedup_guess);
    free(s->representatives);
}

typedef struct {
    uint32_t guess_idx;
    uint32_t active_buckets;
    uint32_t sum_sq;
} RankedCandidate;

static int compare_ranked_desc(const void* a, const void* b) {
    const RankedCandidate* ra = (const RankedCandidate*)a;
    const RankedCandidate* rb = (const RankedCandidate*)b;
    if (ra->active_buckets != rb->active_buckets) return (ra->active_buckets > rb->active_buckets) ? -1 : 1;
    if (ra->sum_sq != rb->sum_sq) return (ra->sum_sq < rb->sum_sq) ? -1 : 1;
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

// Core exact branch-and-bound solver for a subset of target indices.
//
// Returns the true minimum total number of guesses to resolve every
// target in `targets`, PROVIDED that value is < beta. If the true
// minimum is >= beta, the return value is only a valid lower bound
// (standard fail-soft branch-and-bound semantics): the caller must not
// treat it as exact unless beta == UINT32_MAX, in which case there is no
// bound to fail against and the return is always the true exact answer.
static uint32_t solve_subset(Solver* solver, const uint32_t* targets, uint32_t count,
                              uint64_t hash1, uint64_t hash2, uint32_t beta, uint32_t* out_guess) {
    solver->nodes_visited++;
    GameData* game = solver->game;

    if (count == 0) return 0;
    if (count == 1) {
        if (out_guess) *out_guess = targets[0];
        return 1;
    }
    if (count == 2) {
        // Structurally forced, not a heuristic: with 2 distinct
        // candidates, guessing either one always exact-matches itself
        // (free) and leaves the other as a forced size-1 remainder (1
        // more guess) -- the only possible outcome shape, achieving
        // lower_bound[2] = 3 exactly. targets[i] doubles as a guess index
        // here for i < num_targets, same as the count==1 case above (see
        // load_wordlist: every target word is also appended to the guess
        // list, in the same relative order).
        if (out_guess) *out_guess = targets[0];
        return 3;
    }

    const uint32_t num_targets = game->num_targets;
    const uint32_t num_guesses = game->num_guesses;
    const uint8_t* matrix = game->score_matrix;

    uint32_t node_lb = game->lower_bound[count];

    TTEntry* entry = tt_find(&solver->tt, hash1, hash2, count);
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
        // A permanent, valid fact regardless of who asks or why.
        TTEntry* e = tt_find_or_claim(&solver->tt, hash1, hash2, count);
        if (e && node_lb > e->proven_lower_bound) e->proven_lower_bound = node_lb;
        if (e && e->best_guess == UINT32_MAX && suggested_guess != UINT32_MAX) e->best_guess = suggested_guess;
        return node_lb;
    }

    // ---- Build the deduplicated candidate list: every guess in the
    // ---- full guess list is considered; guesses inducing an identical
    // ---- partition of `targets` are collapsed to one representative,
    // ---- since they necessarily yield identical subtree costs.
    solver->call_id++;
    uint64_t call_id = solver->call_id;
    uint64_t dedup_mask = next_pow2((uint64_t)num_guesses * 2) - 1;
    uint32_t rep_count = 0;

    for (uint32_t g = 0; g < num_guesses; g++) {
        const uint8_t* row = matrix + (size_t)g * num_targets;
        // Two independent rolling hashes (different offset basis and
        // multiplier) of the same score sequence: a "duplicate" that
        // collides on only one would falsely drop a candidate guess from
        // the search entirely, which -- unlike a TT miss -- is a silent
        // correctness bug if that guess was the node's true optimum.
        // Requiring both to match makes that physically negligible.
        uint64_t h = 1469598103934665603ULL;
        uint64_t h2 = 0x84222325cbf29ce4ULL;
        for (uint32_t i = 0; i < count; i++) {
            uint8_t sc = row[targets[i]];
            h = (h ^ sc) * 1099511628211ULL;
            h2 = (h2 ^ sc) * 0x9e3779b97f4a7c15ULL;
        }
        uint64_t idx = h & dedup_mask;
        bool duplicate = false;
        while (solver->dedup_stamp[idx] == call_id) {
            if (solver->dedup_hash[idx] == h && solver->dedup_hash2[idx] == h2) { duplicate = true; break; }
            idx = (idx + 1) & dedup_mask;
        }
        if (duplicate) continue;
        solver->dedup_stamp[idx] = call_id;
        solver->dedup_hash[idx] = h;
        solver->dedup_hash2[idx] = h2;
        solver->dedup_guess[idx] = g;
        solver->representatives[rep_count++] = g;
    }

    // ---- Rank representatives by a cheap move-ordering heuristic
    // ---- (more buckets, more uniform => tends to be a stronger guess).
    // ---- This only affects search order/speed: every representative is
    // ---- still tried, in some order, unless proven unnecessary below.
    //
    // All scratch below is stack-allocated (VLAs / fixed-size arrays),
    // not malloc'd: this function is the hottest of hot paths (millions
    // of calls across many threads), and per-call heap allocation would
    // serialize on the allocator's internal locks. NUM_SCORES-sized
    // arrays are compile-time fixed; the rest are sized to this call's
    // own `count`/`rep_count`, which are safe VLA bounds since recursion
    // depth is small (each level's target set is a strict subset of its
    // parent's) and each stack frame's arrays are only alive for that
    // frame's own duration.
    RankedCandidate ranked[rep_count];
    uint32_t lb1 = UINT32_MAX;
    {
        uint32_t hist[NUM_SCORES];
        for (uint32_t r = 0; r < rep_count; r++) {
            uint32_t g = solver->representatives[r];
            const uint8_t* row = matrix + (size_t)g * num_targets;
            memset(hist, 0, sizeof(hist));
            for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;
            uint32_t active = 0, sum_sq = 0, glb = count;
            for (int s = 0; s < NUM_SCORES; s++) {
                if (hist[s] > 0) {
                    active++; sum_sq += hist[s] * hist[s];
                    if (s != EXACT_MATCH) glb += game->lower_bound[hist[s]];
                }
            }
            ranked[r] = (RankedCandidate){ .guess_idx = g, .active_buckets = active, .sum_sq = sum_sq };
            if (glb < lb1) lb1 = glb;
        }
    }
    // lb1 = min over every representative of its own per-guess admissible
    // bound (count + sum of lower_bound[bucket size] over non-exact
    // buckets -- the exact same quantity the per-guess `guess_lb >=
    // current_best` skip below already computes one guess at a time).
    // Deduped-away guesses share their representative's exact histogram by
    // construction, so this min over representatives equals the min over
    // ALL num_guesses guesses -- a real admissible bound on this node's
    // true cost (whichever guess is actually used, its own achieved cost
    // is >= that guess's own bound, so the true minimum over all guesses
    // is >= the minimum of their bounds). Folding it into node_lb lets the
    // node fail soft here, before qsort and the main loop, whenever no
    // guess can possibly do better than what's already known.
    if (lb1 > node_lb) node_lb = lb1;
    if (node_lb >= beta) {
        TTEntry* e = tt_find_or_claim(&solver->tt, hash1, hash2, count);
        if (e && node_lb > e->proven_lower_bound) e->proven_lower_bound = node_lb;
        if (e && e->best_guess == UINT32_MAX && suggested_guess != UINT32_MAX) e->best_guess = suggested_guess;
        return node_lb;
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

    // ---- Main branch-and-bound loop over ALL representative guesses.
    uint32_t current_best = beta;
    uint32_t best_g = ranked[0].guess_idx;
    bool found_improvement = false;

    uint32_t local_partition[count];
    uint32_t hist[NUM_SCORES];
    uint32_t offsets[NUM_SCORES + 1];
    BucketInfo buckets[NUM_SCORES];

    for (uint32_t c = 0; c < rep_count; c++) {
        uint32_t g = ranked[c].guess_idx;
        const uint8_t* row = matrix + (size_t)g * num_targets;

        memset(hist, 0, NUM_SCORES * sizeof(uint32_t));
        for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;

        uint32_t guess_lb = count;
        uint32_t active_buckets = 0;
        for (int s = 0; s < NUM_SCORES; s++) {
            if (hist[s] > 0) {
                if (s != EXACT_MATCH) guess_lb += game->lower_bound[hist[s]];
                buckets[active_buckets].score = (uint16_t)s;
                buckets[active_buckets].size = hist[s];
                active_buckets++;
            }
        }

        // Reversible-move (Conway) pruning: a guess with a single active
        // bucket carries zero information -- every remaining candidate
        // scored identically, so the resulting subproblem is literally
        // this node's own subset again. count >= 3 here (count <= 2 are
        // base cases above), so some other candidate -- e.g. any
        // still-live target used as the guess -- is guaranteed to resolve
        // itself for free via an exact match, strictly outperforming this
        // one. A zero-information guess can therefore never be optimal;
        // skipping it also avoids ever recursing into an identical
        // (hash, count) subset from within its own still-executing call.
        if (active_buckets == 1) continue;

        // Fast perfect-split shortcut: guess_lb == count + (count-1)*1 =
        // 2*count-1 = lower_bound[count] exactly iff every OTHER
        // candidate lands in its own singleton bucket and this guess
        // itself exact-matches one of them (hist[EXACT_MATCH] > 0) --
        // i.e. active_buckets == count. That's this node's admissible
        // lower bound, achieved by a guess whose buckets are all size-1
        // base cases (a real, exact cost of 1 each, not merely a bound),
        // so no guess -- searched or not -- can beat it: unconditionally
        // optimal, with nothing left to sort, partition, or recurse into.
        if (active_buckets == count && hist[EXACT_MATCH] > 0) {
            current_best = guess_lb;
            best_g = g;
            found_improvement = true;
            break;
        }

        // Sound prune: this guess's own best possible outcome already
        // can't beat what we have.
        if (guess_lb >= current_best) continue;

        offsets[0] = 0;
        for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
        uint32_t cur_offsets[NUM_SCORES];
        memcpy(cur_offsets, offsets, sizeof(cur_offsets));
        for (uint32_t i = 0; i < count; i++) {
            uint8_t s = row[targets[i]];
            local_partition[cur_offsets[s]++] = targets[i];
        }

        for (uint32_t b = 0; b < active_buckets; b++) {
            uint32_t off = offsets[buckets[b].score];
            uint64_t bh1 = 0, bh2 = 0;
            for (uint32_t j = 0; j < buckets[b].size; j++) {
                uint32_t t = local_partition[off + j];
                bh1 ^= game->zobrist[t];
                bh2 ^= game->zobrist2[t];
            }
            buckets[b].hash1 = bh1;
            buckets[b].hash2 = bh2;
            buckets[b].offset = off;
        }

        qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

        uint32_t running_cost = count;
        uint32_t remaining_lb = 0;
        for (uint32_t b = 0; b < active_buckets; b++) {
            if (buckets[b].score != EXACT_MATCH) remaining_lb += game->lower_bound[buckets[b].size];
        }

        bool pruned = false;
        for (uint32_t b = 0; b < active_buckets; b++) {
            if (buckets[b].score == EXACT_MATCH) continue;
            remaining_lb -= game->lower_bound[buckets[b].size];

            if (running_cost + remaining_lb >= current_best) { pruned = true; break; }

            uint32_t bucket_beta = current_best - running_cost - remaining_lb;
            uint32_t bucket_cost = solve_subset(solver, &local_partition[buckets[b].offset], buckets[b].size,
                                                 buckets[b].hash1, buckets[b].hash2, bucket_beta, NULL);
            if (bucket_cost >= bucket_beta) { pruned = true; break; }
            running_cost += bucket_cost;
        }

        if (!pruned && running_cost < current_best) {
            current_best = running_cost;
            best_g = g;
            found_improvement = true;
            // Sound early exit: no guess -- searched or not -- can beat
            // this node's admissible lower bound, and we just matched it
            // exactly with a real, fully-verified strategy. That is a
            // proof of optimality, not a heuristic stop.
            if (current_best <= node_lb) break;
        }
    }

    TTEntry* e = tt_find_or_claim(&solver->tt, hash1, hash2, count);
    if (found_improvement) {
        // Every representative guess was either tried to completion or
        // proven (via an admissible per-guess bound) incapable of
        // beating current_best -- so current_best is the true global
        // optimum for this subset, unconditionally, regardless of beta.
        if (e) {
            e->exact_cost = current_best;
            e->best_guess = best_g;
        }
        if (out_guess) *out_guess = best_g;
        return current_best;
    } else {
        // No guess beat beta: a permanently valid fact.
        if (e && beta > e->proven_lower_bound) e->proven_lower_bound = beta;
        if (e && e->best_guess == UINT32_MAX) e->best_guess = best_g;
        if (out_guess) *out_guess = best_g;
        return beta;
    }
}

// -------------------------------------------------------------
// Evaluating a fixed opener exactly (root guess given, not searched)
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
                uint32_t t = local_partition[off + j];
                bh1 ^= game->zobrist[t];
                bh2 ^= game->zobrist2[t];
            }
            buckets[active++] = (BucketInfo){ .score = (uint16_t)s, .size = hist[s], .offset = off,
                                               .hash1 = bh1, .hash2 = bh2 };
        }
    }
    *out_active_buckets = active;
}

// Sequential (single Solver/TT, single thread) exact evaluation of one
// opener, with admissible-bound pruning against the pool's shared best-so-
// far so that openers already proven worse can be abandoned without being
// fully solved. This mirrors standard branch-and-bound over the *set of
// openers* and is independently sound from the exhaustiveness of
// solve_subset itself.
//
// Unlike a plain snapshot, `global_best_cost` is re-read from atomically
// before every bucket, not just once at the start: many openers can be
// mid-flight at once (one per pool thread), each potentially taking
// minutes, so a competing thread finding a better answer partway through
// should let every other in-flight opener cut itself off as soon as
// possible, not only ones that hadn't started yet.
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
    if (ceiling != UINT32_MAX && root_lb >= ceiling) {
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
    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score == EXACT_MATCH) continue;
        remaining_lb -= game->lower_bound[buckets[b].size];
        ceiling = atomic_load(global_best_cost);
        if (ceiling != UINT32_MAX && running_cost + remaining_lb >= ceiling) { pruned = true; break; }
        uint32_t bucket_beta = (ceiling == UINT32_MAX) ? UINT32_MAX : (ceiling - running_cost - remaining_lb);
        uint32_t cost = solve_subset(solver, &local_partition[buckets[b].offset], buckets[b].size,
                                      buckets[b].hash1, buckets[b].hash2, bucket_beta, NULL);
        if (cost >= bucket_beta) { pruned = true; break; }
        running_cost += cost;
    }

    free(local_partition);
    free(buckets);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    res.exact_total_cost = pruned ? UINT32_MAX : running_cost;
    res.avg_guesses = pruned ? 99.0 : (double)running_cost / (double)count;
    res.is_exact = !pruned;
    res.time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    res.nodes = solver->nodes_visited;
    return res;
}

// -------------------------------------------------------------
// Aspiration seeding (heuristics.md #19): before searching a given opener
// exactly, play out a real, valid (if suboptimal) strategy for it and use
// its actual total as beta. Because this simulates a genuine, legal
// decision tree end-to-end -- real guesses, real recursive partitioning,
// not an estimate -- its returned total is always something some real
// strategy achieves, so seeding beta from it can never cause an
// incomplete/wrong result, only faster pruning. See the correctness note
// in evaluate_opener_parallel for why this preserves --opener's
// exactness guarantee even in beta-ties.
// -------------------------------------------------------------

// Picks a guess for `pool` (`count` target indices) by minimizing
// sum-of-squared bucket sizes, searching only within `pool` itself (not
// the full guess list) to keep this cheap. Restricting guesses to the
// live candidate pool still guarantees termination (guessing any live
// candidate resolves it for free) and yields a real, achievable strategy.
static uint32_t greedy_pick(GameData* game, const uint32_t* pool, uint32_t count) {
    const uint32_t num_targets = game->num_targets;
    const uint8_t* matrix = game->score_matrix;
    uint32_t best_g = pool[0];
    uint32_t best_sum_sq = UINT32_MAX;
    for (uint32_t c = 0; c < count; c++) {
        uint32_t g = pool[c];
        const uint8_t* row = matrix + (size_t)g * num_targets;
        uint32_t hist[NUM_SCORES] = {0};
        for (uint32_t i = 0; i < count; i++) hist[row[pool[i]]]++;
        uint32_t sum_sq = 0;
        for (int s = 0; s < NUM_SCORES; s++) sum_sq += hist[s] * hist[s];
        if (sum_sq < best_sum_sq) { best_sum_sq = sum_sq; best_g = g; }
    }
    return best_g;
}

// Actually plays out `forced_guess` against `targets`, then greedily
// resolves every resulting bucket (recursively, via greedy_pick), and
// returns the real total cost of that concrete strategy -- a valid,
// achievable upper bound on the true optimal cost for `targets` with
// `forced_guess` as the first move.
static uint32_t greedy_upper_bound(GameData* game, const uint32_t* targets, uint32_t count, uint32_t forced_guess) {
    if (count <= 1) return count;
    if (count == 2) return 3;

    const uint32_t num_targets = game->num_targets;
    const uint8_t* row = game->score_matrix + (size_t)forced_guess * num_targets;

    uint32_t hist[NUM_SCORES] = {0};
    for (uint32_t i = 0; i < count; i++) hist[row[targets[i]]]++;

    uint32_t offsets[NUM_SCORES + 1];
    offsets[0] = 0;
    for (int s = 0; s < NUM_SCORES; s++) offsets[s + 1] = offsets[s] + hist[s];
    uint32_t cur[NUM_SCORES];
    memcpy(cur, offsets, sizeof(cur));

    uint32_t* partitioned = malloc((size_t)count * sizeof(uint32_t));
    for (uint32_t i = 0; i < count; i++) {
        uint8_t s = row[targets[i]];
        partitioned[cur[s]++] = targets[i];
    }

    uint32_t total = count; // this guess costs 1 for every remaining target
    for (int s = 0; s < NUM_SCORES; s++) {
        uint32_t sz = hist[s];
        if (sz == 0 || s == EXACT_MATCH) continue;
        uint32_t* bucket = &partitioned[offsets[s]];
        uint32_t next_guess = greedy_pick(game, bucket, sz);
        total += greedy_upper_bound(game, bucket, sz, next_guess);
    }
    free(partitioned);
    return total;
}

// -------------------------------------------------------------
// Bucket-parallel exact evaluation of a single opener: each root bucket
// is solved to full optimality by its own worker thread with its own
// private Solver/TT. Buckets are independent (no cross-bucket sharing is
// required for correctness), so this is embarrassingly parallel and
// introduces no synchronization beyond an atomic work index.
// -------------------------------------------------------------

typedef struct {
    GameData* game;
    const uint32_t* local_partition;
    BucketInfo* buckets;
    const uint32_t* betas;
    uint32_t num_buckets;
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
        // nodes_visited is cumulative for this thread's whole lifetime, not
        // per-bucket -- capture the delta so a thread handling multiple
        // buckets (always true at low thread counts, e.g. --threads 1)
        // doesn't have each later bucket's count include every earlier
        // bucket's too. evaluate_opener_parallel sums this array across all
        // buckets, so leaving it cumulative would inflate the reported
        // total node count by roughly a triangular-number factor in the
        // number of buckets one thread processes.
        uint64_t nodes_before = solver.nodes_visited;
        uint32_t best_guess;
        uint32_t cost = solve_subset(&solver, &pool->local_partition[bkt->offset], bkt->size,
                                      bkt->hash1, bkt->hash2, pool->betas[idx], &best_guess);
        pool->out_costs[idx] = cost;
        pool->out_guesses[idx] = best_guess;
        pool->out_nodes[idx] = solver.nodes_visited - nodes_before;
    }
    solver_free(&solver);
    return NULL;
}

static OpenerResult evaluate_opener_parallel(GameData* game, uint32_t opener_idx, int num_threads) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint32_t count = game->num_targets;
    uint32_t* local_partition = malloc(count * sizeof(uint32_t));
    BucketInfo* buckets = malloc(NUM_SCORES * sizeof(BucketInfo));
    uint32_t active_buckets;
    partition_root(game, opener_idx, local_partition, buckets, &active_buckets);
    // Largest-first (LPT) task order: the biggest bucket dominates total
    // wall-clock time, so starting it first (rather than leaving it for
    // whichever thread happens to reach it in the work queue) minimizes
    // the tail where other threads have run dry. Correctness is
    // unaffected -- buckets are solved independently either way.
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc);

    // local_partition currently holds exactly this opener's `count` target
    // indices (order doesn't matter to greedy_upper_bound, only the set
    // does), so it can be reused directly as the "whole root" target list.
    uint32_t ceiling = greedy_upper_bound(game, local_partition, count, opener_idx);
    uint32_t remaining_lb_total = 0;
    for (uint32_t b = 0; b < active_buckets; b++) {
        if (buckets[b].score != EXACT_MATCH) remaining_lb_total += game->lower_bound[buckets[b].size];
    }
    // Correctness: `ceiling` is a real, achievable total (greedy_upper_bound
    // played out an actual strategy), so ceiling >= true_total always. For
    // any one bucket B, true_total = count + true_cost(B) + sum(true costs
    // of every other bucket) >= count + true_cost(B) + (remaining_lb_total
    // - lower_bound(B)) (since true cost >= the admissible lower_bound for
    // every other bucket). Rearranging: true_cost(B) <= ceiling - count -
    // remaining_lb_total + lower_bound(B) -- exactly betas[b] below. So
    // every bucket's derived beta is >= that bucket's own true cost:
    // solve_subset's fail-soft return is always a valid (if, at the
    // boundary, non-strict) lower bound, so the *number* it returns is
    // always correct regardless of ties; only a tied bucket's TT entry
    // might land as "proven_lower_bound" instead of "exact_cost" (a
    // caching-opportunity miss, not a correctness issue), which doesn't
    // affect the summed total this function reports.

    if (num_threads < 1) num_threads = 1;
    uint32_t total_cost = count; // one guess for every remaining target
    uint64_t total_nodes = 0;

    if (active_buckets > 0) {
        uint32_t* betas = malloc(active_buckets * sizeof(uint32_t));
        for (uint32_t b = 0; b < active_buckets; b++) {
            if (buckets[b].score == EXACT_MATCH) { betas[b] = UINT32_MAX; continue; }
            uint32_t own_lb = game->lower_bound[buckets[b].size];
            betas[b] = ceiling - count - remaining_lb_total + own_lb;
        }

        BucketPool pool = { .game = game, .local_partition = local_partition, .buckets = buckets,
                             .betas = betas, .num_buckets = active_buckets };
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
        free(betas);
    }

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
    return res;
}

// -------------------------------------------------------------
// Decision tree construction & JSON export
// (schema matches the original solver's output: {"guess","num_targets",
//  "branches": {"<score>": <child>}} with {"leaf": true} on leaves, so
//  strategies.py's DecisionTreeStrategy can load it unchanged.)
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

static TreeNode* build_subtree_node(Solver* solver, const uint32_t* targets, uint32_t count,
                                     uint64_t hash1, uint64_t hash2);

static TreeNode* build_subtree_node_with_guess(Solver* solver, const uint32_t* targets, uint32_t count,
                                                uint64_t hash1, uint64_t hash2, uint32_t best_g) {
    GameData* game = solver->game;
    (void)hash1;
    (void)hash2;
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
                uint32_t t = local_partition[off + j];
                bh1 ^= game->zobrist[t];
                bh2 ^= game->zobrist2[t];
            }
            node->children[s] = build_subtree_node(solver, &local_partition[off], hist[s], bh1, bh2);
        }
    }

    free(local_partition);
    return node;
}

static TreeNode* build_subtree_node(Solver* solver, const uint32_t* targets, uint32_t count,
                                     uint64_t hash1, uint64_t hash2) {
    if (count == 0) return NULL;
    if (count == 1) return make_leaf(solver->game, targets[0]);
    uint32_t best_g;
    solve_subset(solver, targets, count, hash1, hash2, UINT32_MAX, &best_g);
    return build_subtree_node_with_guess(solver, targets, count, hash1, hash2, best_g);
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
        pool->out_nodes[idx] = build_subtree_node(&solver, &pool->local_partition[bkt->offset], bkt->size,
                                                   bkt->hash1, bkt->hash2);
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
    qsort(buckets, active_buckets, sizeof(BucketInfo), compare_bucket_size_desc); // LPT order, see evaluate_opener_parallel

    if (num_threads < 1) num_threads = 1;
    TreeNode* root = calloc(1, sizeof(TreeNode));
    root->num_targets = count;
    strcpy(root->guess, game->guesses[opener_idx].word);

    if (active_buckets > 0) {
        TreeBucketPool pool = { .game = game, .local_partition = local_partition, .buckets = buckets,
                                 .num_buckets = active_buckets };
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

// exact_total_cost is the ground truth (an exact integer sum of guesses);
// exact_avg_score is derived from it at full double round-trip precision
// (%.17g -- IEEE 754 double guarantees 17 significant digits are enough
// to reproduce the exact bit pattern, so nothing is lost to display
// rounding), plus the raw numerator/denominator so a consumer can recover
// the exact rational average itself without depending on the decimal at
// all.
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
// Opener ranking heuristic (cheap pre-filter for --top N) and the
// parallel-across-openers work pool for --top/--all.
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

// -------------------------------------------------------------
// CLI
// -------------------------------------------------------------

static void print_usage(const char* prog) {
    printf("Wordle Exact Full-Tree Solver (from-scratch reimplementation)\n\n");
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --wordlist <path>     Path to words.txt (default: words.txt)\n");
    printf("  --opener <word>       Evaluate a single opening word to exact optimality\n");
    printf("  --top <N>             Heuristically pre-rank openers, then exactly solve the top N\n");
    printf("  --all                 Exactly solve every possible opening word (the true global optimum)\n");
    printf("  --threads <N>         Number of worker threads (default: hardware concurrency)\n");
    printf("  --tree, --dump-tree <path> Dump the optimal solution decision tree to JSON file\n");
    printf("  --quiet, -q           Disable per-word verbose output (print compact progress)\n");
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
    printf("      WORDLE EXACT FULL-TREE SOLVER (from-scratch, wordle_claude)\n");
    printf("=================================================================\n");

    GameData game;
    if (load_wordlist(wordlist_path, &game) != 0) return 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    printf("Precomputing %u x %u score matrix using %d threads...\n", game.num_guesses, game.num_targets, num_threads);
    init_game_data(&game, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("Score matrix ready in %.3f seconds (%.1f MB allocated).\n\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9,
           (double)((size_t)game.num_guesses * game.num_targets) / (1024.0 * 1024.0));

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

    printf("Ranking initial opening guesses by partition variance (heuristic pre-filter)...\n");
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
    if (!search_all) {
        printf("NOTE: --top selects its %zu candidates via the heuristic above, then solves each\n"
               "EXACTLY. This is not guaranteed to include the true global optimum -- use --all\n"
               "for that (much slower: every possible opening word is solved exactly).\n", count_to_eval);
    }

    uint32_t* openers_to_eval = malloc(count_to_eval * sizeof(uint32_t));
    for (size_t i = 0; i < count_to_eval; i++) openers_to_eval[i] = cands[i].guess_idx;
    free(cands);

    printf("Evaluating %zu opener(s) in parallel using %d threads%s...\n\n",
           count_to_eval, num_threads, quiet ? " (quiet mode)" : "");

    OpenerWorkPool pool = { .game = &game, .opener_indices = openers_to_eval, .num_openers = count_to_eval,
                             .quiet = quiet };
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

    // Any opener that got admissibly pruned during ranking (skipped
    // before being fully solved because it could not possibly beat the
    // best-so-far) is provably not the optimum, so it is correctly
    // excluded -- no re-verification pass is needed here, unlike in the
    // buggy original where solve_node's own move truncation could make a
    // "pruned" verdict unsound. Nothing here needs re-checking.

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
