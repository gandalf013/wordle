CC = clang
CFLAGS = -O3 -march=native -flto -Wall -Wextra -pthread -D_GNU_SOURCE
TARGET = wordle_claude
SRCS = wordle_claude.c

# Sanitizer builds for solver/test_solver.py: -O1 -g keeps stack traces
# useful without -O0's slowdown. ASan and TSan instrumentation are mutually
# exclusive, so they're separate binaries; UBSan rides along with ASan.
# -fno-sanitize-recover=all makes the first violation fatal (nonzero exit)
# instead of printing and continuing, so a failing run can't be missed by
# only checking the exit code.
SANFLAGS = -O1 -g -fno-omit-frame-pointer -Wall -Wextra -pthread -D_GNU_SOURCE -fno-sanitize-recover=all

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm

asan: wordle_claude_asan
wordle_claude_asan: $(SRCS)
	$(CC) $(SANFLAGS) -fsanitize=address,undefined -o wordle_claude_asan $(SRCS) -lm

tsan: wordle_claude_tsan
wordle_claude_tsan: $(SRCS)
	$(CC) $(SANFLAGS) -fsanitize=thread -o wordle_claude_tsan $(SRCS) -lm

sanitizers: asan tsan

clean:
	rm -f $(TARGET) wordle_claude_asan wordle_claude_tsan

.PHONY: all asan tsan sanitizers clean
