CC ?= cc
CFLAGS ?= -O3 -march=native -flto -Wall -Wextra -pthread -D_GNU_SOURCE
TARGET = wordle_gemini
SRCS = wordle_gemini.c

# Shared-library extension differs by platform (macOS: dylib, Linux/BSD: so).
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_EXT := dylib
else
SHARED_EXT := so
endif
LIBTARGET := libwordle_gemini.$(SHARED_EXT)

SANFLAGS = -O1 -g -fno-omit-frame-pointer -Wall -Wextra -pthread -D_GNU_SOURCE -fno-sanitize-recover=all

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm

lib: $(LIBTARGET)
$(LIBTARGET): $(SRCS)
	$(CC) $(CFLAGS) -shared -fPIC -o $(LIBTARGET) $(SRCS) -lm

asan: wordle_gemini_asan
wordle_gemini_asan: $(SRCS)
	$(CC) $(SANFLAGS) -fsanitize=address,undefined -o wordle_gemini_asan $(SRCS) -lm

tsan: wordle_gemini_tsan
wordle_gemini_tsan: $(SRCS)
	$(CC) $(SANFLAGS) -fsanitize=thread -o wordle_gemini_tsan $(SRCS) -lm

sanitizers: asan tsan

profile: wordle_gemini_profile
wordle_gemini_profile: $(SRCS)
	$(CC) -O3 -march=native -g -fno-omit-frame-pointer -Wall -Wextra -pthread -D_GNU_SOURCE -o wordle_gemini_profile $(SRCS) -lm

clean:
	rm -f $(TARGET) wordle_gemini_asan wordle_gemini_tsan wordle_gemini_profile $(LIBTARGET)

.PHONY: all lib asan tsan sanitizers profile clean
