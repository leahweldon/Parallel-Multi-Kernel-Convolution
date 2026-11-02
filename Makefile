CC = gcc
CFLAGS = -O3 -msse4 -fopenmp -Wall -Wextra
LDLIBS = -lm
TARGET = conv-harness
SRC = src/conv-harness.c

.PHONY: all test clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDLIBS)

test: all
	bash scripts/smoke-test.sh

clean:
	rm -f $(TARGET)
