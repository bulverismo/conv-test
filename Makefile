.PHONY= build debug run clean vimutils

CC = gcc

SOURCES = lenet_inference.c

# compiler flags:
#  -Wall turns on most, but not all, compiler warnings
CFLAGS := -Wall 

# example using an lib
# CFLAGS := -Wall -lraylib

# the build target executable:
TARGET ?= main

VIMSPECTOR_FILE = .vimspector.json
COMPILE_COM_FILE = compile_commands.json

build: $(TARGET) vimutils

%: %.c
	$(CC) -o $(TARGET) $(TARGET).c $(SOURCES) $(CFLAGS)

$(VIMSPECTOR_FILE): vimspector.mock
	sed 's/BINARY/$(TARGET)/g' $< > $@

$(COMPILE_COM_FILE):
	compiledb make

vimutils: $(COMPILE_COM_FILE) $(VIMSPECTOR_FILE) 

debug: *.c
	$(CC) -g -o $(TARGET) $(TARGET).c $(SOURCES) $(CFLAGS) -DDEBUG

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET) $(VIMSPECTOR_FILE) $(COMPILE_COM_FILE)
