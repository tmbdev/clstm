TAG=kbai/clstm
FOLDERS = $(shell find . -mindepth 1 -maxdepth 1 -type d)

.PHONY: $(FOLDERS)
all: $(FOLDERS)

$(FOLDERS): % : %/Dockerfile
	cd "$@" && docker build -t "$(TAG):$@" .
