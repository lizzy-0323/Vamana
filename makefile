
# build go lib
GO_BUILD := go build -o vamana go_api/vamana_go_api.go

# build c lib
CC_BUILD := mkdir -p build && cd build && cmake .. && make

all: go-build cc-build

go-build: cc-build
	$(GO_BUILD)

cc-build:
	$(CC_BUILD)

.PHONY: all go-build cc-build




	
	
