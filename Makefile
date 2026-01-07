.PHONY: build up down up-gpu

build:
	docker build . -t trebi
up:
	docker run -d \
	    --gpus "device=0" \
	    --name trebi \
	    trebi tail -f /dev/null
	docker exec -it trebi bash
down:
	docker stop trebi
	docker rm trebi


