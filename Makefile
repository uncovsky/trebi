PHONY: build up attach down

APP_IMAGE := trebi
CONTAINER := trebi

build:
	docker build -t $(APP_IMAGE) .

# Run main app container
up: build
	docker run -d \
	  --gpus "device=0" \
	  --name $(CONTAINER) \
	  -v "$(PWD)":/workspace \
	  $(APP_IMAGE) tail -f /dev/null

	docker exec -it $(CONTAINER) bash

attach:
	docker exec -it $(CONTAINER) bash

down:
	docker stop $(CONTAINER) || true
	docker rm $(CONTAINER) || true
