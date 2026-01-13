.PHONY: build up attach down tag push pull

# ==== CONFIG ====
DOCKER_USER ?= ghcr.io/uncovsky
IMAGE_NAME  ?= trebi
TAG         ?= latest

APP_IMAGE   := $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)
CONTAINER   := trebi

# ==== BUILD ====
build:
	docker build -t $(APP_IMAGE) .

# ==== TAG (optional if build already uses full tag) ====
tag:
	docker tag $(IMAGE_NAME):$(TAG) $(APP_IMAGE)

push: build
	docker push $(APP_IMAGE)

pull:
	docker pull $(APP_IMAGE)

# ==== RUN ====
up:
	docker run -d \
	  --gpus "device=0" \
	  --name $(CONTAINER) \
	  -v "$(PWD)":/workspace \
	  -e WANDB_API_KEY=$$WANDB_API_KEY \
	  $(APP_IMAGE) tail -f /dev/null

attach:
	docker exec -it $(CONTAINER) bash

down:
	docker stop $(CONTAINER) || true
	docker rm $(CONTAINER) || true
