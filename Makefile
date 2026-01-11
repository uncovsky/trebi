PHONY: build build-osrl up attach down datasets clean

DATA_VOLUME := datasets
APP_IMAGE := trebi
OSRL_IMAGE := osrl
CONTAINER := trebi

build:
	docker build -t $(APP_IMAGE) .

build-osrl:
	docker build -f OSRL_Dockerfile -t $(OSRL_IMAGE) .

# Create dataset volume if it doesn't exist
datasets:
	docker volume inspect $(DATA_VOLUME) >/dev/null 2>&1 || \
	docker volume create $(DATA_VOLUME)

# Download datasets using OSRL
prepare-datasets: build-osrl datasets
	docker run --rm \
		-v $(DATA_VOLUME):/data \
		$(OSRL_IMAGE) \
		python download_datasets.py

# Run main app container
up: build datasets
	docker run -d \
		--gpus "device=0" \
		--name $(CONTAINER) \
		-v $(PWD):/workspace \
		-v $(DATA_VOLUME):/data \
		$(APP_IMAGE) tail -f /dev/null
	docker exec -it $(CONTAINER) bash

attach:
	docker exec -it $(CONTAINER) bash

down:
	docker stop $(CONTAINER) || true
	docker rm $(CONTAINER) || true

clean:
	docker volume rm $(DATA_VOLUME) || true
